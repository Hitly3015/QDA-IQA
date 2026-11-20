import os
import numpy as np
import torch
import torch.nn.functional as TF  
from PIL import Image
import clip
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_dataset(img_dir, mos_path):
    paths, scores = [], []
    with open(mos_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                name, mos = parts[0], float(parts[-1])
                p = os.path.join(img_dir, name)
                if os.path.exists(p):
                    paths.append(p)
                    scores.append(mos)
    return paths, np.array(scores)

kadid_img_dir = "./data/kadid10k/images"
kadid_mos_path = "./datasets_label/kadid10k_label.txt"


img_dir = "./data/koniq10k/1024x768"
mos_path = "./datasets_label/koniq10k_label.txt"


K = 2000
paths, mos = load_dataset(img_dir, mos_path)
model, preprocess = clip.load("RN50", device=device)
model.eval()


def extract_features(paths, desc=""):
    feats = []
    valid_indices = []  
    for idx, p in enumerate(tqdm(paths, desc=desc)):
        try:
            img_pil = Image.open(p)
            w, h = img_pil.size
            min_side = min(w, h)
            if min_side > 448:
                scale = 448 / min_side
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            img = preprocess(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(img)
                f = f / f.norm(p=2, dim=-1, keepdim=True)
                feats.append(f.cpu())
            valid_indices.append(idx) 
            torch.cuda.empty_cache()
        except Exception as e:
            continue  
    return torch.cat(feats, dim=0), valid_indices  


def encode_text_prompts(prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
    feats = feats.mean(dim=0)
    feats = feats / (feats.norm() + 1e-12)
    return feats.cpu().float()


def calip_text_calibration(T_good, T_bad, F_best, F_worst, lambda_val=0.1, temp=1.0):
    T = torch.stack([T_good, T_bad], dim=0).float()
    F_anchor = torch.stack([F_best, F_worst], dim=0).float()

    T = TF.normalize(T, dim=-1)
    F_anchor = TF.normalize(F_anchor, dim=-1)

    S = torch.matmul(T, F_anchor.t()) / temp
    P = TF.softmax(S, dim=-1)

    T_hat = torch.matmul(P, F_anchor)

    T_calib = T + lambda_val * T_hat
    T_calib = TF.normalize(T_calib, dim=-1)

    return T_calib[0], T_calib[1], P


def logistic_func(X, beta1, beta2, beta3, beta4, beta5):
    X = np.asarray(X, dtype=np.float64)
    z = np.clip(beta2 * (X - beta3), -50, 50)
    logisticpart = 0.5 - 1.0 / (1.0 + np.exp(z))
    return beta1 * logisticpart + beta4 * X + beta5


def evaluate(preds, mos, name):
    preds = np.asarray(preds, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)

    mask = np.isfinite(preds) & np.isfinite(mos)
    preds = preds[mask]
    mos = mos[mask]


    srcc, _ = spearmanr(preds, mos)
    plcc_raw, _ = pearsonr(preds, mos)

    try:
        beta_init = [
            np.max(mos) - np.min(mos),
            10.0,
            np.mean(preds),
            1.0,
            1.0
        ]
        popt, _ = curve_fit(
            logistic_func, preds, mos,
            p0=beta_init, maxfev=10000
        )
        preds_fit = logistic_func(preds, *popt)
        plcc_fit, _ = pearsonr(preds_fit, mos)
    except Exception:
        plcc_fit = plcc_raw

    print(f"SRCC      : {srcc:.4f}")
    print(f"PLCC      : {plcc_fit:.4f}")
    print(f"{'='*70}")

    return {"srcc": srcc, "plcc_fit": plcc_fit}


def predict_text_only(feats, T_good, T_bad):
    S_good = feats.float() @ T_good.float()
    S_bad  = feats.float() @ T_bad.float()
    return (S_good - S_bad).numpy(), S_good.numpy(), S_bad.numpy()


def predict_projected_component_then_text(feats, V_q, T_good, T_bad, normalize_proj=True):
    q = feats.float() @ V_q.float()
    F_proj = torch.outer(q, V_q.float())
    if normalize_proj:
        F_proj = F_proj / (F_proj.norm(dim=-1, keepdim=True) + 1e-12)
    S_good = F_proj @ T_good.float()
    S_bad  = F_proj @ T_bad.float()
    return (S_good - S_bad).numpy(), q.numpy()


def predict_residual_enhanced_then_text(feats, V_q, T_good, T_bad, alpha=1.5):
    q = feats.float() @ V_q.float()
    F_res = torch.outer(q, V_q.float())
    F_hat = feats.float() + alpha * F_res
    F_hat = F_hat / (F_hat.norm(dim=-1, keepdim=True) + 1e-12)
    S_good = F_hat @ T_good.float()
    S_bad  = F_hat @ T_bad.float()
    return (S_good - S_bad).numpy(), q.numpy(), S_good.numpy(), S_bad.numpy()
    

cache_kadid_dir = "./cache_kadid"
os.makedirs(cache_kadid_dir, exist_ok=True)
cache_kadid_path = os.path.join(cache_kadid_dir, "kadid_cache.pt")

if os.path.exists(cache_kadid_path):
    kadid_cache = torch.load(cache_kadid_path, map_location='cpu')
    kadid_feats = kadid_cache["feats"].float()
    kadid_mos = kadid_cache["mos"]
else:
    kadid_paths, kadid_mos_all = load_dataset(kadid_img_dir, kadid_mos_path)
    kadid_feats, valid_idx = extract_features(kadid_paths, "KADID")
    kadid_mos = kadid_mos_all[valid_idx]
    torch.save({
        "feats": kadid_feats.float(),
        "mos": kadid_mos 
    }, cache_kadid_path)

feats, valid_idx = extract_features(paths, "testset")
feats = feats.float()

mos = mos[valid_idx]

sorted_idx = np.argsort(kadid_mos)
F_best  = kadid_feats[sorted_idx[-K:]].mean(dim=0)
F_worst = kadid_feats[sorted_idx[:K]].mean(dim=0)

V_q = F_best - F_worst
V_q = V_q / (V_q.norm() + 1e-12)


T_good = encode_text_prompts([
    'Good image', 'Sharp image', 'sharp edges',
    'High resolution image', 'Noise-free image',
])
T_bad = encode_text_prompts([
    'bad image', 'blurry image', 'blurry edges',
    'low resolution image', 'noisy image',
])

T_dir = T_good - T_bad
T_dir = T_dir / (T_dir.norm() + 1e-12)
align_score = torch.dot(V_q.float(), T_dir).item()
if align_score < 0:
    V_q = -V_q
    align_score = -align_score


# LAMBDA = 2.0    
TEMP   = 0.5 
ALPHA  = 2.0  

T_good_calib, T_bad_calib, attn_map = calip_text_calibration(
    T_good, T_bad, F_best, F_worst,
    lambda_val=ALPHA, temp=TEMP
)


pred_text_only, _, _ = predict_text_only(feats, T_good, T_bad)
evaluate(pred_text_only, mos,
         "Baseline")

pred_ablation_B, _, S_good_B, S_bad_B = predict_residual_enhanced_then_text(
    feats, V_q, T_good_calib, T_bad_calib, alpha=ALPHA
)
evaluate(pred_ablation_B, mos,
         "Ours")

