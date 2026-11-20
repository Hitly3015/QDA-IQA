import os
import random
import numpy as np
import torch
import torch.nn.functional as TF  
from PIL import Image
import clip
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from utils.kadid10k_distortions import imdist_generator, to_uint8
import pyiqa
import warnings
warnings.filterwarnings("ignore", message=".*Conversion from CIE-LAB.*")
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

KADIS_REF_DIR = "./data/kadis700k/kadis700k/ref_imgs"
NUM_REF_IMAGES = 100 

ACTIVE_IMG_DIR = "./data/BID/ImageDatabase"
ACTIVE_MOS_PATH = "./datasets_label/BID_labels.txt"


SUPPORT_RATIO = 0.1 
GAMMA = 2.0  


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


def get_kadis_unsupervised_clusters(kadis_ref_dir, num_ref=100, cache_dir="./cache_kadis", seed=42, top_k=50, worst_k=50):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"kadis_clusters_ref{num_ref}.pt")

    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location='cpu')
        return data["F_best"].to(device), data["F_worst"].to(device)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')
    all_ref_files = sorted([
        os.path.join(kadis_ref_dir, f) for f in os.listdir(kadis_ref_dir) 
        if f.endswith(valid_exts)
    ])

    rng = random.Random(seed)
    selected_refs = rng.sample(all_ref_files, min(num_ref, len(all_ref_files)))
    F_best_list = []
    F_worst_list = []

    for p in tqdm(selected_refs, desc=""):
        try:
            img_pil = Image.open(p).convert("RGB")
            w, h = img_pil.size
            min_side = min(w, h)
            if min_side > 448:
                scale = 448 / min_side
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                f_orig = model.encode_image(img_tensor)
                f_orig = f_orig / (f_orig.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            candidate_pils = []
            for d in range(1, 26):
                for level in range(1, 6):
                    dist_arr = imdist_generator(img_pil, d, level, seed=seed)
                    candidate_pils.append(Image.fromarray(to_uint8(dist_arr)))

            cand_feats = []
            for cpil in candidate_pils:
                img_tensor = preprocess(cpil).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model.encode_image(img_tensor)
                    f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
                    cand_feats.append(f)
            
            cand_feats = torch.cat(cand_feats, dim=0)
            visual_fidelity = (cand_feats @ f_orig.t()).squeeze(-1).cpu().numpy()
            sorted_indices = np.argsort(visual_fidelity)

            best_idx = sorted_indices[-top_k:]
            F_best_list.append(cand_feats[best_idx].cpu())
            F_best_list.append(f_orig.cpu())

            worst_idx = sorted_indices[:worst_k]
            F_worst_list.append(cand_feats[worst_idx].cpu())

            torch.cuda.empty_cache()
        except Exception as e:
            continue

    F_best_all = torch.cat(F_best_list, dim=0)
    F_worst_all = torch.cat(F_worst_list, dim=0)

    F_best = F_best_all.mean(dim=0)
    F_worst = F_worst_all.mean(dim=0)

    torch.save({"F_best": F_best, "F_worst": F_worst}, cache_path)
    return F_best.to(device), F_worst.to(device)


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

    print(f"\n{'='*70}")
    print(f"实验: {name}")
    print(f"样本数: {len(preds)}")

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


F_best, F_worst = get_kadis_unsupervised_clusters(
    KADIS_REF_DIR, 
    num_ref=NUM_REF_IMAGES
)
V_q = F_best - F_worst
V_q = V_q / (V_q.norm() + 1e-12)


raw_paths, raw_mos = load_dataset(ACTIVE_IMG_DIR, ACTIVE_MOS_PATH)

dataset_tag = os.path.basename(ACTIVE_MOS_PATH).replace(".txt", "")
cache_active_dir = "./.cache_active"
os.makedirs(cache_active_dir, exist_ok=True)
cache_active_path = os.path.join(cache_active_dir, f"{dataset_tag}_all_feats.pt")

if os.path.exists(cache_active_path):

    active_cache = torch.load(cache_active_path, map_location='cpu')
    all_feats = active_cache["feats"]
    valid_mos = active_cache["mos"]
    valid_paths = active_cache["paths"]
else:
    all_feats, valid_idx = extract_features(raw_paths, f"{dataset_tag}")
    all_feats = all_feats.float()
    valid_mos = raw_mos[valid_idx]
    valid_paths = [raw_paths[i] for i in valid_idx]
    torch.save({"feats": all_feats, "mos": valid_mos, "paths": valid_paths}, cache_active_path)


num_total = len(valid_paths)
indices = list(range(num_total))

random.seed(42)
random.shuffle(indices)

num_support = int(num_total * SUPPORT_RATIO)
support_indices = indices[:num_support]
test_indices = indices[num_support:]

support_feats = all_feats[support_indices]
support_mos = valid_mos[support_indices]

test_feats = all_feats[test_indices]
test_mos = valid_mos[test_indices]



T_good = encode_text_prompts([
'Good image', 
'Sharp image', 
'sharp edges',
'High resolution image',
    'Noise-free image',
])
T_bad = encode_text_prompts([
    'bad image', 
    'blurry image', 
    'blurry edges',
    'low resolution image', 
    'noisy image',
])


iaa_keywords = ["ava", "aadb", "baid", "para", "tad66k"]
is_iaa = any(kw in ACTIVE_MOS_PATH.lower() or kw in ACTIVE_IMG_DIR.lower() for kw in iaa_keywords)

V_a = None
F_ae_best, F_ae_worst = None, None

if is_iaa and len(support_indices) > 0:

    sorted_sup_indices = np.argsort(support_mos)
    k_ae = max(5, min(100, len(sorted_sup_indices) // 6)) 
    
    worst_ae_idx = sorted_sup_indices[:k_ae]
    best_ae_idx = sorted_sup_indices[-k_ae:]
    
    F_ae_best = support_feats[best_ae_idx].mean(dim=0).to(device)
    F_ae_worst = support_feats[worst_ae_idx].mean(dim=0).to(device)
    
    V_a = F_ae_best - F_ae_worst
    V_a = V_a / (V_a.norm() + 1e-12)
    
    T_dir = T_good - T_bad
    T_dir = T_dir / (T_dir.norm() + 1e-12)
    align_score_ae = torch.dot(V_a.cpu(), T_dir).item()
    if align_score_ae < 0:
        V_a = -V_a
        F_ae_best, F_ae_worst = F_ae_worst, F_ae_best  # 极性反转
   


if  V_a is not None:
    V_q_final = V_q.cpu() + GAMMA * V_a.cpu()
    V_q_final = V_q_final / (V_q_final.norm() + 1e-12)
    
    F_best_final = F_best.cpu() + GAMMA * F_ae_best.cpu()
    F_worst_final = F_worst.cpu() + GAMMA * F_ae_worst.cpu()
    F_best_final = F_best_final / (F_best_final.norm() + 1e-12)
    F_worst_final = F_worst_final / (F_worst_final.norm() + 1e-12)
else:
    
    V_q_final = V_q.cpu()
    F_best_final = F_best.cpu()
    F_worst_final = F_worst.cpu()


T_dir = T_good - T_bad
T_dir = T_dir / (T_dir.norm() + 1e-12)
align_score = torch.dot(V_q_final.float(), T_dir).item()
if align_score < 0:
    V_q_final = -V_q_final
    align_score = -align_score



TEMP   = 0.5 
ALPHA  = 2.0  

T_good_calib, T_bad_calib, attn_map = calip_text_calibration(
    T_good, T_bad, F_best_final, F_worst_final,
    lambda_val=ALPHA, temp=TEMP
)


pred_text_only, _, _ = predict_text_only(test_feats, T_good, T_bad)
evaluate(pred_text_only, test_mos,
         "Baseline")


pred_ablation_B, _, S_good_B, S_bad_B = predict_residual_enhanced_then_text(
    test_feats, V_q_final, T_good_calib, T_bad_calib, alpha=ALPHA
)
evaluate(pred_ablation_B, test_mos,
         f"Ours")
