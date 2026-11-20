import os
import argparse
import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
import clip
import random
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from utils.kadid10k_distortions import imdist_generator, to_uint8


def parse_args():
    parser = argparse.ArgumentParser(
        description="KADIS-anchored training-free IQA evaluation (BID / KADIS reference images)"
    )

    # Paths
    parser.add_argument("--kadis_ref_dir", type=str, default="./data/kadis700k/kadis700k/ref_imgs",
                        help="directory of KADIS reference images used to build anchors")
    parser.add_argument("--test_img_dir", type=str, default="./data/BID/ImageDatabase",
                        help="directory of test images")
    parser.add_argument("--test_mos_path", type=str, default="./datasets_label/BID_labels.txt",
                        help="test label file (name + score per line)")
    parser.add_argument("--cache_dir", type=str, default="./cache_kadis",
                        help="directory for cached anchor features")

    # Model / inference
    parser.add_argument("--model", type=str, default="RN50", choices=clip.available_models(),
                        help="CLIP backbone to load")
    parser.add_argument("--device", type=str, default=None,
                        help="inference device (default: cuda if available, else cpu)")
    parser.add_argument("--resize_max", type=int, default=448,
                        help="downscale so that the shorter side is at most this value")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of images per forward pass")

    # Anchor construction (KADIS distortion clustering)
    parser.add_argument("--num_ref", type=int, default=100,
                        help="number of KADIS reference images to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reference sampling and distortion generation")
    parser.add_argument("--top_k", type=int, default=50,
                        help="per-reference candidates kept for the good anchor")
    parser.add_argument("--worst_k", type=int, default=50,
                        help="per-reference candidates kept for the bad anchor")

    # Text calibration / residual enhancement
    parser.add_argument("--lambda_val", type=float, default=2.0,
                        help="weight of the anchor-attn term inside text calibration")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="softmax temperature of the text-anchor attention")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="scale of the residual-enhancement term")

    # Text anchors
    parser.add_argument("--good_prompts", type=str, nargs="+",
                        default=["Good image", "Sharp image", "sharp edges",
                                 "High resolution image", "Noise-free image"],
                        help="text prompts for the good-quality anchor")
    parser.add_argument("--bad_prompts", type=str, nargs="+",
                        default=["bad image", "blurry image", "blurry edges",
                                 "low resolution image", "noisy image"],
                        help="text prompts for the bad-quality anchor")

    return parser.parse_args()


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


def normalize_min_side(img_pil, resize_max):
    w, h = img_pil.size
    min_side = min(w, h)
    if min_side > resize_max:
        scale = resize_max / min_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img_pil


def extract_features(paths, model, preprocess, device, resize_max=448, batch_size=1, desc=""):
    feats = []
    valid_indices = []
    buffer, buffer_idx = [], []

    def flush():
        if not buffer:
            return
        batch = torch.stack(buffer, dim=0).to(device)
        with torch.no_grad():
            f = model.encode_image(batch)
            f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        feats.append(f.cpu())
        valid_indices.extend(buffer_idx)
        buffer.clear()
        buffer_idx.clear()
        torch.cuda.empty_cache()

    for start in tqdm(range(0, len(paths), batch_size), desc=desc):
        for idx in range(start, min(start + batch_size, len(paths))):
            p = paths[idx]
            try:
                img_pil = Image.open(p)
                img_pil = normalize_min_side(img_pil, resize_max)
                buffer.append(preprocess(img_pil))
                buffer_idx.append(idx)
            except Exception:
                continue
        if batch_size > 1 or start + batch_size >= len(paths):
            flush()

    return torch.cat(feats, dim=0), valid_indices


def get_kadis_unsupervised_clusters(kadis_ref_dir, model, preprocess, device,
                                    num_ref=100, cache_dir="./cache_kadis", seed=42,
                                    top_k=50, worst_k=50, resize_max=448):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"kadis_clusters_ref{num_ref}.pt")

    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
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

    for p in tqdm(selected_refs):
        try:

            img_pil = Image.open(p).convert("RGB")
            img_pil = normalize_min_side(img_pil, resize_max)

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
            print(f"{p}: {e}")
            continue

    F_best_all = torch.cat(F_best_list, dim=0)
    F_worst_all = torch.cat(F_worst_list, dim=0)

    F_best = F_best_all.mean(dim=0)
    F_worst = F_worst_all.mean(dim=0)

    torch.save({
        "F_best": F_best,
        "F_worst": F_worst
    }, cache_path)

    return F_best.to(device), F_worst.to(device)


def encode_text_prompts(prompts, model, device):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
    feats = feats.mean(dim=0)
    feats = feats / (feats.norm() + 1e-12)
    return feats.cpu().float()


def calip_text_calibration(T_good, T_bad, F_best, F_worst, lambda_val=2.0, temp=0.5):
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


def evaluate(preds, mos):
    preds = np.asarray(preds, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)

    mask = np.isfinite(preds) & np.isfinite(mos)
    preds = preds[mask]
    mos = mos[mask]

    print(f"\n{'='*70}")

    print(f"Samples: {len(preds)}")

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


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # Anchors from unsupervised KADIS distortion clustering
    F_best, F_worst = get_kadis_unsupervised_clusters(
        args.kadis_ref_dir, model, preprocess, device,
        num_ref=args.num_ref, cache_dir=args.cache_dir, seed=args.seed,
        top_k=args.top_k, worst_k=args.worst_k, resize_max=args.resize_max
    )

    # Quality direction spanned by the two anchors
    V_q = F_best - F_worst
    V_q = V_q / (V_q.norm() + 1e-12)

    # Test set
    test_paths, test_mos_all = load_dataset(args.test_img_dir, args.test_mos_path)
    feats, valid_idx = extract_features(
        test_paths, model, preprocess, device,
        resize_max=args.resize_max, batch_size=args.batch_size, desc=""
    )
    feats = feats.float()
    mos = test_mos_all[valid_idx]

    # Text anchors
    T_good = encode_text_prompts(args.good_prompts, model, device)
    T_bad = encode_text_prompts(args.bad_prompts, model, device)

    # Sign fix: align visual quality direction with the textual good-bad direction
    T_dir = T_good - T_bad
    T_dir = T_dir / (T_dir.norm() + 1e-12)
    align_score = torch.dot(V_q.float().cpu(), T_dir).item()
    if align_score < 0:
        V_q = -V_q
        align_score = -align_score
    print(f"align_score: {align_score:.4f}")

    # Text calibration and residual-enhanced scoring
    T_good_calib, T_bad_calib, attn_map = calip_text_calibration(
        T_good, T_bad, F_best.cpu(), F_worst.cpu(),
        lambda_val=args.lambda_val, temp=args.temp
    )

    pred_text_only, _, _ = predict_text_only(feats, T_good, T_bad)
    evaluate(pred_text_only, mos)

    pred_ablation_B, _, S_good_B, S_bad_B = predict_residual_enhanced_then_text(
        feats, V_q.cpu(), T_good_calib, T_bad_calib, alpha=args.alpha
    )
    evaluate(pred_ablation_B, mos)


if __name__ == "__main__":
    main()
