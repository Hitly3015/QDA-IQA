#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist
from scipy.optimize import curve_fit

import clip


warnings.filterwarnings("ignore", message=".*Conversion from CIE-LAB.*")
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
Image.MAX_IMAGE_PIXELS = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CLIP_BACKBONE = "RN50"

REF_RATIO       = 0.10     
CLUSTER_RATIO   = 0.25      
MIN_REF         = 30        
MIN_CLUSTER     = 8        
MIN_REF_GROUPS  = 3        
MIN_TEST_GROUPS = 2       

N_SPLITS        = 15        
BASE_SEED       = 42   
SEEDS           = [BASE_SEED + i for i in random.sample(range(1, 10001), NUM_RUNS -1)]
SPLIT_MODE      = "stratified"  


SUBSPACE_MODE   = "mos"     # 'mos' | 'synthetic' | 'both'


SYN_MAX_REF     = 60       
SYN_DIST_TYPES  = list(range(1, 26))
SYN_LEVELS      = [1, 2, 3, 4, 5]
SYN_TOPK        = 50
SYN_WORSTK      = 50
SYN_USE_PRISTINE = True     


ALPHA = 2.0                
TEMP  = 0.5                

# ---------- 跨域对比 (原 KADIS 方案) ----------
COMPARE_WITH_KADIS = True
KADIS_REF_DIR      = "./data/kadis700k/kadis700k/ref_imgs"
KADIS_NUM_REF      = 100

FEAT_CACHE_DIR  = "./.cache_feats"
KADIS_CACHE_DIR = "./.cache_kadis"
OUT_DIR         = "./results_indomain_seeds"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 1

BASELINE_NAME = "M0"
METRICS = ["srcc", "plcc_raw", "plcc_fit"]



DATASETS = {
   
    "KonIQ10k": dict(img_dir="./data/koniq10k/1024x768",
                     label="./datasets_label/koniq10k_label.txt",
                     polarity=+1, group_by=None),
    "LIVEC":    dict(img_dir="./data/ChallengeDB_release/Images",
                     label="./datasets_label/livec_label.txt",
                     polarity=+1, group_by=None),
    "BID":      dict(img_dir="./data/BID/ImageDatabase",
                     label="./datasets_label/BID_labels.txt",
                     polarity=+1, group_by=None),
    "SPAQ":     dict(img_dir="./data/SPAQ/TestImage",
                     label="./datasets_label/spaq_label.txt",
                     polarity=+1, group_by=None),
    "FLIVE":    dict(img_dir="./data/FLIVE_Database/database",
                     label="./datasets_label/flive_label.txt",
                     polarity=+1, group_by=None),

    
    "KADID10k": dict(img_dir="./data/kadid10k/images",
                     label="./datasets_label/kadid10k_label.txt",
                     polarity=+1, group_by='prefix_us',
                     pristine_dir="./data/kadid10k/images"),
    "TID2013":  dict(img_dir="./data/tid2013/distorted_images",
                     label="./datasets_label/tid2013_label.txt",
                     polarity=+1, group_by='prefix_us',
                     pristine_dir="./data/tid2013/reference_images"),
    "CSIQ":     dict(img_dir="./data/CSIQ/dst_imgs",
                     label="./datasets_label/CSIQ_label.txt",
                     polarity=-1, group_by='first_dot',
                     pristine_dir="./data/CSIQ/src_imgs"),
    "LIVE":     dict(img_dir="./data/LIVEIQA_release2",
                     label="./datasets_label/live_label.txt",
                     polarity=-1, group_by='live_mat',
                     live_ref_mat="./data/LIVEIQA_release2/refnames_all.mat",
                     pristine_dir="./data/LIVEIQA_release2/refimgs"),
    "PIPAL":    dict(img_dir="./data/PIPAL/Dist_Imgs",
                     label="./datasets_label/pipal22_train.txt",
                     polarity=+1, group_by='prefix_us',
                     pristine_dir="./data/PIPAL/Train_Ref"),


    "AGIQA3K":  dict(img_dir="./data/AGIQA_3K",
                     label="./datasets_label/AGI3k_labels.txt",
                     polarity=+1, group_by=None),
    "AGIQA1K":  dict(img_dir="./data/AGIQA-1k/file",
                     label="./datasets_label/AGI1k_labels.txt",
                     polarity=+1, group_by=None),
    "AVA":      dict(img_dir="./data/AVA_dataset/ava_images",
                     label="./datasets_label/ava_labels_1w.txt",
                     polarity=+1, group_by=None),
    "AADB":     dict(img_dir="./data/AADB_aesthetic_IQA_eccv16/datasetImages_originalSize/",
                     label="./datasets_label/AADB_labels.txt",
                     polarity=+1, group_by=None),
    "TAD66K":   dict(img_dir="./data/TAD66k",
                     label="./datasets_label/TDA66K_test_labels.txt",
                     polarity=+1, group_by=None),
    "PARA":     dict(img_dir="./data/PARA/imgs",
                     label="./datasets_label/para_iaa_labels.txt",
                     polarity=+1, group_by=None),
    "BAID":     dict(img_dir="./data/BAID/images",
                     label="./datasets_label/BAID_test_labels.txt",
                     polarity=+1, group_by=None),
}


ACTIVE_DATASETS = ["KonIQ10k", "KADID10k", "PIPAL", "AGIQA3K", "AVA"]
# ACTIVE_DATASETS = None



model, preprocess = clip.load(CLIP_BACKBONE, device=device)
model.eval()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_dataset(img_dir, mos_path):
    """返回 (paths, scores)"""
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
    return paths, np.array(scores, dtype=np.float64)


def _load_and_resize(p):
    img = Image.open(p).convert("RGB")
    w, h = img.size
    ms = min(w, h)
    if ms > 448:
        s = 448.0 / ms
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    return img


@torch.no_grad()
def encode_pils(pils, batch_size=BATCH_SIZE):
   
    out = []
    for i in range(0, len(pils), batch_size):
        batch = torch.stack([preprocess(x) for x in pils[i:i + batch_size]]).to(device)
        f = model.encode_image(batch)
        f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        out.append(f.cpu().float())
    return torch.cat(out, dim=0)


def extract_features(paths, desc="", batch_size=BATCH_SIZE):
    
    feats, valid = [], []
    buf, buf_idx = [], []

    def flush():
        nonlocal buf, buf_idx
        if not buf:
            return
        with torch.no_grad():
            batch = torch.stack(buf).to(device)
            f = model.encode_image(batch)
            f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        feats.append(f.cpu().float())
        valid.extend(buf_idx)
        buf, buf_idx = [], []
        torch.cuda.empty_cache()

    for idx, p in enumerate(tqdm(paths, desc=desc)):
        try:
            buf.append(preprocess(_load_and_resize(p)))
            buf_idx.append(idx)
        except Exception:
            continue
        if len(buf) >= batch_size:
            flush()
    flush()

    if len(feats) == 0:
        return torch.zeros(0, 1024), []
    return torch.cat(feats, dim=0), valid


def get_dataset_features(key, cfg):
    
    os.makedirs(FEAT_CACHE_DIR, exist_ok=True)
    tag = CLIP_BACKBONE.replace('/', '-')
    cache = os.path.join(FEAT_CACHE_DIR, f"{key}_{tag}_feats.pt")

    if os.path.exists(cache):
        d = torch.load(cache, map_location='cpu', weights_only=False)
        print(f"[缓存读取] {key}: {d['feats'].shape[0]} 张图像特征 <- {cache}")
        return d['feats'].float(), d['scores'], d['paths']

    paths, scores = load_dataset(cfg['img_dir'], cfg['label'])
    if len(paths) == 0:
        raise RuntimeError(f"{key}: 未找到任何有效图像, 请检查路径 {cfg['img_dir']} / {cfg['label']}")
    feats, valid = extract_features(paths, desc=f"[{key}] 特征提取")
    scores = scores[valid]
    paths = [paths[i] for i in valid]
    torch.save({'feats': feats, 'scores': scores, 'paths': paths}, cache)
    print(f"[缓存保存] {key}: {feats.shape[0]} 张 -> {cache}")
    return feats.float(), scores, paths



def _rel_name(p, img_dir):
    try:
        return os.path.relpath(p, img_dir)
    except Exception:
        return os.path.basename(p)


def _g_prefix_us(rel):      # I01_01_05.png -> i01     (KADID / TID2013 / PIPAL)
    return os.path.basename(rel).split('_')[0].lower()


def _g_first_dot(rel):      # 1600.AWGN.1.png -> 1600  (CSIQ)
    return os.path.basename(rel).split('.')[0].lower()


def _g_parent_dir(rel):     # subdir/xxx.png -> subdir
    return os.path.dirname(rel).lower()


_GROUP_FNS = {'prefix_us': _g_prefix_us, 'first_dot': _g_first_dot, 'parent_dir': _g_parent_dir}


def _live_group_map(cfg):
   
    try:
        from scipy.io import loadmat
        refs = [str(r[0]) for r in loadmat(cfg['live_ref_mat'])['refnames_all'][0]]
    except Exception as e:
    
        return {}
    rels = []
    with open(cfg['label'], 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rels.append(line.replace(',', ' ').split()[0])
    if len(rels) != len(refs):
     
        return {}
    return {r: os.path.splitext(os.path.basename(g))[0].lower() for r, g in zip(rels, refs)}


def compute_groups(paths, cfg):

    gb = cfg.get('group_by', None)
    if gb is None:
        return np.array([f"__img{i}" for i in range(len(paths))])

    gmap = _live_group_map(cfg) if gb == 'live_mat' else None
    fn = gb if callable(gb) else _GROUP_FNS.get(gb if isinstance(gb, str) else None)

    out = []
    for i, p in enumerate(paths):
        rel = _rel_name(p, cfg['img_dir'])
        if gmap is not None:
            out.append(gmap.get(rel, f"__img{i}"))
        elif fn is not None:
            out.append(fn(rel))
        else:
            out.append(f"__img{i}")
    return np.array(out)


def resolve_pristine_paths(group_ids, cfg):
    d = cfg.get('pristine_dir')
    if not d or not os.path.isdir(d):
        return None
    exts = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.PNG', '.BMP', '.JPG', '.JPEG', '.TIF')
    out = []
    for g in group_ids:
        g = str(g)
        if g.startswith('__img'):
            continue
        cands = [g, g.upper(), g.lower(), g[:1].upper() + g[1:]]
        found = None
        for c in cands:
            for e in exts:
                fp = os.path.join(d, c + e)
                if os.path.exists(fp):
                    found = fp
                    break
            if found:
                break
        if found:
            out.append(found)
    return out if len(out) > 0 else None



def encode_text_prompts(prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
    feats = feats.mean(dim=0)
    feats = feats / (feats.norm() + 1e-12)
    return feats.cpu().float()


def calip_text_calibration(T_good, T_bad, F_best, F_worst, lambda_val=0.1, temp=1.0):
   
    T = TF.normalize(torch.stack([T_good, T_bad], dim=0).float(), dim=-1)
    F_anchor = TF.normalize(torch.stack([F_best, F_worst], dim=0).float(), dim=-1)

    S = torch.matmul(T, F_anchor.t()) / temp
    P = TF.softmax(S, dim=-1)
    T_hat = torch.matmul(P, F_anchor)

    T_calib = TF.normalize(T + lambda_val * T_hat, dim=-1)
    return T_calib[0], T_calib[1], P



def build_subspace_from_mos(feats_ref, q_ref, cluster_ratio=CLUSTER_RATIO, min_cluster=MIN_CLUSTER):
    
    n = len(q_ref)
    k = int(round(n * cluster_ratio))
    k = max(min(min_cluster, n // 2), k)
    k = max(1, min(k, n // 2))

    order = np.argsort(q_ref)          
    low_idx = order[:k]
    high_idx = order[-k:]

    F_worst = TF.normalize(feats_ref[low_idx].mean(dim=0), dim=-1)
    F_best = TF.normalize(feats_ref[high_idx].mean(dim=0), dim=-1)
    return F_best, F_worst, k


def build_subspace_from_synthetic(ref_paths, seed=42,
                                  dist_types=SYN_DIST_TYPES, levels=SYN_LEVELS,
                                  top_k=SYN_TOPK, worst_k=SYN_WORSTK,
   
    from utils.kadid10k_distortions import imdist_generator, to_uint8

    rng = random.Random(seed)
    use_paths = ref_paths if len(ref_paths) <= max_ref else rng.sample(list(ref_paths), max_ref)

    F_best_list, F_worst_list = [], []
    for p in tqdm(use_paths, desc="  域内无监督两极簇"):
        try:
            img_pil = _load_and_resize(p)
            f_orig = encode_pils([img_pil])                       # [1,D]

            cand_pils = []
            for d in dist_types:
                for lv in levels:
                    arr = imdist_generator(img_pil, d, lv, seed=seed)
                    cand_pils.append(Image.fromarray(to_uint8(arr)))

            cand_feats = encode_pils(cand_pils)                   # [125,D]
            fidelity = (cand_feats @ f_orig.t()).squeeze(-1).numpy()
            order = np.argsort(fidelity)                          

            F_best_list.append(cand_feats[order[-top_k:]])
            F_best_list.append(f_orig)
            F_worst_list.append(cand_feats[order[:worst_k]])
            torch.cuda.empty_cache()
        except Exception as e:
            
            continue


    F_best = TF.normalize(torch.cat(F_best_list, 0).mean(dim=0), dim=-1)
    F_worst = TF.normalize(torch.cat(F_worst_list, 0).mean(dim=0), dim=-1)
    return F_best, F_worst


def get_kadis_unsupervised_clusters(kadis_ref_dir, num_ref=100, cache_dir=KADIS_CACHE_DIR,
                                    seed=42, top_k=50, worst_k=50):
   
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"kadis_clusters_ref{num_ref}.pt")
    if os.path.exists(cache_path):
      
        d = torch.load(cache_path, map_location='cpu', weights_only=False)
        return TF.normalize(d["F_best"].float(), dim=-1), TF.normalize(d["F_worst"].float(), dim=-1)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')
    all_ref = sorted([os.path.join(kadis_ref_dir, f) for f in os.listdir(kadis_ref_dir)
                      if f.endswith(valid_exts)])
    rng = random.Random(seed)
    sel = rng.sample(all_ref, min(num_ref, len(all_ref)))
    F_best, F_worst = build_subspace_from_synthetic(sel, seed=seed, top_k=top_k,
                                                    worst_k=worst_k, max_ref=num_ref)
    torch.save({"F_best": F_best, "F_worst": F_worst}, cache_path)
    print(f"[缓存保存] KADIS 跨域锚点 -> {cache_path}")
    return F_best, F_worst


def align_direction(V_q, T_dir):
    
    s = torch.dot(V_q.float(), T_dir.float()).item()
    if s < 0:
        V_q = -V_q
        s = -s
    return V_q, s



def predict_text_only(feats, T_good, T_bad):
    Sg = feats.float() @ T_good.float()
    Sb = feats.float() @ T_bad.float()
    return (Sg - Sb).numpy()


def predict_projection(feats, V_q):
    return (feats.float() @ V_q.float()).numpy()


def predict_projected_component_then_text(feats, V_q, T_good, T_bad, normalize_proj=True):
    q = feats.float() @ V_q.float()
    F_proj = torch.outer(q, V_q.float())
    if normalize_proj:
        F_proj = F_proj / (F_proj.norm(dim=-1, keepdim=True) + 1e-12)
    return (F_proj @ T_good.float() - F_proj @ T_bad.float()).numpy()


def predict_residual_enhanced_then_text(feats, V_q, T_good, T_bad, alpha=ALPHA):
    q = feats.float() @ V_q.float()
    F_hat = feats.float() + alpha * torch.outer(q, V_q.float())
    F_hat = F_hat / (F_hat.norm(dim=-1, keepdim=True) + 1e-12)
    return (F_hat @ T_good.float() - F_hat @ T_bad.float()).numpy()



def logistic_func(X, beta1, beta2, beta3, beta4, beta5):
    X = np.asarray(X, dtype=np.float64)
    z = np.clip(beta2 * (X - beta3), -50, 50)
    return beta1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + beta4 * X + beta5


def evaluate(preds, mos, name="", verbose=False):
    preds = np.asarray(preds, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)
    m = np.isfinite(preds) & np.isfinite(mos)
    preds, mos = preds[m], mos[m]

    if len(preds) < 5 or np.std(preds) < 1e-12:
        return {"srcc": np.nan, "plcc_raw": np.nan, "plcc_fit": np.nan, "n": len(preds)}

    srcc, _ = spearmanr(preds, mos)
    plcc_raw, _ = pearsonr(preds, mos)
    try:
        p0 = [np.max(mos) - np.min(mos), 10.0, float(np.mean(preds)), 1.0, 1.0]
        popt, _ = curve_fit(logistic_func, preds, mos, p0=p0, maxfev=10000)
        plcc_fit, _ = pearsonr(logistic_func(preds, *popt), mos)
    except Exception:
        plcc_fit = plcc_raw

    if verbose:
        print(f"  {name:<46s} SRCC={srcc:+.4f}  PLCC_raw={plcc_raw:+.4f}  "
              f"PLCC_fit={plcc_fit:+.4f}  (n={len(preds)})")
    return {"srcc": srcc, "plcc_raw": plcc_raw, "plcc_fit": plcc_fit, "n": len(preds)}


def summarize_values(vals):

    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {k: np.nan for k in
                ["n", "mean", "std", "var", "min", "max", "range", "iqr", "median", "cv", "ci95"]}

    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    var = float(v.var(ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(v, [25, 75])
    tcrit = float(t_dist.ppf(0.975, n - 1)) if n > 1 else 0.0
    ci95 = tcrit * std / np.sqrt(n) if n > 1 else 0.0

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "var": var,
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
        "iqr": float(q3 - q1),
        "median": float(np.median(v)),
        "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan,
        "ci95": float(ci95),
    }



def split_ref_test(q_scores, groups=None, ref_ratio=REF_RATIO, seed=0, mode=SPLIT_MODE,
                   min_ref=MIN_REF, min_ref_groups=MIN_REF_GROUPS,
                   min_test_groups=MIN_TEST_GROUPS):
  
    n = len(q_scores)
    if groups is None:
        groups = np.array([f"__img{i}" for i in range(n)])

    uniq, inv = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    g_mean = np.array([q_scores[inv == i].mean() for i in range(n_g)])

    target_imgs = max(min_ref, int(round(n * ref_ratio)))
    avg_per_g = n / float(n_g)
    n_ref_g = int(np.ceil(target_imgs / avg_per_g))
    n_ref_g = max(n_ref_g, min(min_ref_groups, max(1, n_g - min_test_groups)))
    n_ref_g = int(np.clip(n_ref_g, 1, max(1, n_g - min_test_groups)))

    rng = np.random.RandomState(seed)
    if mode == "random" or n_g < 3:
        sel = rng.permutation(n_g)[:n_ref_g]
    else:
        order = np.argsort(g_mean)                      
        bins = np.array_split(order, n_ref_g)
        sel = np.array([b[rng.randint(len(b))] for b in bins if len(b) > 0])

    m = np.isin(inv, sel)
    return np.where(m)[0], np.where(~m)[0]



def run_predictors(feats_test, V_q, T_good, T_bad, T_good_c, T_bad_c, alpha=ALPHA, prefix=""):
    P = {}
    P[f"{prefix}q = f·V_q"] = predict_projection(feats_test, V_q)

    P[f"{prefix}Ours (α={alpha}) × 校准文本"] = predict_residual_enhanced_then_text(
        feats_test, V_q, T_good_c, T_bad_c, alpha)
    return P


# =====================================================================================
#                            结果打印 & 保存
# =====================================================================================
def print_mean_std_table(key, acc, n_g, grouped):
    print("\n" + "-" * 96)
    print(f"{key}  mean ± std over {N_SPLITS} splits (seeds {SEEDS[0]}–{SEEDS[-1]}, "
          f"极性已校正, 数值越大越好)")
    print(f"划分粒度: {'参考内容级(无内容泄漏)' if grouped else '图像级'} | 内容数={n_g}")
    print("-" * 96)
    print(f"{'Method':<48s}{'SRCC':>18s}{'PLCC(fit)':>18s}")
    print("-" * 96)
    for name, rs in acc.items():
        sr = np.array([r['srcc'] for r in rs], dtype=np.float64)
        pl = np.array([r['plcc_fit'] for r in rs], dtype=np.float64)
        print(f"{name:<48s}"
              f"{np.nanmean(sr):>9.4f}±{np.nanstd(sr, ddof=1):<7.4f}"
              f"{np.nanmean(pl):>9.4f}±{np.nanstd(pl, ddof=1):<7.4f}")
    print("-" * 96)


def print_stability_table(key, summary, metric):
    print(f"\n[{key}] {metric.upper()} 稳定性统计 ({N_SPLITS} 次划分, std/var 为样本口径 ddof=1)")
    head = (f"{'Method':<48s}{'Mean':>9}{'Std':>9}{'Var':>11}{'Min':>9}{'Max':>9}"
            f"{'Range':>9}{'IQR':>9}{'CV%':>8}{'CI95':>9}")
    print(head)
    print("-" * len(head))
    for name, st in summary.items():
        s = st[metric]
        cv = s["cv"] * 100 if np.isfinite(s["cv"]) else np.nan
        print(f"{name:<48s}{s['mean']:>9.4f}{s['std']:>9.4f}{s['var']:>11.3e}"
              f"{s['min']:>9.4f}{s['max']:>9.4f}{s['range']:>9.4f}"
              f"{s['iqr']:>9.4f}{cv:>8.2f}{s['ci95']:>9.4f}")
    print("-" * len(head))


def print_paired_table(key, acc, metric="srcc"):
    if BASELINE_NAME not in acc:
        return
    base = np.array([r[metric] for r in acc[BASELINE_NAME]], dtype=np.float64)

    print(f"\n[{key}] 相对基线 {BASELINE_NAME} 的配对差值 ({metric.upper()}, 同一划分下逐次相减)")
    head = (f"{'Method':<48s}{'ΔMean':>9}{'ΔStd':>9}{'ΔMin':>9}{'ΔMax':>9}"
            f"{'Win':>5}{'Tie':>5}{'Loss':>5}")
    print(head)
    print("-" * len(head))
    for name, rs in acc.items():
        if name == BASELINE_NAME:
            continue
        cur = np.array([r[metric] for r in rs], dtype=np.float64)
        d = cur - base
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        wins = int((d > 1e-9).sum())
        ties = int((np.abs(d) <= 1e-9).sum())
        losses = int((d < -1e-9).sum())
        print(f"{name:<48s}{d.mean():>+9.4f}{d.std(ddof=1):>9.4f}"
              f"{d.min():>+9.4f}{d.max():>+9.4f}{wins:>5}{ties:>5}{losses:>5}")
    print("-" * len(head))



def main():
    set_seed(BASE_SEED)

   
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

  
    V_q_kadis, F_best_k, F_worst_k = None, None, None
    if COMPARE_WITH_KADIS:
        try:
            F_best_k, F_worst_k = get_kadis_unsupervised_clusters(KADIS_REF_DIR, num_ref=KADIS_NUM_REF)
            V_q_kadis = TF.normalize(F_best_k - F_worst_k, dim=-1)
            V_q_kadis, s = align_direction(V_q_kadis, T_dir)
            print(f"[KADIS 跨域] cos(V_q, T_good-T_bad) = {s:.4f}")
        except Exception as e:
            V_q_kadis = None

    keys = ACTIVE_DATASETS if ACTIVE_DATASETS else list(DATASETS.keys())
    all_rows = []
    all_payload = OrderedDict()

    for key in keys:
        cfg = DATASETS.get(key)
        if cfg is None:
            print(f"[skip]: {key}")
            continue
        if not (os.path.exists(cfg['img_dir']) and os.path.exists(cfg['label'])):
            print(f"[skip]: {key}")
            continue

        print("\n" + "#" * 92)
        print(f"#dataset: {key}   (polarity={cfg['polarity']:+d},split {int(REF_RATIO*100)}% ref : "
              f"{100-int(REF_RATIO*100)}% test, {N_SPLITS} 次)")
        print("#" * 92)

        try:
            feats, scores, paths = get_dataset_features(key, cfg)
        except Exception as e:
            continue

     
        q_all = cfg['polarity'] * scores

 
        groups = compute_groups(paths, cfg)
        n_g = len(np.unique(groups))
        grouped = not str(groups[0]).startswith('__img')
      

        ref0, _ = split_ref_test(q_all, groups, seed=BASE_SEED)
        chk = spearmanr(predict_text_only(feats[ref0], T_good, T_bad), q_all[ref0])[0]
        

        acc = {}
        runs_rows = []
        n_test_ref = 0

        for si, seed in enumerate(SEEDS):
            ref_idx, test_idx = split_ref_test(q_all, groups, ref_ratio=REF_RATIO,
                                               seed=seed, mode=SPLIT_MODE)
            feats_ref, feats_test = feats[ref_idx], feats[test_idx]
            q_ref, q_test = q_all[ref_idx], q_all[test_idx]
            ref_groups = np.unique(groups[ref_idx])
            test_groups = np.unique(groups[test_idx])
            n_test_ref = len(test_idx)

    
            preds_this_split = {}

            
            preds_this_split[BASELINE_NAME] = predict_text_only(feats_test, T_good, T_bad)

           
            if SUBSPACE_MODE in ("mos", "both"):
                F_best, F_worst, k = build_subspace_from_mos(feats_ref, q_ref)
                V_q = TF.normalize(F_best - F_worst, dim=-1)
                V_q, s = align_direction(V_q, T_dir)
                Tg_c, Tb_c, attn = calip_text_calibration(T_good, T_bad, F_best, F_worst,
                                                          lambda_val=ALPHA, temp=TEMP)
                preds_this_split.update(
                    run_predictors(feats_test, V_q, T_good, T_bad, Tg_c, Tb_c, ALPHA,
                                   prefix="[域内-MOS] ")
                )
                if si == 0:
              
                    print(f"  cos(V_q_indomain, T_good-T_bad) = {s:.4f}")
                    print(f"  CaLIP attn: T_good→[best,worst]=[{attn[0,0]:.3f},{attn[0,1]:.3f}]  "
                          f"T_bad→[best,worst]=[{attn[1,0]:.3f},{attn[1,1]:.3f}]")

         
            if SUBSPACE_MODE in ("synthetic", "both"):
                try:
                    src = resolve_pristine_paths(ref_groups, cfg) if SYN_USE_PRISTINE else None
                    if src is None:
                        src = [paths[i] for i in ref_idx]
                        src_kind = f"ref集失真图({len(src)})"
                    else:
                        src_kind = f"ref集原图({len(src)})"
                    Fb_s, Fw_s = build_subspace_from_synthetic(src, seed=seed)
                    V_q_s = TF.normalize(Fb_s - Fw_s, dim=-1)
                    V_q_s, s2 = align_direction(V_q_s, T_dir)
                    Tg_s, Tb_s, _ = calip_text_calibration(T_good, T_bad, Fb_s, Fw_s,
                                                           lambda_val=ALPHA, temp=TEMP)
                    preds_this_split.update(
                        run_predictors(feats_test, V_q_s, T_good, T_bad, Tg_s, Tb_s, ALPHA,
                                       prefix="[域内-合成] ")
                    )
                    if si == 0:
                        print(f"  合成源={src_kind}, cos(V_q_syn, T_good-T_bad) = {s2:.4f}")
                except Exception as e:
                    if si == 0:
                        print(f"  [warn] 域内合成失真模式失败: {e}")

            
            if V_q_kadis is not None:
                Tg_k, Tb_k, _ = calip_text_calibration(T_good, T_bad, F_best_k, F_worst_k,
                                                       lambda_val=ALPHA, temp=TEMP)
                preds_this_split.update(
                    run_predictors(feats_test, V_q_kadis, T_good, T_bad, Tg_k, Tb_k, ALPHA,
                                   prefix="[跨域-KADIS] ")
                )

         
            for name, pr in preds_this_split.items():
                m = evaluate(pr, q_test, name)
                acc.setdefault(name, []).append(m)
                runs_rows.append({
                    "seed": int(seed),
                    "method": name,
                    "srcc": float(m["srcc"]),
                    "plcc_raw": float(m["plcc_raw"]),
                    "plcc_fit": float(m["plcc_fit"]),
                    "n": int(m["n"]),
                })

        summary = OrderedDict()
        for name, rs in acc.items():
            summary[name] = OrderedDict()
            for mt in METRICS:
                summary[name][mt] = summarize_values([r[mt] for r in rs])

        print_mean_std_table(key, acc, n_g, grouped)
        for mt in ["srcc", "plcc_fit", "plcc_raw"]:
            print_stability_table(key, summary, mt)
        print_paired_table(key, acc, "srcc")

       
        for name in acc:
            s_sr = summary[name]["srcc"]
            s_pl = summary[name]["plcc_fit"]
            all_rows.append(dict(
                dataset=key, method=name,
                grouped=int(grouped), n_groups=int(n_g),
                n_splits=N_SPLITS, n_test=int(n_test_ref),
                srcc_mean=round(s_sr["mean"], 4), srcc_std=round(s_sr["std"], 4),
                srcc_var=round(s_sr["var"], 6), srcc_range=round(s_sr["range"], 4),
                srcc_iqr=round(s_sr["iqr"], 4),
                plcc_fit_mean=round(s_pl["mean"], 4), plcc_fit_std=round(s_pl["std"], 4),
                plcc_fit_var=round(s_pl["var"], 6), plcc_fit_range=round(s_pl["range"], 4),
                plcc_fit_iqr=round(s_pl["iqr"], 4),
            ))

   
        runs_csv = os.path.join(OUT_DIR, f"{key}_runs.csv")
        summary_csv = os.path.join(OUT_DIR, f"{key}_summary.csv")
        json_path = os.path.join(OUT_DIR, f"{key}.json")

        with open(runs_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "method", "srcc", "plcc_raw", "plcc_fit", "n"])
            w.writeheader()
            w.writerows(runs_rows)

        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method", "metric", "n", "mean", "std", "var",
                        "min", "max", "range", "iqr", "median", "cv", "ci95"])
            for name, st in summary.items():
                for mt in METRICS:
                    s = st[mt]
                    w.writerow([key, name, mt, s["n"], s["mean"], s["std"], s["var"],
                                s["min"], s["max"], s["range"], s["iqr"],
                                s["median"], s["cv"], s["ci95"]])

        all_payload[key] = dict(
            n_images=int(len(paths)),
            n_groups=int(n_g),
            grouped=bool(grouped),
            n_test=int(n_test_ref),
            summary=summary,
            runs=runs_rows,
        )
        with open(json_path, "w") as f:
            json.dump({"config": _config_dict(), "dataset": key, "result": all_payload[key]},
                      f, indent=2, ensure_ascii=False)

        print(f"[保存] {key}: {runs_csv} | {summary_csv} | {json_path}")

    
    if all_rows:
        all_csv = os.path.join(OUT_DIR, "ALL_summary.csv")
        with open(all_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[保存] 全部结果已写入: {all_csv}")

        with open(os.path.join(OUT_DIR, "ALL.json"), "w") as f:
            json.dump({"config": _config_dict(), "results": all_payload}, f,
                      indent=2, ensure_ascii=False)

        print("\n" + "=" * 96)
        print("Mean Results")
        print("=" * 96)
        methods = []
        for r in all_rows:
            if r['method'] not in methods:
                methods.append(r['method'])
        print(f"{'Method':<48s}{'avg SRCC':>16s}{'avg PLCC(fit)':>18s}")
        print("-" * 96)
        for m in methods:
            sr = [r['srcc_mean'] for r in all_rows if r['method'] == m]
            pl = [r['plcc_fit_mean'] for r in all_rows if r['method'] == m]
            print(f"{m:<48s}{np.nanmean(sr):>16.4f}{np.nanmean(pl):>18.4f}")
        print("=" * 96)


def _config_dict():
    return dict(
        backbone=CLIP_BACKBONE,
        ref_ratio=REF_RATIO,
        cluster_ratio=CLUSTER_RATIO,
        min_ref=MIN_REF,
        min_cluster=MIN_CLUSTER,
        min_ref_groups=MIN_REF_GROUPS,
        min_test_groups=MIN_TEST_GROUPS,
        n_splits=N_SPLITS,
        base_seed=BASE_SEED,
        seeds=SEEDS,
        split_mode=SPLIT_MODE,
        subspace_mode=SUBSPACE_MODE,
        alpha=ALPHA,
        temp=TEMP,
        compare_with_kadis=COMPARE_WITH_KADIS,
        kadis_num_ref=KADIS_NUM_REF,
        syn_max_ref=SYN_MAX_REF,
        syn_use_pristine=SYN_USE_PRISTINE,
    )


if __name__ == "__main__":
    main()
