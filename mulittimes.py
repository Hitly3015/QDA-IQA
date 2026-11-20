import os, gc, json, csv, random, warnings
import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
import clip
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from utils.kadid10k_distortions import imdist_generator, to_uint8

warnings.filterwarnings("ignore", message=".*Conversion from CIE-LAB.*")
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")
Image.MAX_IMAGE_PIXELS = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

KADIS_REF_DIR  = "./data/kadis700k/kadis700k/ref_imgs"
LABEL_ROOT     = "./datasets_label"
DATA_ROOT      = "./"

NUM_REF_IMAGES = 100
NUM_RUNS       = 15
SEEDS          = [42, 42 + i for i in random.sample(range(1, 10001), NUM_RUNS -1 )]
TEMP           = 0.5
ALPHA          = 2.0
TOP_K, WORST_K = 50, 50
BATCH_SIZE     = 1
NUM_WORKERS    = 8
MAX_SHORT_SIDE = 448

OUT_DIR   = "./results_multirun"
FEAT_CACHE= "./.cache_feats_all"
ANCH_CACHE= "./.cache_kadis_mulitimes"
for d in (OUT_DIR, FEAT_CACHE, ANCH_CACHE): os.makedirs(d, exist_ok=True)

DATASETS = [
    dict(name="LIVE",     img_dir=f"{DATA_ROOT}/LIVEIQA_release2",
         mos=f"{LABEL_ROOT}/live_label.txt",        sign="auto"),
    dict(name="CSIQ",     img_dir=f"{DATA_ROOT}/CSIQ/dst_imgs",
         mos=f"{LABEL_ROOT}/CSIQ_label.txt",        sign="auto"),
    dict(name="TID2013",  img_dir=f"{DATA_ROOT}/tid2013/distorted_images",
         mos=f"{LABEL_ROOT}/tid2013_label.txt",     sign="auto"),
    dict(name="KADID10K", img_dir=f"{DATA_ROOT}/kadid10k/images",
         mos=f"{LABEL_ROOT}/kadid10k_label.txt",    sign="auto"),
    dict(name="PIPAL",    img_dir=f"{DATA_ROOT}/PIPAL/Dist_Imgs",
         mos=f"{LABEL_ROOT}/pipal22_train.txt",     sign="auto"),

    dict(name="LIVEC",    img_dir=f"{DATA_ROOT}/ChallengeDB_release/Images",
         mos=f"{LABEL_ROOT}/livec_label.txt",       sign="auto"),
    dict(name="KonIQ10K", img_dir=f"{DATA_ROOT}/koniq10k/1024x768",
         mos=f"{LABEL_ROOT}/koniq10k_label.txt",    sign="auto"),
    dict(name="BID",      img_dir=f"{DATA_ROOT}/BID/ImageDatabase",
         mos=f"{LABEL_ROOT}/BID_labels.txt",        sign="auto"),
    dict(name="SPAQ",     img_dir=f"{DATA_ROOT}/SPAQ/TestImage",
         mos=f"{LABEL_ROOT}/spaq_label.txt",        sign="auto"),
    dict(name="FLIVE",    img_dir=f"{DATA_ROOT}/FLIVE_Database/database",
         mos=f"{LABEL_ROOT}/flive_label.txt",       sign="auto"),

]
DATASETS_TO_RUN = None      
METHOD_KEYS = ["M1_img×text", "M2_proj_q", "M3_projcomp×text",
               "M4_res×text", "M5_img×calibtext", "M6_res×calibtext"]

model, preprocess = clip.load("RN50", device=device)
model.eval()


def load_dataset(img_dir, mos_path):
    paths, scores = [], []
   
    with open(mos_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                name, mos = parts[0], float(parts[-1])
                p = os.path.join(img_dir, name)
                if os.path.exists(p):
                    paths.append(p); scores.append(mos)
    return paths, np.array(scores)


def _resize_short_side(img_pil, max_short=MAX_SHORT_SIDE):
    w, h = img_pil.size
    if min(w, h) > max_short:
        s = max_short / min(w, h)
        img_pil = img_pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    return img_pil


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self):  return len(self.paths)
    def __getitem__(self, i):
        try:
            img = _resize_short_side(Image.open(self.paths[i]).convert("RGB"))
            return preprocess(img), i, 1
        except Exception:
            return torch.zeros(3, 224, 224), i, 0


@torch.no_grad()
def extract_features_fast(paths, desc=""):
    ds = ImgListDataset(paths)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)
    feats, idxs = [], []
    for x, i, ok in tqdm(dl, desc=desc):
        m = ok.bool()
        if m.sum() == 0: continue
        f = model.encode_image(x[m].to(device, non_blocking=True))
        f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        feats.append(f.float().cpu()); idxs.append(i[m])
    torch.cuda.empty_cache()
    if not feats: return None, []
    return torch.cat(feats, 0), torch.cat(idxs).tolist()


def prepare_dataset(ds):

    tag = f"{ds['name']}_RN50_feats"
    cache = os.path.join(FEAT_CACHE, tag + ".pt")
    paths, mos_all = load_dataset(ds["img_dir"], ds["mos"])
    if len(paths) == 0:
        print(f"[skip] {ds['name']}); return None, None
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        if d.get("n_paths") == len(paths):
            print(f"[cache read] {ds['name']} cache: {cache}  (N={len(d['valid_idx'])})")
            return d["feats"].float(), mos_all[d["valid_idx"]]

    feats, vidx = extract_features_fast(paths, f"{ds['name']}")
    if feats is None: return None, None
    torch.save({"feats": feats.half(), "valid_idx": vidx, "n_paths": len(paths)}, cache)
    return feats.float(), mos_all[vidx]


@torch.no_grad()
def encode_pils_batched(pils, bs=BATCH_SIZE):
    out = []
    for i in range(0, len(pils), bs):
        b = torch.stack([preprocess(p) for p in pils[i:i + bs]], 0).to(device)
        f = model.encode_image(b)
        f = f / (f.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        out.append(f.float().cpu())
    return torch.cat(out, 0)


def get_kadis_unsupervised_clusters(kadis_ref_dir, num_ref, seed,
                                    top_k=TOP_K, worst_k=WORST_K):
    cache_path = os.path.join(
        ANCH_CACHE, f"kadis_clusters_ref{num_ref}_seed{seed}_t{top_k}_w{worst_k}.pt")
    if os.path.exists(cache_path):
        print(f"[缓存读取] 锚点 seed={seed}")
        d = torch.load(cache_path, map_location='cpu')
        return d["F_best"].float(), d["F_worst"].float()

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')
    allf = sorted([os.path.join(kadis_ref_dir, f)
                   for f in os.listdir(kadis_ref_dir) if f.endswith(exts)])
    refs = random.Random(seed).sample(allf, min(num_ref, len(allf)))

    Fb, Fw = [], []
    for ri, p in enumerate(tqdm(refs, desc=f"锚点构建 seed={seed}")):
        try:
            img = _resize_short_side(Image.open(p).convert("RGB"))
            with torch.no_grad():
                f0 = model.encode_image(preprocess(img).unsqueeze(0).to(device))
                f0 = (f0 / (f0.norm(p=2, dim=-1, keepdim=True) + 1e-12)).float().cpu()
            dseed = seed * 100000 + ri                       
            cands = [Image.fromarray(to_uint8(imdist_generator(img, d, l, seed=dseed)))
                     for d in range(1, 26) for l in range(1, 6)]
            cf = encode_pils_batched(cands)                  # (125, D)
            fid = (cf @ f0.t()).squeeze(-1).numpy()
            order = np.argsort(fid)
            Fb.append(cf[order[-top_k:]]); Fb.append(f0)
            Fw.append(cf[order[:worst_k]])
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[warn] {p}: {e}")
    F_best  = torch.cat(Fb, 0).mean(0)
    F_worst = torch.cat(Fw, 0).mean(0)
    torch.save({"F_best": F_best, "F_worst": F_worst, "refs": refs, "seed": seed}, cache_path)
    return F_best.float(), F_worst.float()


def build_anchor_bank(seeds):

    bank = []
    for s in seeds:
        fb, fw = get_kadis_unsupervised_clusters(KADIS_REF_DIR, NUM_REF_IMAGES, s)
        bank.append({"seed": s, "F_best": fb, "F_worst": fw})
    return bank


def encode_text_prompts(prompts):
    with torch.no_grad():
        f = model.encode_text(clip.tokenize(prompts).to(device))
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
    f = f.mean(0); f = f / (f.norm() + 1e-12)
    return f.cpu().float()


def calip_text_calibration(T_good, T_bad, F_best, F_worst, lambda_val=0.1, temp=1.0):
    T  = TF.normalize(torch.stack([T_good, T_bad], 0).float(), dim=-1)
    FA = TF.normalize(torch.stack([F_best, F_worst], 0).float(), dim=-1)
    P  = TF.softmax(T @ FA.t() / temp, dim=-1)
    Tc = TF.normalize(T + lambda_val * (P @ FA), dim=-1)
    return Tc[0], Tc[1], P


def logistic_func(X, b1, b2, b3, b4, b5):
    X = np.asarray(X, dtype=np.float64)
    z = np.clip(b2 * (X - b3), -50, 50)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * X + b5


def evaluate(preds, mos, name="", verbose=False):
    preds = np.asarray(preds, np.float64); mos = np.asarray(mos, np.float64)
    m = np.isfinite(preds) & np.isfinite(mos); preds, mos = preds[m], mos[m]
    srcc, _ = spearmanr(preds, mos); plcc_raw, _ = pearsonr(preds, mos)
    try:
        p0 = [np.max(mos) - np.min(mos), 10.0, np.mean(preds), 1.0, 1.0]
        popt, _ = curve_fit(logistic_func, preds, mos, p0=p0, maxfev=10000)
        plcc_fit, _ = pearsonr(logistic_func(preds, *popt), mos)
    except Exception:
        plcc_fit = plcc_raw
    if verbose:
        print(f"{name}: SRCC {srcc:.4f} | PLCC_raw {plcc_raw:.4f} | PLCC_fit {plcc_fit:.4f}")
    return {"srcc": float(srcc), "plcc_raw": float(plcc_raw), "plcc_fit": float(plcc_fit)}


def predict_text_only(feats, Tg, Tb):
    return (feats @ Tg - feats @ Tb).numpy()

def predict_projected_component_then_text(feats, Vq, Tg, Tb):
    q = feats @ Vq
    Fp = torch.outer(q, Vq); Fp = Fp / (Fp.norm(dim=-1, keepdim=True) + 1e-12)
    return (Fp @ Tg - Fp @ Tb).numpy()

def predict_residual_enhanced_then_text(feats, Vq, Tg, Tb, alpha):
    q = feats @ Vq
    Fh = feats + alpha * torch.outer(q, Vq)
    Fh = Fh / (Fh.norm(dim=-1, keepdim=True) + 1e-12)
    return (Fh @ Tg - Fh @ Tb).numpy()


def run_one_config(F_best, F_worst, feats, mos, T_good, T_bad, alpha=ALPHA, temp=TEMP):
    Fb, Fw = F_best.cpu().float(), F_worst.cpu().float()
    Vq = Fb - Fw; Vq = Vq / (Vq.norm() + 1e-12)
    Td = T_good - T_bad; Td = Td / (Td.norm() + 1e-12)
    align = torch.dot(Vq, Td).item()
    if align < 0: Vq, align = -Vq, -align       # 与原脚本一致：只翻 V_q
    Tgc, Tbc, attn = calip_text_calibration(T_good, T_bad, Fb, Fw, alpha, temp)

    r = OrderedDict()
    r["M1_img×text"]       = evaluate(predict_text_only(feats, T_good, T_bad), mos)
    r["M2_proj_q"]         = evaluate((feats @ Vq).numpy(), mos)
    r["M3_projcomp×text"]  = evaluate(predict_projected_component_then_text(feats, Vq, T_good, T_bad), mos)
    r["M4_res×text"]       = evaluate(predict_residual_enhanced_then_text(feats, Vq, T_good, T_bad, alpha), mos)
    r["M5_img×calibtext"]  = evaluate(predict_text_only(feats, Tgc, Tbc), mos)
    r["M6_res×calibtext"]  = evaluate(predict_residual_enhanced_then_text(feats, Vq, Tgc, Tbc, alpha), mos)
    extra = {"align_score": align,
             "attn_good_best": float(attn[0, 0]), "attn_good_worst": float(attn[0, 1]),
             "attn_bad_best": float(attn[1, 0]),  "attn_bad_worst": float(attn[1, 1])}
    return r, extra


def resolve_sign(ds, feats, mos_raw, anchor0, T_good, T_bad):
    s = ds.get("sign", "auto")
    if s in (1, -1): 
        print(f"[{ds['name']}] sign={s}"); return int(s)
    r, _ = run_one_config(anchor0["F_best"], anchor0["F_worst"], feats, mos_raw, T_good, T_bad)
    v = np.median([r[k]["srcc"] for k in ["M1_img×text", "M2_proj_q", "M6_res×calibtext"]])
    sign = 1 if v >= 0 else -1
 
    return sign


def summarize_runs(res_all, metric, title):
    print(f"\n{'='*108}\n# {title} —— {metric.upper()}  ({len(SEEDS)} 次随机, num_ref={NUM_REF_IMAGES})\n{'='*108}")
    head = (f"{'Method':<20}{'Mean':>8}{'Std':>8}{'Var':>10}{'Min':>8}"
            f"{'Max':>8}{'Range':>8}{'CI95':>9}{'Median':>9}")
    print(head); print("-" * len(head))
    tab = {}
    for k in METHOD_KEYS:
        v = np.asarray(res_all[k][metric], np.float64)
        std = v.std(ddof=1) if len(v) > 1 else 0.0
        var = v.var(ddof=1) if len(v) > 1 else 0.0
        ci  = 1.96 * std / np.sqrt(len(v)) if len(v) > 1 else 0.0
        print(f"{k:<20}{v.mean():>8.4f}{std:>8.4f}{var:>10.2e}{v.min():>8.4f}"
              f"{v.max():>8.4f}{v.max()-v.min():>8.4f}{ci:>9.4f}{np.median(v):>9.4f}")
        tab[k] = dict(mean=v.mean(), std=std, var=var, min=float(v.min()), max=float(v.max()),
                      range=float(v.max() - v.min()), ci95=ci, median=float(np.median(v)),
                      values=v.tolist())
    return tab


def print_grand_table(grand, metric):
    names = [n for n in grand if metric in grand[n]["summary"]]
    if not names: return
    print(f"\n{'#'*(24 + 18*len(METHOD_KEYS))}")
    print(f"# {metric.upper()}  mean±std over {NUM_RUNS} random reference sets")
    print(f"{'#'*(24 + 18*len(METHOD_KEYS))}")
    head = f"{'Dataset':<12}{'N':>7}" + "".join([f"{k:>18}" for k in METHOD_KEYS])
    print(head); print("-" * len(head))
    acc = defaultdict(list)
    for n in names:
        s = grand[n]["summary"][metric]
        row = f"{n:<12}{grand[n]['n_img']:>7}"
        for k in METHOD_KEYS:
            row += f"{s[k]['mean']:>11.4f}±{s[k]['std']:.3f}"
            acc[k].append(s[k]["mean"])
        print(row)
    print("-" * len(head))
    row = f"{'AVG':<12}{'-':>7}" + "".join([f"{np.mean(acc[k]):>18.4f}" for k in METHOD_KEYS])
    print(row)
    row = f"{'AVG std':<12}{'-':>7}" + "".join(
        [f"{np.mean([grand[n]['summary'][metric][k]['std'] for n in names]):>18.4f}" for k in METHOD_KEYS])
    print(row)


def print_ensemble_table(grand, metric):
    names = [n for n in grand if grand[n].get("ensemble")]
    if not names: return
    print(f"\n[Ensemble 锚点: {NUM_RUNS} 次平均, 等效 {NUM_RUNS*NUM_REF_IMAGES} 张参考图]  {metric.upper()}")
    head = f"{'Dataset':<12}" + "".join([f"{k:>18}" for k in METHOD_KEYS])
    print(head); print("-" * len(head))
    for n in names:
        e = grand[n]["ensemble"]
        print(f"{n:<12}" + "".join([f"{e[k][metric]:>18.4f}" for k in METHOD_KEYS]))


anchor_bank = build_anchor_bank(SEEDS)

T_good = encode_text_prompts(['Good image', 'Sharp image', 'sharp edges',
                              'High resolution image', 'Noise-free image'])
T_bad  = encode_text_prompts(['bad image', 'blurry image', 'blurry edges',
                              'low resolution image', 'noisy image'])

Fb_ens = torch.stack([a["F_best"]  for a in anchor_bank], 0).mean(0)
Fw_ens = torch.stack([a["F_worst"] for a in anchor_bank], 0).mean(0)

print(f"\n{'#'*90}\n# 阶段 2/2: 逐数据集评测\n{'#'*90}")
todo = [d for d in DATASETS if (DATASETS_TO_RUN is None or d["name"] in DATASETS_TO_RUN)]
grand = OrderedDict()

for ds in todo:
    print(f"\n{'*'*90}\n*  DATASET: {ds['name']}\n{'*'*90}")
    try:
        feats, mos_raw = prepare_dataset(ds)
        if feats is None: continue
        sign = resolve_sign(ds, feats, mos_raw, anchor_bank[0], T_good, T_bad)
        mos = sign * mos_raw
        print(f"[{ds['name']}] N={len(mos)}  MOS range=[{mos_raw.min():.3f},{mos_raw.max():.3f}]")

        res_all = defaultdict(lambda: defaultdict(list)); extras = []
        for run_i, a in enumerate(anchor_bank, 1):
            r, ex = run_one_config(a["F_best"], a["F_worst"], feats, mos, T_good, T_bad)
            for m, dd in r.items():
                for mk, mv in dd.items(): res_all[m][mk].append(mv)
            ex["seed"] = a["seed"]; extras.append(ex)
            print(f"  run {run_i:2d}/{NUM_RUNS} seed={a['seed']:<4} align={ex['align_score']:.4f} | "
                  + " | ".join([f"{m.split('_')[0]} {r[m]['srcc']:.4f}"
                                for m in ["M1_img×text", "M2_proj_q", "M4_res×text", "M6_res×calibtext"]]))

        print(f"\n[{ds['name']}] 逐次 SRCC 明细")
        h = f"{'seed':<7}" + "".join([f"{k:>19}" for k in METHOD_KEYS]); print(h); print("-"*len(h))
        for i, s in enumerate(SEEDS):
            print(f"{s:<7}" + "".join([f"{res_all[k]['srcc'][i]:>19.4f}" for k in METHOD_KEYS]))

        summary = {mt: summarize_runs(res_all, mt, ds["name"])
                   for mt in ["srcc", "plcc_raw", "plcc_fit"]}
        ens, _ = run_one_config(Fb_ens, Fw_ens, feats, mos, T_good, T_bad)
        print(f"\n[{ds['name']}] Ensemble 锚点: " +
              " | ".join([f"{k} {ens[k]['srcc']:.4f}" for k in METHOD_KEYS]))

        al = np.array([e["align_score"] for e in extras])
        print(f"[{ds['name']}] align(V_q,T_dir) = {al.mean():.4f} ± {al.std(ddof=1):.4f}")

        grand[ds["name"]] = dict(n_img=int(len(mos)), sign=int(sign),
                                 summary=summary, ensemble=ens, extras=extras)

        with open(os.path.join(OUT_DIR, f"{ds['name']}_ref{NUM_REF_IMAGES}_x{NUM_RUNS}.json"), "w") as f:
            json.dump(grand[ds["name"]], f, indent=2, ensure_ascii=False)
        with open(os.path.join(OUT_DIR, f"{ds['name']}_runs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed"] + [f"{m}_{k}" for m in METHOD_KEYS
                                   for k in ["srcc", "plcc_raw", "plcc_fit"]])
            for i, s in enumerate(SEEDS):
                w.writerow([s] + [res_all[m][k][i] for m in METHOD_KEYS
                                  for k in ["srcc", "plcc_raw", "plcc_fit"]])
    except Exception as e:
        import traceback; print(f"[ERROR] {ds['name']}: {e}"); traceback.print_exc()
    finally:
        feats = None; gc.collect(); torch.cuda.empty_cache()

for mt in ["srcc", "plcc_fit", "plcc_raw"]:
    print_grand_table(grand, mt)
for mt in ["srcc", "plcc_fit"]:
    print_ensemble_table(grand, mt)

with open(os.path.join(OUT_DIR, f"ALL_ref{NUM_REF_IMAGES}_x{NUM_RUNS}.json"), "w") as f:
    json.dump({"config": dict(num_ref=NUM_REF_IMAGES, seeds=SEEDS, alpha=ALPHA, temp=TEMP,
                              top_k=TOP_K, worst_k=WORST_K, backbone="RN50",
                              max_short=MAX_SHORT_SIDE),
               "results": grand}, f, indent=2, ensure_ascii=False)

for mt in ["srcc", "plcc_fit"]:
    with open(os.path.join(OUT_DIR, f"TABLE_{mt}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Dataset", "N"] + METHOD_KEYS)
        for n, g in grand.items():
            w.writerow([n, g["n_img"]] +
                       [f"{g['summary'][mt][k]['mean']:.4f}±{g['summary'][mt][k]['std']:.4f}"
                        for k in METHOD_KEYS])
print(f"\n[完成] 结果目录: {OUT_DIR}")
