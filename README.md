# Quality Signal Is Already There: Training-Free Cross-Domain IQA by Amplifying latent Quality Direction in Frozen CLIP

Training-free evaluation code for image quality assessment (IQA) and image aesthetic assessment (IAA), built on CLIP (RN50). All four scripts run inference only and print SRCC / PLCC to the terminal.

| Script | Experiment |
|---|---|
| `mos_anchor_iqa.py` | MOS-supervised anchors from KADID-10k |
| `kadis_anchor_iqa.py` | MOS-free anchors from KADIS-700K distorted images |
| `intra_dataset.py` | Intra-dataset quality direction correction |
| `transfer_iaa.py` | Cross-domain transfer to aesthetic assessment (IAA) |

---

## 1. Installation

```bash
conda create -n iqa python=3.10 -y
conda activate iqa

# 1) Install torch / torchvision matching your CUDA version (keep them paired)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU / macOS: pip install torch torchvision

# 2) Install the remaining dependencies
pip install -r requirements.txt
```

Notes:

- The CLIP source code is vendored in `./clip`. **Do not** run `pip install clip`; the PyPI package
  with that name is unrelated.
- On the first run, `clip.load("RN50")` downloads `RN50.pt` from `openaipublic.azureedge.net` into
  `~/.cache/clip`. For offline machines, place the weight file there in advance.
- A single GPU with ≥4GB memory is enough. Without a GPU the scripts fall back to CPU, which is slow.

---

## 2. Datasets

All test datasets can be downloaded here:

> https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets/tree/main

Unpack them under `data/` in the project root (folder names are up to you, as long as they match the
paths set in the scripts):

```
QDA-IQA/
├── data/
│   ├── kadid10k/images/                 # KADID-10k distorted images (anchor source)
│   ├── koniq10k/1024x768/               # default test set
│   ├── BID/ImageDatabase/               # default test set
│   ├── kadis700k/kadis700k/ref_imgs/    # KADIS-700K reference images (ref_imgs only)
│   ├── AVA_dataset/ava_images/          # AVA support set (transfer_iaa.py)
│   └── PARA/imgs/                       # PARA aesthetic test set
│   └── ..........                       # other test sets
├── datasets_label/                      # shipped with this repo, no download needed
└── *.py
```

- Label files in `datasets_label/` use the format `filename<comma/tab/space>MOS`. The parser takes the
  **first field as the image path** and the **last field as the score**; extra columns are ignored.
    Available label files: `live`, `livec`, `tid2013`, `CSIQ`, `spaq`, `flive`, `kadid10k`, `koniq10k`, `BID`, `AADB`, `AGI1k`, `AGI3k`, `BAID`, `TDA66K`, `pipal22`, `para_iqa`, `para_iaa`, `ava`.

---

## 3. Dataset paths

- `kadis_anchor_iqa.py` and `transfer_iaa.py` take command-line arguments. The defaults reproduce the
  original hard-coded settings, so running them without arguments gives the same result as before;
  pass `--test_img_dir` / `--test_mos_path` (and the other options shown by `--help`) to switch
  datasets or sweep hyperparameters.
- `mos_anchor_iqa.py` and `intra_dataset.py` still read paths from the hard-coded constants at the
  top of the file (`img_dir` / `mos_path`, `ACTIVE_IMG_DIR` / `ACTIVE_MOS_PATH`); edit them directly.

---

## 4. Running the experiments

```bash
python mos_anchor_iqa.py      # anchors from MOS-labelled KADID-10k
python kadis_anchor_iqa.py    # anchors from KADIS-700K, no MOS used
python intra_dataset.py       # in-domain correction on the test set itself
python transfer_iaa.py        # transfer to aesthetic assessment (AVA support -> PARA test)

# examples with arguments
python kadis_anchor_iqa.py --test_img_dir ./data/kadis_lived/kadis_lived \
    --test_mos_path ./datasets_label/livec_label.txt --cache_dir ./cache_kadis_all
python transfer_iaa.py --test_img_dir ./data/AADB/images \
    --test_mos_path ./datasets_label/AADB_labels.txt --gamma 1.0 --alpha 2.0
```
