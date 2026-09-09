# SMA Dental Image Analysis System

**Repository:** [github.com/angelcms31/SMA-Dental-Image-Analysis-System](https://github.com/angelcms31/SMA-Dental-Image-Analysis-System)

An Enhancement of the Slime Mould Algorithm (ESMA) applied to automated
multilevel thresholding segmentation of dental abnormalities in
orthopantomogram (OPG) images.

> ⚠️ **Research prototype for academic demonstration only.** This system
> does not produce a validated clinical diagnosis. Findings are generated
> by threshold-based segmentation and a supplementary trained detector,
> not by any tool cleared for clinical use. Always consult a licensed
> dentist for actual diagnosis or treatment.

---

## What this project does

The standard Slime Mould Algorithm (SMA) is a metaheuristic optimizer
used here to find the best intensity thresholds for segmenting dental
OPG X-rays via Kapur's entropy. This project identifies three
structural weaknesses in the standard SMA (single-leader guidance,
random initialization, static parameter scheduling) and proposes an
Enhanced Slime Mould Algorithm (ESMA) that addresses each one:

1. **Fitness-weighted multi-leader guidance** instead of single-leader
   dependence
2. **Quasi-uniform (Latin Hypercube) initialization** instead of
   random uniform initialization
3. **Performance-feedback adaptive control** instead of a fixed
   parameter schedule

The two algorithms are compared on 300 real dental OPG images (DENTEX
dataset) using paired statistical testing, plus two supplementary
downstream tasks (a Random Forest diagnosis classifier and a YOLOv8
per-tooth detector) to test whether the segmentation-level improvement
carries over to practical diagnostic use.

A FastAPI backend exposes both algorithms and the trained models over
HTTP, with a React/TypeScript frontend for interactive use: upload an
OPG image, run Standard SMA and/or ESMA, and compare results side by
side.

---

## Repository structure

```
SMA-Dental-Image-Analysis-System/
├── run.py                     # Launches backend + frontend together
├── backend/
│   ├── main.py                 # FastAPI app (all API endpoints)
│   ├── sma_algorithms.py       # Standard SMA and ESMA implementations
│   ├── metrics.py              # PSNR / SSIM evaluation
│   ├── overlay.py              # Heuristic annotated-overlay rendering
│   ├── predict_yolo.py         # YOLO inference + overlay rendering
│   ├── compare_three_way.py    # Core comparative benchmark (300 images)
│   ├── ablate_esma_v2.py       # Component ablation study
│   ├── bench_common.py         # Shared dev/holdout split helpers
│   ├── dentex_to_yolo.py       # Converts DENTEX annotations -> YOLO format
│   ├── merge_mendeley_into_dentex_yolo.py  # Merges supplementary dataset
│   ├── train_yolo_teeth.py     # Trains the YOLOv8 per-tooth detector
│   ├── train_dentex_classifier.py  # Trains the Random Forest classifier
│   ├── requirements.txt
│   ├── tests/
│   │   ├── test_sma_algorithms.py
│   │   └── fixtures/mendeley_reference.npz
│   ├── models/                 # Trained RF classifiers (generated)
│   └── runs_yolo/               # Trained YOLO weights (see below)
└── frontend/
    ├── src/OpgAnalyzer.tsx      # Main UI component
    └── package.json
```

---

## Setup

**Requirements:** Python 3.10+ (developed on 3.14), Node.js 18+, Git,
[Git LFS](https://git-lfs.com/) (for the trained YOLO weights).

**1. Clone the repository**
```bash
git clone https://github.com/angelcms31/SMA-Dental-Image-Analysis-System.git
```

**2. Move into the project folder**
```bash
cd SMA-Dental-Image-Analysis-System
```

**3. Download the trained YOLO weights (Git LFS)**
```bash
git lfs pull
```
This pulls the actual `best.pt` model files. Without this step, they
will appear as small placeholder text files instead of real models.

**4. Run the system**
```bash
python run.py
```
On first run, this installs `backend/requirements.txt` and runs
`npm install` in `frontend/` automatically (later runs skip this
unless those files change), then starts both servers:

- FastAPI backend → `http://localhost:8000`
- Vite frontend → `http://localhost:5173`

**5. Open the app**
Open `http://localhost:5173` in your browser, upload a panoramic OPG
image, and run Standard SMA, ESMA, or both.

---

## Reproducing the results

**Core comparative benchmark** (Standard SMA vs. ESMA, 300 DENTEX images):

```bash
cd backend
python compare_three_way.py --images_dir <path to DENTEX xrays> \
    --subset all --max_images 300 --seed 42 \
    --sma_population 30 --sma_iterations 100 --skip_v1 \
    --out three_way_300.json
```

**DENTEX diagnosis classifier:**

```bash
python train_dentex_classifier.py \
    --json <path to train_quadrant_enumeration_disease.json> \
    --images_dir <path to xrays> \
    --sma_population 20 --sma_iterations 40 --include_v2
```

**YOLO per-tooth detector** (repeat with `--sma_algorithm standard` /
`enhanced_v2` and with/without the merged supplementary dataset for
the full 5-configuration comparison reported in the thesis):

```bash
python dentex_to_yolo.py --json <...> --images_dir <...> \
    --out_dir ./yolo_dataset --use_esma --sma_algorithm enhanced_v2
python train_yolo_teeth.py --data ./yolo_dataset/data.yaml \
    --epochs 40 --imgsz 640 --batch 4 --device cpu --name my_run
```

**Unit / regression tests:**

```bash
python -m pytest tests -v
```

---

## Trained YOLO weights (Git LFS)

The trained `best.pt` files under `backend/runs_yolo/*/weights/` are
tracked via Git LFS (see `.gitattributes`). After cloning, run
`git lfs pull` if the weight files appear as small placeholder text
instead of actual model files. `yolov8n.pt` (the untrained base model)
is not tracked -- Ultralytics downloads it automatically on first use.

---

## Thesis context

Developed as part of a BSCS thesis at Pamantasan ng Lungsod ng
Maynila: *"An Enhancement of Slime Mould Algorithm Applied in
Automated Dental Image Analysis System for Segmentation of
Abnormalities in Orthopantomogram Images."*

**Authors:** Apelledo, Mark Daniel A. & Camus, Angel Lyn R.

**Datasets used:**
- [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) (CC-BY-NC-SA-4.0)
- Supplementary: [Dental OPG X-ray Dataset](https://data.mendeley.com/datasets/c4hhrkxytw/4) (CC BY-NC 4.0), Mendeley Data
