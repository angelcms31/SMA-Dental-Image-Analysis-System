# Retraining the 4-class detector on Colab (GPU)

Runtime > Change runtime type > GPU. Get the `backend/*.py` files onto Colab first
(commit + push this branch and `git clone` it, or zip `backend/*.py` and upload).

```python
!pip install -q ultralytics==8.4.133 remotezip opencv-python-headless scikit-image pyyaml
%cd /content/<repo>/backend
```

## 1. Build both datasets (downloads DENTEX + Mendeley itself)

Mendeley and Kaggle are the same images (byte-identical), so only Mendeley is fetched.
DENTEX is CC BY-NC-SA 4.0 (non-commercial research use, cite Hamamci et al. 2023); Mendeley is CC BY 4.0.

```python
!python build_improved_dataset.py --fetch --raw_dir /content/raw --out_dir /content/ds_standard --sma_algorithm standard
# second dataset reuses the downloaded data (no --fetch):
!python build_improved_dataset.py --dentex_dir /content/raw/training_data/quadrant-enumeration-disease \
    --dentex_test_dir /content/raw/disease --dentex_val_dir /content/raw \
    --mendeley_dir "$(find /content/raw/mendeley -name Original\ Dataset)" \
    --out_dir /content/ds_enhanced_v2 --sma_algorithm enhanced_v2
```

## 2. Train (same recipe as the old models: YOLOv8n; 640 = like-for-like, 960 = small-lesion test)

```python
for algo, ds in (("standard", "ds_standard"), ("enhanced_v2", "ds_enhanced_v2")):
    for sz in (640, 960):
        !python train_yolo_teeth.py --data /content/{ds}/data.yaml --epochs 80 --imgsz {sz} --batch 16 \
            --model_size n --device 0 --name v3_{algo}_{sz}
```

`train_yolo_teeth.py` stops early (patience 15) on the val split, which is the old DENTEX val split plus
10% of Mendeley; the test sets are never touched during training.
To fine-tune an existing model instead of starting from COCO: `--init_weights runs_yolo/<run>/weights/best.pt --lr0 0.002`
(not used for the reported numbers, because the old ESMA model has seen most Mendeley images).

## 3. Pick the winner on VALIDATION, then evaluate locally on the identical test files

Per algorithm, choose between the 640 and 960 run by the best validation fitness in
`runs_yolo/v3_<algo>_<size>/results.csv` (never by test score) and download that run's `weights/best.pt`.

Evaluate on the local machine, on the same preprocessed test files the old models were scored on
(`ds_standard` / `ds_enhanced_v2`; the Mendeley split is a filename hash, so Colab and local agree):

```
python evaluate_models.py --weights runs_yolo/v3_standard_960/weights/best.pt --data <ds_standard>/test_dentex250.yaml --out eval/standard_v3_dentex250.json
```

Repeat for `test_dentex50.yaml` and `test_mendeley.yaml` and for enhanced_v2 (`--imgsz` defaults to the checkpoint's size).
The old models' results are already in `backend/eval/*_old_*.json`.

**Promotion rule (fixed before seeing results):** swap a weights path in `frontend/src/OpgAnalyzer.tsx` only if,
at conf 0.25, Periapical Lesion recall on dentex250 and Deep Caries recall on dentex50 both improve and Impacted
recall/F1 does not drop by more than 3 points. Otherwise keep the old model and report that.
