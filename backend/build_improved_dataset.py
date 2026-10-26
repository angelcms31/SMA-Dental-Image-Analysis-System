"""
build_improved_dataset.py
---------------------------
Builds the combined YOLO dataset used to retrain the 4-class detector
(Impacted, Caries, Periapical Lesion, Deep Caries) from:

  * DENTEX disease subset (CC BY-NC-SA 4.0)  -- same seed-42 / 15% split as
    dentex_to_yolo.py, so its val split is the one the old models selected on
  * Mendeley "Dental OPG XRAY Dataset" v4 (CC BY 4.0) -- ORIGINAL images only.
    The 604 "augmented" images are copies of the originals; using them would
    leak the same patient across train/val/test. The Kaggle copy
    (arfintanim/dental-opg-xray) is a byte-identical mirror, so pass either.
  * DENTEX official validation set (50 images, documented 4-class labels) and official test set (250 images,
    raw Turkish labels: only Caries/Impacted/Periapical Lesion are evaluable) as untouched test splits.
    The old models never trained on either.

Label mapping (Mendeley YOLO ids were derived from the data, see
MENDELEY_TO_DENTEX): only Caries and Impacted map to the four classes.
Infection / Fractured / Broken Down Crown-Root / Healthy are dropped.

Images are preprocessed with the SAME SMA/ESMA step the app uses
(sma_preprocess, crop_borders=False so boxes stay aligned). Run once per
--sma_algorithm that the app serves (standard, enhanced_v2).

Outputs (under --out_dir):
    images/{train,val}, labels/{train,val}, train.txt (rare classes repeated),
    data.yaml, test_dentex50.yaml, test_dentex250.yaml, test_mendeley.yaml, split_report.json

Usage:
    python build_improved_dataset.py --fetch --raw_dir ./raw --out_dir ./ds_standard --sma_algorithm standard
    python build_improved_dataset.py --dentex_dir D --dentex_test_dir T --dentex_val_dir V --mendeley_dir M --out_dir O --sma_algorithm enhanced_v2
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import cv2

from sma_preprocess import sma_preprocess

CLASS_NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]  # order of the deployed weights
MENDELEY_TO_DENTEX = {"0": "1", "2": "0"}  # Caries -> Caries, Impacted -> Impacted
# leading number of DENTEX test labels ("1-<name>-<FDI>"): 1 cu-ruk=caries, 6 go-mu-lu-=impacted, 7 lezyon=periapical lesion
# (per-image box rates match the train split). 0 saglam, 2 kuretaj, 3 kanal, 5 cekim, 8 kirik are ignored.
DENTEX_TEST_LABEL_TO_ID = {"6": 0, "1": 1, "7": 2}
DENTEX_URL = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/{}.zip"
MENDELEY_URL = "https://data.mendeley.com/public-api/zip/c4hhrkxytw/download/4"


def fetch(raw_dir):
    from remotezip import RemoteZip  # pip install remotezip
    os.makedirs(raw_dir, exist_ok=True)
    url = DENTEX_URL.format("training_data")  # only the disease subset (~2.9 GB of 10.9 GB)
    with RemoteZip(url) as z:
        names = [n for n in z.namelist() if n.startswith("training_data/quadrant-enumeration-disease/") and not n.endswith("/")]

    def extract(chunk):  # one RemoteZip per thread; per-file range requests are slow serially
        with RemoteZip(url) as z:
            for n in chunk:
                z.extract(n, raw_dir)
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(extract, [names[i::12] for i in range(12)]))
    for zip_name in ("test_data", "validation_data"):  # small: download whole zip
        zp = os.path.join(raw_dir, zip_name + ".zip")
        if not os.path.exists(zp):
            urllib.request.urlretrieve(DENTEX_URL.format(zip_name), zp)
        zipfile.ZipFile(zp).extractall(raw_dir)
    urllib.request.urlretrieve(DENTEX_URL.format("validation_triple").replace(".zip", ".json"),
                               os.path.join(raw_dir, "validation_triple.json"))
    zpath = os.path.join(raw_dir, "mendeley.zip")
    if not os.path.exists(zpath):  # Mendeley rejects urllib's default User-Agent
        req = urllib.request.Request(MENDELEY_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(os.path.join(raw_dir, "mendeley"), [n for n in z.namelist() if "/Original Dataset/" in n])
    return (os.path.join(raw_dir, "training_data", "quadrant-enumeration-disease"),
            os.path.join(raw_dir, "disease"),
            raw_dir,
            os.path.join(raw_dir, "mendeley"))


def _prep(job):
    src, dst, algo = job
    if os.path.exists(dst):  # rebuilds only redo missing files; delete a file to force it
        return src, True
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return src, False
    cv2.imwrite(dst, sma_preprocess(img, algorithm=algo, crop_borders=False))
    return src, True


def dentex_train_items(dentex_dir, seed):
    """[(src_image, yolo_lines, split)] using dentex_to_yolo.py's exact split logic."""
    with open(os.path.join(dentex_dir, "train_quadrant_enumeration_disease.json")) as f:
        data = json.load(f)
    id_to_name = {c["id"]: c["name"] for c in data["categories_3"]}
    assert [id_to_name[i] for i in sorted(id_to_name)] == CLASS_NAMES, id_to_name
    files = {i["id"]: (i["file_name"], i["width"], i["height"]) for i in data["images"]}
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    ids = list(by_img)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[:max(1, int(len(ids) * 0.15))])
    items = []
    for iid in ids:
        name, w, h = files[iid]
        src = os.path.join(dentex_dir, "xrays", name)
        if not os.path.exists(src):
            continue
        lines = []
        for a in by_img[iid]:
            x, y, bw, bh = a["bbox"]
            v = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            lines.append(f"{a['category_id_3']} " + " ".join(f"{max(0, min(1, t)):.6f}" for t in v))
        items.append((src, lines, "val" if iid in val_ids else "train"))
    return items


def dentex_val_items(val_dir):
    """Official 50-image validation set, documented 4-class COCO labels (same scheme as train)."""
    with open(os.path.join(val_dir, "validation_triple.json")) as f:
        data = json.load(f)
    files = {i["id"]: (i["file_name"], i["width"], i["height"]) for i in data["images"]}
    by_img = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    items = []
    for iid, anns in by_img.items():
        name, w, h = files[iid]
        lines = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            v = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            lines.append(f"{a['category_id_3']} " + " ".join(f"{max(0, min(1, t)):.6f}" for t in v))
        items.append((os.path.join(val_dir, "validation_data", "quadrant_enumeration_disease", "xrays", name), lines))
    return items


def dentex_test_items(test_dir):
    """Official 250-image test set: labelme polygons with raw Turkish names. Only the three names whose meaning
    is unambiguous are kept; Deep Caries has no documented raw name, so it is NOT evaluable on this set."""
    items = []
    for fn in sorted(os.listdir(os.path.join(test_dir, "input"))):
        src = os.path.join(test_dir, "input", fn)
        with open(os.path.join(test_dir, "label", os.path.splitext(fn)[0] + ".json"), encoding="utf-8") as f:
            shapes = json.load(f)["shapes"]
        h, w = cv2.imread(src, cv2.IMREAD_GRAYSCALE).shape
        lines = []
        for s in shapes:
            cls = DENTEX_TEST_LABEL_TO_ID.get(s["label"].split("-")[0])
            if cls is None:
                continue
            xs, ys = zip(*s["points"])
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            v = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
            lines.append(f"{cls} " + " ".join(f"{max(0, min(1, t)):.6f}" for t in v))
        items.append((src, lines))
    return items


def mendeley_items(mendeley_dir):
    """Originals only, deduped by content hash. Split BY IMAGE (one image = one patient) from md5(filename):
    20% test / 10% val / 70% train. Hash-based so the Kaggle copy (two extra images) and the Mendeley copy get
    the SAME split for every shared image -- a seeded shuffle of a list would reshuffle everything.
    Train/val additionally drop images with Infection (1) or Broken-Down-Crown/Root (4) boxes: those regions are
    unlabeled for our 4 classes and would be learned as background, suppressing Periapical Lesion / Deep Caries."""
    base = next(r for r, _, fs in os.walk(mendeley_dir) if any(f.endswith(".txt") for f in fs))
    seen, items = set(), []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".jpg") or not os.path.exists(os.path.join(base, fn[:-4] + ".txt")):
            continue
        src = os.path.join(base, fn)
        digest = hashlib.md5(open(src, "rb").read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows = [l.split() for l in open(src[:-4] + ".txt") if l.split()]
        kept = [" ".join([MENDELEY_TO_DENTEX[r[0]]] + r[1:5]) for r in rows if r[0] in MENDELEY_TO_DENTEX]
        if not kept:
            continue
        bucket = int(hashlib.md5(fn[:-4].encode()).hexdigest(), 16) % 10
        split = "test" if bucket < 2 else "val" if bucket == 2 else "train"
        if split != "test" and any(r[0] in ("1", "4") for r in rows):
            continue
        items.append((src, kept, split))
    return items


def main(a):
    if a.fetch:
        a.dentex_dir, a.dentex_test_dir, a.dentex_val_dir, a.mendeley_dir = fetch(a.raw_dir)
    out = os.path.abspath(a.out_dir)
    dirs = {k: os.path.join(out, k) for k in ("images/train", "images/val", "labels/train", "labels/val",
                                               "test_dentex50/images", "test_dentex50/labels", "test_dentex250/images", "test_dentex250/labels",
                                               "test_mendeley/images", "test_mendeley/labels")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    jobs, labels = [], {}  # labels: dst_label_path -> lines
    train_files = []  # (dst_image, set of class ids in it)

    def add(src, lines, img_dir, lbl_dir, name):
        dst = os.path.join(dirs[img_dir], name + ".png")
        jobs.append((src, dst, a.sma_algorithm))
        labels[os.path.join(dirs[lbl_dir], name + ".txt")] = lines
        return dst

    for src, lines, split in dentex_train_items(a.dentex_dir, a.seed):
        dst = add(src, lines, f"images/{split}", f"labels/{split}", "dentex_" + os.path.splitext(os.path.basename(src))[0])
        if split == "train":
            train_files.append((dst, {int(l.split()[0]) for l in lines}))
    for src, lines, split in mendeley_items(a.mendeley_dir):
        name = "mendeley_" + os.path.basename(src)[:-4]
        if split == "test":
            add(src, lines, "test_mendeley/images", "test_mendeley/labels", name)
        else:
            dst = add(src, lines, f"images/{split}", f"labels/{split}", name)
            if split == "train":
                train_files.append((dst, {int(l.split()[0]) for l in lines}))
    for src, lines in dentex_val_items(a.dentex_val_dir):
        add(src, lines, "test_dentex50/images", "test_dentex50/labels", "dentex_val_" + os.path.splitext(os.path.basename(src))[0])
    for src, lines in dentex_test_items(a.dentex_test_dir):
        add(src, lines, "test_dentex250/images", "test_dentex250/labels", "dentex_test_" + os.path.splitext(os.path.basename(src))[0])

    print(f"preprocessing {len(jobs)} images with {a.sma_algorithm} ...", flush=True)
    with Pool(a.workers) as pool:
        bad = [s for s, ok in pool.imap_unordered(_prep, jobs, chunksize=4) if not ok]
    assert not bad, f"unreadable images: {bad}"
    for path, lines in labels.items():
        open(path, "w").write("\n".join(lines))

    # class imbalance: repeat images containing rare classes in the train list (no file copies)
    rep = {int(k): int(v) for k, v in (kv.split(":") for kv in a.rep.split(","))}
    with open(os.path.join(out, "train.txt"), "w") as f:
        for dst, classes in train_files:
            f.write((dst + "\n") * max([1] + [rep.get(c, 1) for c in classes]))

    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    for fname, train, val in (("data.yaml", "train.txt", "images/val"),
                              ("test_dentex50.yaml", "test_dentex50/images", "test_dentex50/images"),
                              ("test_dentex250.yaml", "test_dentex250/images", "test_dentex250/images"),
                              ("test_mendeley.yaml", "test_mendeley/images", "test_mendeley/images")):
        open(os.path.join(out, fname), "w").write(
            f"path: {out}\ntrain: {train}\nval: {val}\nnc: 4\nnames:\n{names}")

    def count(d):
        c = Counter()
        for fn in os.listdir(dirs[d]):
            c.update(CLASS_NAMES[int(l.split()[0])] for l in open(os.path.join(dirs[d], fn)) if l.strip())
        return {"images": len(os.listdir(dirs[d])), "boxes": dict(c)}
    report = {d: count(d) for d in ("labels/train", "labels/val", "test_dentex50/labels", "test_dentex250/labels", "test_mendeley/labels")}
    report["train_list_entries"] = sum(1 for _ in open(os.path.join(out, "train.txt")))
    json.dump(report, open(os.path.join(out, "split_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true", help="download DENTEX (disease subset + test) and Mendeley into --raw_dir")
    p.add_argument("--raw_dir", default="./raw")
    p.add_argument("--dentex_dir", help="folder with train_quadrant_enumeration_disease.json and xrays/")
    p.add_argument("--dentex_test_dir", help="DENTEX test 'disease' folder (input/ + label/)")
    p.add_argument("--dentex_val_dir", help="folder with validation_triple.json and validation_data/")
    p.add_argument("--mendeley_dir", help="Mendeley/Kaggle ORIGINAL images + .txt labels")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sma_algorithm", choices=["standard", "enhanced_v2"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rep", default="2:3,3:2", help="class_id:repeat for train.txt (default Periapical x3, Deep Caries x2)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    main(p.parse_args())
