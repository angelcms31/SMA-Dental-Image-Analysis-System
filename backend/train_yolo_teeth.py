"""
train_yolo_teeth.py
---------------------
Trains a real object detector (YOLOv8, via the `ultralytics` library)
on the YOLO-format DENTEX dataset produced by dentex_to_yolo.py. This
detects individual teeth AND classifies each one's diagnosis in a
single pass -- this is what replaces the overlay's heuristic
localization with something actually trained for the task.

RUN THIS IN GOOGLE COLAB with a GPU runtime (Runtime > Change runtime
type > GPU). Training on CPU will be extremely slow.

Steps in Colab:
    1. Upload dentex_to_yolo.py, this file, and your DENTEX dataset
       (or mount Google Drive where they live).
    2. !pip install ultralytics
    3. !python dentex_to_yolo.py --json ... --images_dir ... --out_dir ./yolo_dataset
    4. !python train_yolo_teeth.py --data ./yolo_dataset/data.yaml --epochs 50

A pretrained YOLOv8n (nano) checkpoint is used as the starting point
(transfer learning) -- this trains much faster and needs far less data
than training from scratch, which matters given DENTEX's ~700 images.

After training, the best weights are saved to:
    runs/detect/train/weights/best.pt
Download that file -- it's what you'll load for inference (see
predict_yolo.py) and what you'll wire into the FastAPI backend.
"""

import argparse


def main(data_yaml, epochs, imgsz, batch, model_size, device=None, run_name="train"):
    import os
    from ultralytics import YOLO

    # yolov8n = nano (fastest, least accurate), yolov8s = small,
    # yolov8m = medium -- 'n' or 's' is a reasonable choice for a
    # ~700-image dataset; larger models need more data to avoid overfitting
    model = YOLO(f"yolov8{model_size}.pt")

    # Using an ABSOLUTE path for project= avoids ultralytics silently
    # nesting this under its own default runs directory (which is what
    # turned "runs/detect" into "runs/detect/runs/detect/train" in an
    # earlier version of this script -- if that happened to you, your
    # weights are at that doubled path, not the one this run will use).
    project_dir = os.path.abspath("runs_yolo")

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,       # early stop if val loss stops improving
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )
    if device is not None:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    metrics = model.val()
    print("\n=== Validation metrics ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"\nBest weights saved to: {os.path.join(project_dir, run_name, 'weights', 'best.pt')}")
    print("Download this file -- you'll need it for inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml from dentex_to_yolo.py")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960,
                         help="Training image size -- panoramic X-rays are wide, "
                              "so a larger size than YOLO's usual 640 helps small teeth stay visible. "
                              "On CPU-only training, use 640 instead (much faster, some accuracy trade-off).")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model_size", default="n", choices=["n", "s", "m"],
                         help="YOLOv8 variant: n=nano (fastest, best choice for CPU training), "
                              "s=small, m=medium")
    parser.add_argument("--device", default=None,
                         help="'cpu' to force CPU training, or a GPU index like '0'. "
                              "Leave unset to let ultralytics auto-detect.")
    parser.add_argument("--name", default="train", dest="run_name",
                         help="Folder name under runs_yolo/ for this run's results -- use a "
                              "DIFFERENT name for each dataset (e.g. train_standard, "
                              "train_enhanced) so runs never collide with each other's files.")
    args = parser.parse_args()
    main(args.data, args.epochs, args.imgsz, args.batch, args.model_size, args.device, args.run_name)