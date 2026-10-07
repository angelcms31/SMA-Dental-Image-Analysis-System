import glob, os, yaml
from ultralytics import YOLO

for best in sorted(glob.glob("runs_yolo/*/weights/best.pt")):
    run = os.path.dirname(os.path.dirname(best))
    data = yaml.safe_load(open(os.path.join(run, "args.yaml")))["data"]
    m = YOLO(best)
    r = m.val(data=data, verbose=False, plots=False)
    print(f"\n=== {os.path.basename(run)} | data: {data}")
    print(f"mAP50 overall: {r.box.map50:.3f}")
    for i, c in enumerate(r.box.ap_class_index):
        print(f"  {m.names[int(c)]}: {r.box.ap50[i]:.3f}")