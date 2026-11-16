"""Build the site data: score the final model on the clean test photos, list every training run, render the demo clip.

    python tools/build_site_data.py --project path/to/yolov10 --video path/to/event.mp4

--project is the training folder: dateset/<version>/, TrainGender/, NewGender/ and trainResult/ run folders.
Needs Ultralytics 8.3 or newer (YOLOv10 layers), a CUDA GPU and ffmpeg (FFMPEG env var or on PATH).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data"
MODEL = ROOT / "models" / "male_female.pt"
FINAL_RUN = "NewGender/Model_1"  # models/male_female.pt is this run's best.pt (new.pt in the project)
CONF = 0.5
COLORS = {"Female": (46, 138, 201), "Male": (181, 120, 58)}  # BGR of #C98A2E and #3A78B5
KEEP = ["results.png", "PR_curve.png", "confusion_matrix_normalized.png"]
# ponytail: fixed patch for the event clip (static camera, 1280 wide): someone half out of frame at the left edge
# whom no detector finds. Another video needs its own check.
ALWAYS_PIXELATE = [(0, 110, 130, 340)]
base = lambda name: name.split(".rf.")[0]


def start_of(model: str) -> str:
    name = model.replace("\\", "/").split("/")[-1]
    return "from scratch" if name.endswith(".yaml") else "COCO pre-trained" if name.startswith("yolov10") else "continued"


def history(project: Path) -> list[dict]:
    """Every run whose dataset is still on disk, in date order, with its own logged validation score."""
    runs = []
    for d in sorted(p for g in ("TrainGender", "NewGender", "trainResult") for p in (project / g).iterdir()):
        a = yaml.safe_load(open(d / "args.yaml", encoding="utf-8"))
        data = Path(str(a["data"]).replace("\\", "/")).parent.name
        if not (d / "results.csv").exists() or not (project / "dateset" / data).is_dir():
            continue
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(d / "results.csv"))]
        if not rows:
            continue
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        name = f"{d.parent.name}/{d.name}"
        keep = ROOT / "training_results" / name
        keep.mkdir(parents=True, exist_ok=True)
        for f in ["args.yaml", "results.csv"] + (KEEP if name == FINAL_RUN else []):
            shutil.copy2(d / f, keep / f)
        runs.append({"run": name, "date": datetime.date.fromtimestamp((d / "args.yaml").stat().st_mtime).isoformat(),
                     "dataset": data, "start": start_of(str(a["model"])), "epochs_run": len(rows),
                     "best_epoch": int(best["epoch"]), "val_map50": round(float(best["metrics/mAP50(B)"]), 3),
                     "val_map50_95": round(float(best["metrics/mAP50-95(B)"]), 3)})
    return sorted(runs, key=lambda r: (r["date"], r["run"]))


def evaluate(project: Path) -> dict:
    """Score the final model on the mix3 test photos that are in neither the training nor the validation split."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou

    ds = project / "dateset" / "mix3"
    seen = {base(f) for s in ("train", "valid") for f in os.listdir(ds / s / "images")}
    test = sorted((ds / "test" / "images").glob("*.jpg"))
    clean = [p for p in test if base(p.name) not in seen]
    lst = ROOT / "test_clean.txt"
    lst.write_text("\n".join(str(p) for p in clean))
    data = ROOT / "test_split.yaml"
    data.write_text(yaml.safe_dump({"path": str(ds), "train": "train/images", "val": "valid/images",
                                    "test": str(lst), "names": {0: "Female", 1: "Male"}}))
    model = YOLO(MODEL)
    m = model.val(data=str(data), split="test", imgsz=640, batch=16, device=0, plots=False, verbose=False)
    per_class = {model.names[c]: {"map50": round(float(m.box.ap50[i]), 3), "map50_95": round(float(m.box.ap[i]), 3)}
                 for i, c in enumerate(m.box.ap_class_index)}
    # Gender errors: a detection that overlaps a labelled person (IoU >= 0.5) but carries the other label.
    tally = {"people": 0, "found": 0, "right": 0, "female_as_male": 0, "male_as_female": 0, "extra": 0}
    for img in clean:
        r = model.predict(img, conf=CONF, device=0, verbose=False)[0]
        h, w = r.orig_shape
        rows = [list(map(float, line.split())) for line in
                (img.parent.parent / "labels" / f"{img.stem}.txt").read_text().splitlines() if line.strip()]
        gt = torch.tensor([[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                           for _, x, y, bw, bh in rows]).reshape(-1, 4)
        gc = [int(r_[0]) for r_ in rows]
        pc = r.boxes.cls.int().tolist()
        iou = box_iou(gt, r.boxes.xyxy.cpu())
        tally["people"] += len(gt)
        # ponytail: best-IoU match per labelled person, no one-to-one assignment
        for i in range(len(gt)):
            if iou.shape[1] and iou[i].max() >= 0.5:
                tally["found"] += 1
                p = pc[int(iou[i].argmax())]
                tally["right"] += p == gc[i]
                tally["female_as_male"] += gc[i] == 0 and p == 1
                tally["male_as_female"] += gc[i] == 1 and p == 0
        tally["extra"] += int((iou.max(0).values < 0.5).sum()) if iou.shape[0] else len(pc)
    lst.unlink()
    data.unlink()
    labels = [line.split()[0] for p in clean
              for line in (p.parent.parent / "labels" / f"{p.stem}.txt").read_text().splitlines() if line.strip()]
    return {"test_photos": len(test), "leaked": len(test) - len(clean), "clean_photos": len(clean),
            "female_boxes": labels.count("0"), "male_boxes": labels.count("1"),
            "test": {"precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
                     "map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3)},
            "per_class": per_class, "gender": {k: int(v) for k, v in tally.items()},
            "dataset": {s: len(os.listdir(ds / s / "images")) for s in ("train", "valid", "test")},
            "unique_train_photos": len({base(f) for f in os.listdir(ds / "train" / "images")})}


def pixelate(img, x1, y1, x2, y2):
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def render_demo(video: Path, project: Path, seconds: int = 20, width: int = 1280, start: int | None = None) -> dict:
    """Pick the busiest window, then write it with every head pixelated and the gender boxes drawn on top."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(MODEL)
    people = [YOLO(project / "yolo11n.pt"), YOLO(project / "yolov10s.pt")]  # two COCO person detectors
    faces = [cv2.CascadeClassifier(cv2.data.haarcascades + f) for f in
             ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]
    cap = cv2.VideoCapture(str(video))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * 5)
    score = {} if start is None else {start: 1}
    for f in range(0, total - int(fps * seconds), step) if start is None else []:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if ok:
            cls = model.predict(frame, conf=CONF, device=0, verbose=False)[0].boxes.cls.int().tolist()
            score[f] = min(cls.count(0), cls.count(1))  # want both labels on screen
    windows = {f: sum(score.get(f + k * step, 0) for k in range(seconds * int(fps) // step)) for f in score}
    start = max(windows, key=windows.get)
    cap.release()
    # Cut the window out first: seeking deep into a long H.264 file with OpenCV is slow.
    clip = OUT / "_window.mp4"
    subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-ss", str(start / fps), "-i",
                    str(video), "-t", str(seconds), "-an", "-c:v", "libx264", "-crf", "12", str(clip)], check=True)
    cap = cv2.VideoCapture(str(clip))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    ff = subprocess.Popen([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-s", f"{width}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "demo.mp4")],
                          stdin=subprocess.PIPE)
    counts = {"Female": 0, "Male": 0}
    recent = []  # head regions of the last few frames, so a one-frame miss never shows a face
    for _ in range(int(fps * seconds)):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, h))
        det = model.predict(frame, conf=CONF, device=0, verbose=False)[0]
        # Privacy: pixelate the top of every person box from both models and every face OpenCV finds.
        boxes = det.boxes.xyxy.tolist()
        for m in people:
            boxes += m.predict(frame, conf=0.15, classes=[0], device=0, verbose=False)[0].boxes.xyxy.tolist()
        heads = [(x1 - 0.15 * (x2 - x1), y1 - 0.15 * (x2 - x1), x2 + 0.15 * (x2 - x1),
                  y1 + max(0.45 * (y2 - y1), 1.2 * (x2 - x1))) for x1, y1, x2, y2 in boxes]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for c in faces:
            heads += [(x - fw * 0.3, y - fh * 0.3, x + fw * 1.3, y + fh * 1.3)
                      for x, y, fw, fh in c.detectMultiScale(gray, 1.1, 4, minSize=(16, 16))]
        recent = (recent + [heads + ALWAYS_PIXELATE])[-6:]
        for region in (r for frame_heads in recent for r in frame_heads):
            pixelate(frame, *region)
        for (x1, y1, x2, y2), c, s in zip(det.boxes.xyxy.int().tolist(), det.boxes.cls.int().tolist(),
                                          det.boxes.conf.tolist()):
            name = model.names[c]
            counts[name] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[name], 2)
            label = f"{name} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLORS[name], -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    cap.release()
    clip.unlink()
    return {"clip_s": seconds, "start_s": round(start / fps), "boxes": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = history(a.project)
    ev = evaluate(a.project)
    print("test", ev["test"], ev["gender"], flush=True)
    for f in KEEP[:1]:
        shutil.copy2(ROOT / "training_results" / FINAL_RUN / f, ROOT / "web" / "img" / "final-results.png")
    demo = render_demo(a.video, a.project)
    (OUT / "site.json").write_text(json.dumps({"final_run": FINAL_RUN, "conf": CONF, **ev, "runs": runs,
                                               "demo": demo}, indent=1))
    print("done:", len(runs), "runs,", demo)


if __name__ == "__main__":
    main()
