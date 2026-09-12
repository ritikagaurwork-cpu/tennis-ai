import cv2
import csv
import json
from pathlib import Path

VIDEO = Path("dataset/racketvision/tennis/videos/match172_000.mp4")
BALL = Path("dataset/racketvision/tennis/all/match172/csv/000_ball.csv")
RACKET_DIR = Path("dataset/racketvision/tennis/all/match172/racket/000")

IMG_OUT = Path("dataset/yolo_test/images")
LBL_OUT = Path("dataset/yolo_test/labels")

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1920
HEIGHT = 1080

ball = {}

with open(BALL) as f:
    for row in csv.DictReader(f):
        ball[int(row["Frame"])] = (
            float(row["X"]),
            float(row["Y"]),
            int(row["Visibility"])
        )

cap = cv2.VideoCapture(str(VIDEO))

saved = 0

for frame_id in sorted(ball.keys()):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    ret, frame = cap.read()

    if not ret:
        continue

    img_name = f"match172_{frame_id:04d}.jpg"

    cv2.imwrite(str(IMG_OUT / img_name), frame)

    labels = []

    x, y, visible = ball[frame_id]

    if visible:
        labels.append(
            f"0 {x/WIDTH:.6f} {y/HEIGHT:.6f} 0.01 0.01"
        )

    racket_file = RACKET_DIR / f"{frame_id:04d}.json"

    if racket_file.exists():

        with open(racket_file) as f:
            rackets = json.load(f)

        for r in rackets:
            bx, by, w, h = r["bbox_xywh"]

            labels.append(
                f"1 {(bx+w/2)/WIDTH:.6f} {(by+h/2)/HEIGHT:.6f} {w/WIDTH:.6f} {h/HEIGHT:.6f}"
            )

    with open(
        LBL_OUT / img_name.replace(".jpg", ".txt"),
        "w"
    ) as f:
        f.write("\n".join(labels))

    saved += 1

cap.release()

print("Frames saved:", saved)
