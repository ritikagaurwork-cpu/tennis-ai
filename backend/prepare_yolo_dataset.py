import cv2
import csv
import json
from pathlib import Path


ROOT = Path("dataset/racketvision/tennis")
OUT = Path("dataset/yolo")

WIDTH = 1920
HEIGHT = 1080


def load_ball(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["Frame"])] = (
                float(row["X"]),
                float(row["Y"]),
                int(row["Visibility"])
            )
    return data


def process_rally(match, rally, split):

    video = ROOT / "videos" / f"{match}_{rally}.mp4"
    ball_file = ROOT / "all" / match / "csv" / f"{rally}_ball.csv"
    racket_dir = ROOT / "all" / match / "racket" / rally

    if not video.exists() or not ball_file.exists():
        return 0


    ball = load_ball(ball_file)

    cap = cv2.VideoCapture(str(video))

    saved = 0

    for frame_id in sorted(ball.keys()):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()

        if not ret:
            continue


        labels = []

        x, y, visible = ball[frame_id]

        if visible:
            labels.append(
                f"0 {x/WIDTH:.6f} {y/HEIGHT:.6f} 0.01 0.01"
            )


        racket_file = racket_dir / f"{frame_id:04d}.json"

        if racket_file.exists():

            with open(racket_file) as f:
                rackets = json.load(f)

            for r in rackets:
                bx, by, w, h = r["bbox_xywh"]

                labels.append(
                    f"1 {(bx+w/2)/WIDTH:.6f} {(by+h/2)/HEIGHT:.6f} {w/WIDTH:.6f} {h/HEIGHT:.6f}"
                )


        if labels:

            name = f"{match}_{rally}_{frame_id:04d}"

            cv2.imwrite(
                str(OUT/"images"/split/f"{name}.jpg"),
                frame
            )

            with open(
                OUT/"labels"/split/f"{name}.txt",
                "w"
            ) as f:
                f.write("\n".join(labels))

            saved += 1


    cap.release()

    return saved



for split in ["train","val","test"]:

    with open(ROOT/"info"/f"{split}.json") as f:
        rallies = json.load(f)

    total = 0

    print("Processing", split, len(rallies), "rallies")

    for match, rally in rallies:

        n = process_rally(match, rally, split)
        total += n

        print(match, rally, n)


    print(split, "frames:", total)
