from pathlib import Path
import csv
import json


class RacketVisionTennisDataset:
    def __init__(self, root="dataset/racketvision/tennis"):
        self.root = Path(root)
        self.video_dir = self.root / "videos"
        self.all_dir = self.root / "all"

    def get_rally(self, match_id, rally_id):
        match_dir = self.all_dir / match_id

        video_path = self.video_dir / f"{match_id}_{rally_id}.mp4"
        ball_path = match_dir / "csv" / f"{rally_id}_ball.csv"
        racket_dir = match_dir / "racket" / rally_id

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if not ball_path.exists():
            raise FileNotFoundError(f"Ball annotation not found: {ball_path}")

        if not racket_dir.exists():
            raise FileNotFoundError(
                f"Racket annotation directory not found: {racket_dir}"
            )

        return {
            "match_id": match_id,
            "rally_id": rally_id,
            "video": video_path,
            "ball": ball_path,
            "racket": racket_dir,
        }

    def load_ball(self, match_id, rally_id):
        sample = self.get_rally(match_id, rally_id)

        ball_data = []

        with open(sample["ball"], newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                ball_data.append(
                    {
                        "frame": int(row["Frame"]),
                        "visibility": int(row["Visibility"]),
                        "x": float(row["X"]),
                        "y": float(row["Y"]),
                    }
                )

        return ball_data

    def load_racket(self, match_id, rally_id, frame):
        sample = self.get_rally(match_id, rally_id)

        racket_file = sample["racket"] / f"{frame:04d}.json"

        if not racket_file.exists():
            return []

        with open(racket_file) as f:
            return json.load(f)
