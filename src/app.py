"""Consent-first smart-glasses software prototype.
Detect face regions; look up a synthetic student record by *supplied* ID.
No biometric identity matching, network transmission or personal records bundled.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def detect_faces(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(CASCADE).detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    return image, [tuple(map(int, box)) for box in faces]

def lookup(csv_path, student_id):
    with open(csv_path, newline="", encoding="utf-8") as file:
        for record in csv.DictReader(file):
            if record["student_id"] == student_id:
                return {
                    key: value for key, value in record.items()
                    if key in {"student_id", "display_name", "hobbies"}
                }
    return None

def run(image_path, csv_path, student_id, output_dir):
    image, faces = detect_faces(image_path)
    profile = lookup(csv_path, student_id) if student_id else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 180, 0), 2)
    if not cv2.imwrite(str(output / "detected_faces.jpg"), image):
        raise OSError("Could not save annotated image")
    result = {
        "faces_detected": len(faces),
        "profile": profile,
        "identity_matching": "not performed; user-supplied ID lookup only"
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--profiles", default="data/demo_profiles.csv")
    parser.add_argument("--student-id", default="DEMO001")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.profiles, args.student_id, args.output), indent=2))
