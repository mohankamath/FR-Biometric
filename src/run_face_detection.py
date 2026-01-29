import os
from face_detection import detect_and_crop

DATA_DIR = "data"
SPLITS = ["enrollment", "test"]

INPUT_DIR = DATA_DIR
OUTPUT_DIR = os.path.join(DATA_DIR, "detected")

os.makedirs(OUTPUT_DIR, exist_ok=True)

log = []

for split in SPLITS:
    in_split = os.path.join(INPUT_DIR, split)
    out_split = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)

    for person in os.listdir(in_split):
        person_in = os.path.join(in_split, person)
        person_out = os.path.join(out_split, person)
        os.makedirs(person_out, exist_ok=True)

        for img in os.listdir(person_in):
            in_path = os.path.join(person_in, img)
            out_path = os.path.join(person_out, img)

            success = detect_and_crop(in_path, out_path)

            log.append({
                "split": split,
                "person": person,
                "image": img,
                "detected": success
            })

# Summary
total = len(log)
failed = sum(1 for x in log if not x["detected"])

print(f"Detection complete: {total - failed}/{total} faces detected")
print(f"Failures: {failed}")
