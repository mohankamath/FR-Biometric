import os
from face_detection import detect_and_crop

RAW_ROOT = "data"
OUT_ROOT = "data/processed"

for split in ["enrollment", "test"]:
    raw_split = os.path.join(RAW_ROOT, split)
    out_split = os.path.join(OUT_ROOT, split)

    for person in os.listdir(raw_split):
        person_raw = os.path.join(raw_split, person)
        person_out = os.path.join(out_split, person)

        for img in os.listdir(person_raw):
            img_path = os.path.join(person_raw, img)
            detect_and_crop(img_path, person_out)

