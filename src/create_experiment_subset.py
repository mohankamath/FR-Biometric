import os
import random
import shutil

RAW_DIR = "data/lfw"  # original dataset
OUT_DIR = "data/experiment"
MIN_IMAGES = 3
MAX_IMAGES = 5
NUM_PEOPLE = 150   # change later

os.makedirs(OUT_DIR, exist_ok=True)

people = []

for person in os.listdir(RAW_DIR):
    person_path = os.path.join(RAW_DIR, person)
    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]

    if len(images) >= MIN_IMAGES:
        people.append(person)

random.shuffle(people)
people = people[:NUM_PEOPLE]

for person in people:
    src = os.path.join(RAW_DIR, person)
    dst = os.path.join(OUT_DIR, person)
    os.makedirs(dst, exist_ok=True)

    images = [f for f in os.listdir(src) if f.endswith(".jpg")]
    random.shuffle(images)
    images = images[:MAX_IMAGES]

    for img in images:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("Subset created.")
