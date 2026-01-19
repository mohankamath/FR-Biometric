import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from PIL import Image

# Where your project lives
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ENROLL_DIR = os.path.join(DATA_DIR, "enrollment")
TEST_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(ENROLL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Load dataset
lfw = fetch_lfw_people(color=True, resize=1.0)

images = lfw.images
labels = lfw.target
names = lfw.target_names

# Choose identities with enough images
MIN_IMAGES = 6
MAX_IDENTITIES = 12

selected = []
counts = {}

for label in labels:
    counts[label] = counts.get(label, 0) + 1

for label, count in counts.items():
    if count >= MIN_IMAGES:
        selected.append(label)
    if len(selected) == MAX_IDENTITIES:
        break

# Export images
for label in selected:
    person_name = names[label].replace(" ", "_")
    person_indices = np.where(labels == label)[0]

    os.makedirs(os.path.join(ENROLL_DIR, person_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, person_name), exist_ok=True)

    for i, idx in enumerate(person_indices[:6]):
        img = images[idx]
        img = (img * 255).astype("uint8")

        img_pil = Image.fromarray(img)

        if i < 3:
            path = os.path.join(ENROLL_DIR, person_name, f"{i}.jpg")
        else:
            path = os.path.join(TEST_DIR, person_name, f"{i}.jpg")

        img_pil.save(path)

print("Subset exported successfully.")
