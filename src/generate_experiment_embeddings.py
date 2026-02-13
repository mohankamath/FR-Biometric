import os
import numpy as np
from deepface import DeepFace

DATA_DIR = "data/experiment"
OUT_DIR = "data/experiment_embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)

    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(rep)

        except Exception as e:
            print("Skipping", img_path, e)

    if len(embeddings) > 0:
        np.save(os.path.join(OUT_DIR, f"{person}.npy"), embeddings)
        print(f"Saved {len(embeddings)} for {person}")

print("Done.")
