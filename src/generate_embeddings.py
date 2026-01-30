import os
import numpy as np
from deepface import DeepFace

PROCESSED_DIR = "data/processed/enrollment"
EMBEDDINGS_DIR = "data/embeddings/enrollment"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"

for person in os.listdir(PROCESSED_DIR):
    person_dir = os.path.join(PROCESSED_DIR, person)

    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        try:
            reps = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            # DeepFace returns a list
            embedding = reps[0]["embedding"]
            embeddings.append(embedding)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if len(embeddings) == 0:
        print(f"No embeddings for {person}")
        continue

    embeddings = np.array(embeddings)
    out_path = os.path.join(EMBEDDINGS_DIR, f"{person}.npy")
    np.save(out_path, embeddings)

    print(f"Saved {embeddings.shape[0]} embeddings for {person}")
