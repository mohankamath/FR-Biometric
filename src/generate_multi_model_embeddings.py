import os
import numpy as np
from deepface import DeepFace

DATA_DIR = "data/experiment"
BASE_OUT_DIR = "data/experiment_embeddings"

MODELS = ["ArcFace", "Facenet512", "VGG-Face", "SFace"]

def generate_all():
    for model_name in MODELS:
        out_dir = os.path.join(BASE_OUT_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n========== Extracting for {model_name} ==========")
        
        for person in os.listdir(DATA_DIR):
            person_dir = os.path.join(DATA_DIR, person)
            if not os.path.isdir(person_dir):
                continue

            # Skip if already generated (allows resuming if interrupted)
            out_path = os.path.join(out_dir, f"{person}.npy")
            if os.path.exists(out_path):
                continue

            embeddings = []
            for img in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img)
                try:
                    reps = DeepFace.represent(
                        img_path=img_path,
                        model_name=model_name,
                        enforce_detection=False
                    )
                    embeddings.append(reps[0]["embedding"])
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

            if len(embeddings) > 0:
                np.save(out_path, embeddings)
                print(f"Saved {len(embeddings)} embeddings for {person} using {model_name}")

if __name__ == "__main__":
    generate_all()
    print("\nAll model embeddings generated!")