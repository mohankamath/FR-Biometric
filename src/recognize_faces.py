import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Paths
ENROLLMENT_DIR = "data/processed/enrollment"
TEST_DIR = "data/test"
THRESHOLD = 0.4  # Cosine distance threshold for recognition

# Step 1: Build embedding database
def build_embedding_database(enroll_dir):
    db = {}
    for person in os.listdir(enroll_dir):
        person_path = os.path.join(enroll_dir, person)
        embeddings = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for {img_path}: {e}")
        if embeddings:
            db[person] = embeddings
    return db

# Step 2: Recognize a test image
def recognize_face(test_img_path, embedding_db):
    try:
        test_embedding = DeepFace.represent(test_img_path, model_name="Facenet")[0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding for {test_img_path}: {e}")
        return "Error"

    best_match = "Unknown"
    best_score = float("inf")

    for person, embeddings in embedding_db.items():
        for db_emb in embeddings:
            score = cosine(test_embedding, db_emb)  # smaller is more similar
            if score < best_score:
                best_score = score
                best_match = person

    if best_score > THRESHOLD:
        best_match = "Unknown"

    return best_match, best_score

# Run recognition on all test images
if __name__ == "__main__":
    db = build_embedding_database(ENROLLMENT_DIR)
    for img_name in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img_name)
        person, score = recognize_face(img_path, db)
        print(f"{img_name} → {person} (cosine: {score:.3f})")
