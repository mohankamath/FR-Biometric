import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

EMBEDDINGS_DIR = "data/embeddings"
TEST_IMAGE_PATH = "data/processed/test/Bill_Clinton/face_0.jpg"
MODEL_NAME = "ArcFace"
THRESHOLD = 0.75


def load_enrollment_embeddings(embeddings_dir):
    database = {}

    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            identity = file.replace(".npy", "")
            path = os.path.join(embeddings_dir, file)
            database[identity] = np.load(path)

    return database


def get_embedding(image_path):
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        enforce_detection=False
    )
    return np.array(result[0]["embedding"])


def recognize_face(test_embedding, database):
    best_identity = None
    best_score = float("inf")

    for identity, embeddings in database.items():
        for emb in embeddings:
            score = cosine(test_embedding, emb)
            if score < best_score:
                best_score = score
                best_identity = identity

    return best_identity, best_score


if __name__ == "__main__":
    print("Loading enrollment embeddings...")
    db = load_enrollment_embeddings(EMBEDDINGS_DIR)

    print("Generating embedding for test image...")
    test_embedding = get_embedding(TEST_IMAGE_PATH)

    print("Matching...")
    identity, score = recognize_face(test_embedding, db)

    print("\n=== Recognition Result ===")
    print(f"Best match: {identity}")
    print(f"Cosine distance: {score:.4f}")

    if score < THRESHOLD:
        print("Decision: ACCEPT")
    else:
        print("Decision: REJECT (Unknown)")
