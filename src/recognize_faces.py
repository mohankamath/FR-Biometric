import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

EMBEDDINGS_DIR = "data/embeddings/enrollment"
TEST_IMAGE_PATH = "data/processed/test/test_face.jpg"
MODEL_NAME = "ArcFace"
THRESHOLD = 0.75


def load_embeddings(embeddings_dir):
    db = {}

    for file in os.listdir(embeddings_dir):
        if file.endswith(".npy"):
            identity = os.path.splitext(file)[0]
            embedding_path = os.path.join(embeddings_dir, file)
            db[identity] = np.load(embedding_path)

    return db




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
    db = load_embeddings(EMBEDDINGS_DIR)
    print(f"Loaded {len(db)} identities")


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
