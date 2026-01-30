import os
import numpy as np
from deepface import DeepFace

def extract_embedding(image_path, model_name="ArcFace"):
    """
    Extract a face embedding using a pretrained DeepFace model.
    Returns a 512-D numpy array.
    """
    embedding = DeepFace.represent(
        img_path=image_path,
        model_name=model_name,
        enforce_detection=True,
        detector_backend="retinaface"
    )
    return np.array(embedding[0]["embedding"])


def build_embedding_database(face_dir):
    """
    Builds an embedding database from cropped face images.
    Assumes directory structure: face_dir/person_name/*.jpg
    """
    db = {}

    for person in os.listdir(face_dir):
        person_path = os.path.join(face_dir, person)
        if not os.path.isdir(person_path):
            continue

        db[person] = []

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            emb = extract_embedding(img_path)
            db[person].append(emb)

    return db
