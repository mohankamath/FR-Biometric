import numpy as np
from numpy.linalg import norm

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def match_face(query_embedding, database, threshold=0.35):
    """
    Compare a query embedding against the database.
    Returns best match + decision.
    """
    best_match = None
    best_distance = float("inf")

    for person, embeddings in database.items():
        for emb in embeddings:
            dist = cosine_distance(query_embedding, emb)
            if dist < best_distance:
                best_distance = dist
                best_match = person

    decision = best_distance < threshold
    return best_match, best_distance, decision
