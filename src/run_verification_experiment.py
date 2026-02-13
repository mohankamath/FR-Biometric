import os
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances

EMBEDDINGS_DIR = "data/embeddings/enrollment"
THRESHOLD = 0.6   # can tune later


def load_embeddings():
    db = []
    labels = []

    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            person = file.replace(".npy", "")
            arr = np.load(os.path.join(EMBEDDINGS_DIR, file))  # shape (N, 512)

            for emb in arr:
                db.append(emb)
                labels.append(person)

    return db, labels



def run_experiment():
    embeddings, labels = load_embeddings()

    TP = TN = FP = FN = 0

    total_pairs = 0

    for (i, j) in combinations(range(len(embeddings)), 2):
        total_pairs += 1

        dist = cosine_distances(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        same_person = labels[i] == labels[j]
        predict_same = dist < THRESHOLD

        if same_person and predict_same:
            TP += 1
        elif same_person and not predict_same:
            FN += 1
        elif not same_person and predict_same:
            FP += 1
        else:
            TN += 1

    print("\n=== Verification Results ===")
    print(f"Total pairs: {total_pairs}")
    print(f"TP: {TP}")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")

    accuracy = (TP + TN) / total_pairs
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    run_experiment()
