import os
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances

EMBEDDINGS_DIR = "data/experiment_embeddings"
THRESHOLD = 0.6   # tune later


def load_embeddings():
    db = []
    labels = []
    sources = []   # NEW → track which file / image index it came from

    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            person = file.replace(".npy", "")
            arr = np.load(os.path.join(EMBEDDINGS_DIR, file))  # (N, 512)

            for idx, emb in enumerate(arr):
                db.append(emb)
                labels.append(person)
                sources.append(f"{person}__img{idx}")

    return db, labels, sources


def run_experiment():
    embeddings, labels, sources = load_embeddings()

    TP = TN = FP = FN = 0
    total_pairs = 0

    false_accepts = []
    false_rejects = []

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
            false_rejects.append((sources[i], sources[j], dist))
        elif not same_person and predict_same:
            FP += 1
            false_accepts.append((sources[i], sources[j], dist))
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

    print("\n--- False Accepts (different people but matched) ---")
    for x in false_accepts[:10]:
        print(x)

    print("\n--- False Rejects (same person but not matched) ---")
    for x in false_rejects[:10]:
        print(x)

    # OPTIONAL → save to file for demo
    with open("experiment_mistakes.txt", "w") as f:
        f.write("False Accepts:\n")
        for x in false_accepts:
            f.write(str(x) + "\n")

        f.write("\nFalse Rejects:\n")
        for x in false_rejects:
            f.write(str(x) + "\n")

    print("\nSaved mistakes to experiment_mistakes.txt")


if __name__ == "__main__":
    run_experiment()
