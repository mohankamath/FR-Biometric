import os
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
DATA_DIR = "data/experiment_1000"  # Point this to your dedicated folder
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68                 # The shared "reasonable threshold"

def run_10_subject_report():
    print(f"Loading images from {DATA_DIR}...")
    embeddings = []
    labels = []

    # 1. Extract Embeddings directly into memory
    for person in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for img in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img)
            try:
                rep = DeepFace.represent(
                    img_path=img_path, 
                    model_name=MODEL_NAME, 
                    enforce_detection=False
                )[0]["embedding"]
                embeddings.append(rep)
                labels.append(person)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f"Extracted {len(embeddings)} embeddings across {len(set(labels))} subjects.\n")

    # 2. Matrix Math for all possible image pairs
    dist_matrix = cosine_distances(embeddings)
    i_indices, j_indices = np.triu_indices(len(embeddings), k=1)

    # Initialize counters
    TA_count = TR_count = FA_count = FR_count = 0
    total_genuine = 0  # Pairs of the same person
    total_impostor = 0 # Pairs of different people

    # 3. Evaluate pairs
    for k in range(len(i_indices)):
        i, j = i_indices[k], j_indices[k]
        dist = dist_matrix[i, j]
        same_person = (labels[i] == labels[j])
        predict_same = (dist < THRESHOLD)

        if same_person:
            total_genuine += 1
            if predict_same: 
                TA_count += 1  # True Accept
            else: 
                FR_count += 1  # False Reject
        else:
            total_impostor += 1
            if not predict_same: 
                TR_count += 1  # True Reject
            else: 
                FA_count += 1  # False Accept

    # 4. Calculate Rates (Percentages)
    ta_rate = TA_count / total_genuine if total_genuine > 0 else 0
    fr_rate = FR_count / total_genuine if total_genuine > 0 else 0
    tr_rate = TR_count / total_impostor if total_impostor > 0 else 0
    fa_rate = FA_count / total_impostor if total_impostor > 0 else 0

    # 5. Output Report
    print("=== Tuesday Report: 10 Subjects (ArcFace) ===")
    print(f"Total Genuine Pairs:  {total_genuine}")
    print(f"Total Impostor Pairs: {total_impostor}")
    print(f"Threshold Used:       {THRESHOLD}")
    print("-" * 45)
    print(f"TA (True Accept) Rate:  {ta_rate:.4f}  ({ta_rate*100:.2f}%)")
    print(f"TR (True Reject) Rate:  {tr_rate:.4f}  ({tr_rate*100:.2f}%)")
    print(f"FA (False Accept) Rate: {fa_rate:.4f}  ({fa_rate*100:.2f}%)")
    print(f"FR (False Reject) Rate: {fr_rate:.4f}  ({fr_rate*100:.2f}%)")

if __name__ == "__main__":
    run_10_subject_report()