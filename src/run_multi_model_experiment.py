import os
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_distances

BASE_EMBEDDINGS_DIR = "data/experiment_embeddings"
RESULTS_FILE = "data/benchmark_results.csv"

MODELS = ["ArcFace", "Facenet512", "VGG-Face", "SFace"]
SUBSET_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Standard DeepFace thresholds for cosine distance
THRESHOLDS = {
    "ArcFace": 0.68,
    "Facenet512": 0.30,
    "VGG-Face": 0.40,
    "SFace": 0.593
}

def load_embeddings(model_name, max_subjects):
    model_dir = os.path.join(BASE_EMBEDDINGS_DIR, model_name)
    db = []
    labels = []
    
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
    
    for file in files[:max_subjects]:
        person = file.replace(".npy", "")
        arr = np.load(os.path.join(model_dir, file))
        
        # Using the exact fix from your summary to avoid flattening errors
        for emb in arr:
            db.append(emb)
            labels.append(person)
            
    return np.array(db), np.array(labels)

def run_benchmark():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Subjects", "Total_Pairs", "Accuracy", "FAR", "FRR"])

        for model in MODELS:
            print(f"\n--- Testing Model: {model} ---")
            threshold = THRESHOLDS.get(model, 0.5)

            for size in SUBSET_SIZES:
                embeddings, labels = load_embeddings(model, size)
                if len(embeddings) == 0:
                    continue
                
                # Optimized matrix calculation for all combinations
                dist_matrix = cosine_distances(embeddings)
                
                TP = TN = FP = FN = 0
                total_pairs = 0
                
                # Extract upper triangle indices (i < j) to mimic itertools.combinations
                i_indices, j_indices = np.triu_indices(len(embeddings), k=1)
                
                for k in range(len(i_indices)):
                    i, j = i_indices[k], j_indices[k]
                    total_pairs += 1
                    
                    dist = dist_matrix[i, j]
                    same_person = (labels[i] == labels[j])
                    predict_same = (dist < threshold)
                    
                    if same_person and predict_same:
                        TP += 1
                    elif same_person and not predict_same:
                        FN += 1
                    elif not same_person and predict_same:
                        FP += 1
                    else:
                        TN += 1

                accuracy = (TP + TN) / total_pairs if total_pairs > 0 else 0
                far = FP / (FP + TN) if (FP + TN) > 0 else 0
                frr = FN / (FN + TP) if (FN + TP) > 0 else 0
                
                print(f"Subjects: {size:4d} | Acc: {accuracy:.4f} | FAR: {far:.4f} | FRR: {frr:.4f}")
                writer.writerow([model, size, total_pairs, accuracy, far, frr])

if __name__ == "__main__":
    run_benchmark()
    print(f"\nResults saved to {RESULTS_FILE}")