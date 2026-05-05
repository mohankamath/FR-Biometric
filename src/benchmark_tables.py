import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
BASE_EMBEDDINGS_DIR = "data/experiment_embeddings"
MODELS = ["ArcFace", "VGG-Face", "Facenet", "Facenet512", "SFace"]
MAX_SUBJECTS = 873 # Your final dataset size
THRESHOLD = 0.68 # The static threshold chosen for ArcFace

def load_embeddings(model_name):
    """Loads pre-calculated embeddings into memory."""
    model_dir = os.path.join(BASE_EMBEDDINGS_DIR, model_name)
    db = []
    labels = []
    
    if not os.path.exists(model_dir):
        return np.array([]), np.array([])
        
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
    for file in files[:MAX_SUBJECTS]:
        person = file.replace(".npy", "")
        arr = np.load(os.path.join(model_dir, file))
        for emb in arr:
            db.append(emb)
            labels.append(person)
            
    return np.array(db), np.array(labels)

def run_benchmarks():
    print("=== BEGINNING BENCHMARK EVALUATION ===\n")
    
    for model in MODELS:
        print(f"Loading {model}...")
        embeddings, labels = load_embeddings(model)
        
        if len(embeddings) == 0:
            print(f"  -> Skipping {model} (No embeddings found)\n")
            continue
            
        # Start timer
        start_time = time.time()
        
        # 1. Calculate all pairwise distances rapidly
        dist_matrix = cosine_distances(embeddings)
        i_indices, j_indices = np.triu_indices(len(embeddings), k=1)
        distances = dist_matrix[i_indices, j_indices]
        
        # Stop timer
        end_time = time.time()
        compute_time = end_time - start_time
        
        # 2. Define Ground Truth: True if same person, False if different
        y_true = (labels[i_indices] == labels[j_indices])
        
        # 3. Predict based on Threshold
        # Note: If you want to use a dynamic threshold for each model, you would calculate 
        # it via ROC here. But for Table 2, we are evaluating how the ArcFace threshold (0.68) 
        # applies, or evaluating the default distance. We will use 0.68 for ArcFace.
        current_threshold = THRESHOLD if model == "ArcFace" else 0.40 # Standard rough default for others
        y_pred = (distances < current_threshold)
        
        # 4. Calculate Metrics
        total_pairs = len(y_true)
        total_genuine = np.sum(y_true)
        total_impostor = total_pairs - total_genuine
        
        # True Accepts (TA), False Rejects (FR)
        genuine_distances = distances[y_true]
        TA = np.sum(genuine_distances < current_threshold)
        FR = total_genuine - TA
        
        # True Rejects (TR), False Accepts (FA)
        impostor_distances = distances[~y_true]
        TR = np.sum(impostor_distances >= current_threshold)
        FA = total_impostor - TR
        
        # Rates
        accuracy = (TA + TR) / total_pairs * 100
        far = (FA / total_impostor) * 100 if total_impostor > 0 else 0
        frr = (FR / total_genuine) * 100 if total_genuine > 0 else 0
        
        # --- Output for Tables ---
        if model == "ArcFace":
            print("\n" + "="*40)
            print("DATA FOR TABLE 1 (OVERALL METRICS)")
            print("="*40)
            print(f"Overall Accuracy: {accuracy:.2f}% | {TA + TR} out of {total_pairs}")
            print(f"False Rejection Rate (FRR): {frr:.2f}% | {FR} out of {total_genuine}")
            print(f"False Acceptance Rate (FAR): {far:.2f}% | {FA} out of {total_impostor}")
            print("="*40 + "\n")
            
        print(f"DATA FOR TABLE 2: {model}")
        print(f"  Avg Accuracy: {accuracy:.2f}%")
        print(f"  Avg FAR: {far:.2f}%")
        print(f"  Avg FRR: {frr:.2f}%")
        print(f"  Avg Time: {compute_time:.4f} seconds\n")

if __name__ == "__main__":
    run_benchmarks()