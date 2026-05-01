import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
BASE_EMBEDDINGS_DIR = "data/experiment_embeddings"
MODELS = ["ArcFace"]  # Updated to only run ArcFace
MAX_SUBJECTS = 873

def load_embeddings(model_name, max_subjects):
    model_dir = os.path.join(BASE_EMBEDDINGS_DIR, model_name)
    db = []
    labels = []
    
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".npy")])
    for file in files[:max_subjects]:
        person = file.replace(".npy", "")
        arr = np.load(os.path.join(model_dir, file))
        for emb in arr:
            db.append(emb)
            labels.append(person)
            
    return np.array(db), np.array(labels)

def plot_roc_curves():
    plt.figure(figsize=(8, 6))
    
    for model in MODELS:
        print(f"Calculating ROC for {model}...")
        embeddings, labels = load_embeddings(model, MAX_SUBJECTS)
        
        if len(embeddings) == 0:
            print(f"No embeddings found for {model}. Skipping.")
            continue
            
        dist_matrix = cosine_distances(embeddings)
        i_indices, j_indices = np.triu_indices(len(embeddings), k=1)
        
        distances = dist_matrix[i_indices, j_indices]
        y_true = (labels[i_indices] == labels[j_indices]).astype(int)
        y_scores = 1 - distances
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plotting ArcFace with a distinct color
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model} (AUC = {roc_auc:.4f})')

    # --- Plot Formatting ---
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - ArcFace', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    out_path = "data/roc_curve_arcface.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ ROC curve generated and saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_roc_curves()