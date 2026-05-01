import os
import cv2
import random
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- Configuration ---
DATA_DIR = "data/experiment_1000"  # Or whichever folder has your subset
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68
NUM_PAIRS = 50  # Test on 50 random genuine pairs

def apply_darkness(img):
    """Simulate extreme low-light conditions."""
    return cv2.convertScaleAbs(img, alpha=0.3, beta=0)

def apply_low_res(img):
    """Simulate a cheap, low-resolution security camera."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (20, 20), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_synthetic_mask(img):
    """Draw a black rectangle over the bottom half of the face."""
    masked = img.copy()
    h, w = masked.shape[:2]
    cv2.rectangle(masked, (0, int(h * 0.55)), (w, h), (0, 0, 0), -1)
    return masked

def get_embedding(img_array):
    try:
        rep = DeepFace.represent(img_path=img_array, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        return np.array(rep)
    except Exception:
        return None

def run_experiment():
    people = [p for p in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, p))]
    random.shuffle(people)
    
    results = {"Baseline": [], "Dark": [], "LowRes": [], "Masked": []}
    
    print(f"Running robustness tests on {NUM_PAIRS} pairs...")
    
    pairs_tested = 0
    for person in people:
        if pairs_tested >= NUM_PAIRS: break
        
        person_dir = os.path.join(DATA_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        if len(images) < 2: continue
            
        # Load two different images of the same person
        img1 = cv2.imread(os.path.join(person_dir, images[0]))
        img2 = cv2.imread(os.path.join(person_dir, images[1]))
        
        # Create variations of the second image
        img2_dark = apply_darkness(img2)
        img2_lowres = apply_low_res(img2)
        img2_mask = apply_synthetic_mask(img2)
        
        # Get embeddings
        emb1 = get_embedding(img1)
        if emb1 is None: continue
            
        dist_base = cosine(emb1, get_embedding(img2))
        dist_dark = cosine(emb1, get_embedding(img2_dark))
        dist_lowres = cosine(emb1, get_embedding(img2_lowres))
        dist_mask = cosine(emb1, get_embedding(img2_mask))
        
        results["Baseline"].append(dist_base)
        results["Dark"].append(dist_dark)
        results["LowRes"].append(dist_lowres)
        results["Masked"].append(dist_mask)
        
        pairs_tested += 1

    # --- Print Report ---
    print("\n=== Robustness & Bias Results ===")
    print(f"Threshold: {THRESHOLD} (Scores above this are REJECTED)\n")
    
    for condition, scores in results.items():
        avg_dist = np.mean(scores)
        frr = sum(1 for s in scores if s > THRESHOLD) / len(scores) * 100
        print(f"[{condition}]")
        print(f"  Average Distance: {avg_dist:.4f}")
        print(f"  False Rejection Rate: {frr:.1f}%\n")

if __name__ == "__main__":
    run_experiment()