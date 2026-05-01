import os
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_distances

# --- Configuration ---
MALE_DIR = "data/demographics/male"
FEMALE_DIR = "data/demographics/female"
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68

def get_embeddings_for_group(directory):
    """Extracts embeddings and labels for a demographic group."""
    embeddings = []
    labels = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return np.array([]), np.array([])

    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if not os.path.isdir(person_dir): continue
        
        for img in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img)
            try:
                rep = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
                embeddings.append(rep)
                labels.append(person)
            except Exception:
                continue
                
    return np.array(embeddings), np.array(labels)

def run_evaluation(enroll_name, enroll_dir, probe_name, probe_dir):
    print(f"\n=== Experiment: Enroll {enroll_name} | Probe {probe_name} ===")
    
    enroll_emb, enroll_labels = get_embeddings_for_group(enroll_dir)
    probe_emb, probe_labels = get_embeddings_for_group(probe_dir)
    
    if len(enroll_emb) == 0 or len(probe_emb) == 0:
        return

    # Calculate distance matrix between probe images and enrolled images
    dist_matrix = cosine_distances(probe_emb, enroll_emb)
    
    TA = TR = FA = FR = 0
    total_genuine = 0
    total_impostor = 0

    # Iterate through every probe against every enrolled image
    for i in range(len(probe_emb)):
        for j in range(len(enroll_emb)):
            # Skip comparing an exact image file to itself if testing same group
            if enroll_dir == probe_dir and i == j:
                continue 
                
            dist = dist_matrix[i, j]
            same_person = (probe_labels[i] == enroll_labels[j])
            predict_same = (dist < THRESHOLD)

            if same_person:
                total_genuine += 1
                if predict_same: TA += 1
                else: FR += 1
            else:
                total_impostor += 1
                if not predict_same: TR += 1
                else: FA += 1

    # Calculate Rates
    accuracy = (TA + TR) / (total_genuine + total_impostor) if (total_genuine + total_impostor) > 0 else 0
    far = FA / total_impostor if total_impostor > 0 else 0
    frr = FR / total_genuine if total_genuine > 0 else 0

    print(f"Total Genuine Pairs:  {total_genuine}")
    print(f"Total Impostor Pairs: {total_impostor}")
    print(f"Accuracy: {accuracy*100:.2f}% | FAR: {far*100:.2f}% | FRR: {frr*100:.2f}%")

if __name__ == "__main__":
    print(f"Extracting features using {MODEL_NAME}...")
    
    # 1. Enroll Female, Probe Female
    run_evaluation("Female", FEMALE_DIR, "Female", FEMALE_DIR)
    
    # 2. Enroll Male, Probe Male
    run_evaluation("Male", MALE_DIR, "Male", MALE_DIR)
    
    # 3. Enroll Female, Probe Male (Cross-Demographic)
    run_evaluation("Female", FEMALE_DIR, "Male", MALE_DIR)
    
    # 4. Enroll Male, Probe Female (Cross-Demographic)
    run_evaluation("Male", MALE_DIR, "Female", FEMALE_DIR)