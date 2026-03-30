import cv2
import os
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Configuration
EMBEDDINGS_DIR = "data/embeddings/enrollment"
PROCESSED_ENROLLMENT_DIR = "data/processed/enrollment"
TEST_IMAGE_PATH = "data/processed/test/test_face.jpg"
MODEL_NAME = "Facenet512"
THRESHOLD = 0.3  # Tuned threshold for Facenet512

def load_embeddings():
    """Load pre-computed embeddings."""
    db = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            identity = os.path.splitext(file)[0]
            db[identity] = np.load(os.path.join(EMBEDDINGS_DIR, file))
    return db

def run_visual_demo():
    print("Loading database...")
    db = load_embeddings()
    
    print("Processing test image...")
    try:
        # Extract embedding for the test image
        test_rep = DeepFace.represent(
            img_path=TEST_IMAGE_PATH,
            model_name=MODEL_NAME,
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        print(f"Error processing test image: {e}")
        return

    test_embedding = np.array(test_rep)
    
    # Find the best match
    best_identity = None
    best_score = float("inf")

    for identity, embeddings in db.items():
        for emb in embeddings:
            score = cosine(test_embedding, emb)
            if score < best_score:
                best_score = score
                best_identity = identity

    decision = "ACCEPT" if best_score < THRESHOLD else "REJECT"
    color = (0, 255, 0) if decision == "ACCEPT" else (0, 0, 255) # Green for accept, Red for reject

    # --- GUI VISUALIZATION ---
    # 1. Load the test image
    img_test = cv2.imread(TEST_IMAGE_PATH)
    img_test = cv2.resize(img_test, (300, 300))

    # 2. Load the matched database image (if found)
    img_match = np.zeros((300, 300, 3), dtype=np.uint8) # Default to black square
    if best_identity:
        person_dir = os.path.join(PROCESSED_ENROLLMENT_DIR, best_identity)
        if os.path.exists(person_dir):
            images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if images:
                match_path = os.path.join(person_dir, images[0])
                img_match = cv2.imread(match_path)
                img_match = cv2.resize(img_match, (300, 300))

    # 3. Stitch images side-by-side
    display_img = np.hstack((img_test, img_match))
    display_img = cv2.copyMakeBorder(display_img, 100, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))

    # 4. Add Text Overlays
    cv2.putText(display_img, "TEST FACE", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_img, "DATABASE MATCH", (360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    result_text = f"Result: {decision} | Match: {best_identity} | Score: {best_score:.3f}"
    if decision == "REJECT":
        result_text = f"Result: REJECT (Unknown) | Closest: {best_identity} | Score: {best_score:.3f}"
        
    cv2.putText(display_img, result_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 5. Display the result
    cv2.imshow("Face Recognition Demo", display_img)
    print("\nPress any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_demo()