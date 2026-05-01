import os
import shutil
from deepface import DeepFace

# --- Configuration ---
SOURCE_DIR = "data\\experiment_1000"  # Change this to your actual subset folder
BASE_OUT_DIR = "data\\demographics"

def auto_sort_demographics():
    male_dir = os.path.join(BASE_OUT_DIR, "Male")
    female_dir = os.path.join(BASE_OUT_DIR, "Female")
    
    os.makedirs(male_dir, exist_ok=True)
    os.makedirs(female_dir, exist_ok=True)
    
    people = [p for p in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, p))]
    print(f"Starting automated gender classification for {len(people)} subjects...")
    
    male_count = 0
    female_count = 0
    
    for person in people:
        person_dir = os.path.join(SOURCE_DIR, person)
        images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        
        if not images:
            continue
            
        # We only need to check the first image of the person to classify them
        img_path = os.path.join(person_dir, images[0])
        
        try:
            # Tell DeepFace to only run the 'gender' model to save time
            result = DeepFace.analyze(img_path, actions=['gender'], enforce_detection=False)
            
            # DeepFace returns a list of dictionaries, we just need the first face
            if isinstance(result, list):
                result = result[0]
                
            dominant_gender = result['dominant_gender'] # Returns 'Man' or 'Woman'
            
            # Determine destination folder
            if dominant_gender == 'Man':
                dest_dir = os.path.join(male_dir, person)
                male_count += 1
            else:
                dest_dir = os.path.join(female_dir, person)
                female_count += 1
                
            # Copy the person's folder and all their images to the new demographic folder
            shutil.copytree(person_dir, dest_dir, dirs_exist_ok=True)
            
        except Exception as e:
            print(f"Skipping {person} due to analysis error...")
            
    print("\n=== Sorting Complete ===")
    print(f"Identified {male_count} Male subjects")
    print(f"Identified {female_count} Female subjects")
    print(f"Folders successfully built in {BASE_OUT_DIR}")

if __name__ == "__main__":
    auto_sort_demographics()