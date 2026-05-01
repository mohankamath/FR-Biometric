import os
import random
import shutil

# --- Configuration ---
RAW_DIR = "C:\\Users\\mohan\\scikit_learn_data\\lfw_home\\lfw_funneled"  # Change this to wherever your FULL dataset is
OUT_DIR = "data/experiment_1000"       # The new folder it will create
MIN_IMAGES = 3
MAX_IMAGES = 5
TARGET_NUM_PEOPLE = 1000

def create_subset():
    os.makedirs(OUT_DIR, exist_ok=True)
    eligible_people = []

    print(f"Scanning {RAW_DIR} for subjects with at least {MIN_IMAGES} images...")

    # 1. Filter for eligible people
    for person in os.listdir(RAW_DIR):
        person_path = os.path.join(RAW_DIR, person)
        if not os.path.isdir(person_path):
            continue

        # Count how many jpgs this person has
        images = [f for f in os.listdir(person_path) if f.endswith(".jpg")]

        if len(images) >= MIN_IMAGES:
            eligible_people.append((person, images))

    print(f"Found {len(eligible_people)} total eligible subjects.")

    # 2. Randomly select the requested amount (or as many as possible)
    random.shuffle(eligible_people)
    selected_people = eligible_people[:TARGET_NUM_PEOPLE]
    
    actual_count = len(selected_people)
    print(f"Randomly selected {actual_count} subjects for the experiment.")

    # 3. Copy the images over
    print(f"Copying files to {OUT_DIR}...")
    for person, images in selected_people:
        src_dir = os.path.join(RAW_DIR, person)
        dst_dir = os.path.join(OUT_DIR, person)
        os.makedirs(dst_dir, exist_ok=True)

        # Shuffle their images and take up to MAX_IMAGES
        random.shuffle(images)
        images_to_copy = images[:MAX_IMAGES]

        for img in images_to_copy:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

    print("\n✅ Subset creation complete!")
    if actual_count < TARGET_NUM_PEOPLE:
        print(f"Note: The dataset only had {actual_count} people with {MIN_IMAGES}+ images, so we used all of them.")

if __name__ == "__main__":
    create_subset()