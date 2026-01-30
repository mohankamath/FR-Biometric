import os
import cv2
from deepface import DeepFace

def detect_and_crop(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=False
        )
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        detections = []

    # Count existing images so we don't overwrite
    existing_faces = len([
        f for f in os.listdir(output_dir)
        if f.lower().endswith(".jpg")
    ])
    face_idx = existing_faces

    # Fallback: no faces detected
    if len(detections) == 0:
        print(f"No faces detected in {image_path}. Saving original image as fallback.")
        img = cv2.imread(image_path)
        if img is not None:
            out_path = os.path.join(output_dir, f"face_{face_idx}.jpg")
            cv2.imwrite(out_path, img)
        return

    for face in detections:
        face_img = face["face"]

        # DeepFace may return float64 in [0,1]
        if face_img.dtype != "uint8":
            face_img = (face_img * 255).clip(0, 255).astype("uint8")

        # Convert RGB → BGR for OpenCV
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, f"face_{face_idx}.jpg")
        cv2.imwrite(out_path, face_img)

        face_idx += 1
