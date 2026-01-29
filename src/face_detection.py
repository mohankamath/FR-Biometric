import os
import cv2
from deepface import DeepFace

def detect_and_crop(image_path, output_path, detector="opencv"):
    """
    Detects a face in an image and saves the cropped face.
    Returns True if detection succeeds, False otherwise.
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=image_path,
            target_size=(224, 224),
            detector_backend=detector,
            enforce_detection=True
        )

        face = detections[0]["face"]
        face = (face * 255).astype("uint8")

        cv2.imwrite(output_path, face)
        return True

    except Exception as e:
        return False
