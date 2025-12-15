import os
import cv2
import mediapipe as mp


# CONFIGURATION


INPUT_DIR = r"images/lva_passport"   
OUTPUT_DIR = "extracted_faces"

os.makedirs(OUTPUT_DIR, exist_ok=True)


mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


# FONCTION EXTRACTION


def extract_face_from_image(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[X] Image illisible : {image_path}")
        return False

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        print(f"[X] Aucun visage dans {image_path}")
        return False


    best_det = None
    max_area = 0

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        area = bbox.width * bbox.height
        if area > max_area:
            max_area = area
            best_det = det

    bbox = best_det.location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # Ajouter une petite marge
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return False

    cv2.imwrite(save_path, face)
    print(f"[✔] Face extraite : {save_path}")
    return True


# PARCOURIR LE DATASET


for folder in os.listdir(INPUT_DIR):
    folder_path = os.path.join(INPUT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)

            save_name = f"{folder}_{os.path.splitext(file)[0]}_face.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            extract_face_from_image(img_path, save_path)

print("Extraction terminée")
