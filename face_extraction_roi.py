import cv2
import numpy as np
import os
import json
from glob import glob

# =====================================================
# CONFIGURATION
# =====================================================
TEMPLATE_PATH = "images/est_id/00.cg0001/000053.jpg"
DATASET_DIR = "images/est_id"
OUT_DIR = "extracted_faces_roi"
ROI_JSON = "roi.json"

ORB_FEATURES = 3000
MIN_MATCHES = 20
RATIO_TEST = 0.75

MAX_DISPLAY_WIDTH = 1200

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# LOAD TEMPLATE
# =====================================================
template = cv2.imread(TEMPLATE_PATH)
if template is None:
    raise FileNotFoundError("Template introuvable")

H_T, W_T = template.shape[:2]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# =====================================================
# ROI SELECTION (une seule fois)
# =====================================================
if not os.path.exists(ROI_JSON):
    scale = min(1.0, MAX_DISPLAY_WIDTH / W_T)

    display = cv2.resize(
        template,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.namedWindow("Selection ROI PHOTO", cv2.WINDOW_NORMAL)
    roi_small = cv2.selectROI(
        "Selection ROI PHOTO",
        display,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyAllWindows()

    if roi_small == (0, 0, 0, 0):
        raise RuntimeError("ROI non selectionne")

    roi = (
        int(roi_small[0] / scale),
        int(roi_small[1] / scale),
        int(roi_small[2] / scale),
        int(roi_small[3] / scale)
    )

    with open(ROI_JSON, "w") as f:
        json.dump(
            {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
            f,
            indent=2
        )

# =====================================================
# LOAD ROI
# =====================================================
with open(ROI_JSON, "r") as f:
    r = json.load(f)

ROI = (r["x"], r["y"], r["w"], r["h"])

# =====================================================
# ORB TEMPLATE
# =====================================================
orb = cv2.ORB_create(ORB_FEATURES)
kp_t, des_t = orb.detectAndCompute(template_gray, None)

if des_t is None:
    raise RuntimeError("Aucun point ORB sur le template")

# =====================================================
# UTILS
# =====================================================
def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

# =====================================================
# PROCESS DATASET
# =====================================================
for folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_paths.extend(glob(os.path.join(folder_path, ext)))

    images = [cv2.imread(p) for p in image_paths if cv2.imread(p) is not None]

    if not images:
        continue

    ref_img = max(images, key=sharpness)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    kp_r, des_r = orb.detectAndCompute(ref_gray, None)
    if des_r is None:
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_r, des_t, k=2)

    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
    if len(good) < MIN_MATCHES:
        continue

    src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        continue

    out_folder = os.path.join(OUT_DIR, folder)
    os.makedirs(out_folder, exist_ok=True)

    crops = []

    for img in images:
        aligned = cv2.warpPerspective(img, H, (W_T, H_T))
        x, y, w, h = ROI

        if x < 0 or y < 0 or x + w > W_T or y + h > H_T:
            continue

        crop = aligned[y:y+h, x:x+w]
        if crop.size > 0:
            crops.append(crop)

    if not crops:
        continue

    best = max(crops, key=sharpness)
    cv2.imwrite(os.path.join(out_folder, "photo_id.jpg"), best)

print("Pipeline termine")
