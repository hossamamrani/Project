from ultralytics import YOLO
from pathlib import Path

# Setup
MODEL = r"C:\Users\Republic Of Computer\Documents\document id\runs\pose\train2\weights\best.pt"
#TEST_DIR = r"C:\Users\Republic Of Computer\Documents\document id\yolo_dataset\test\images"
TEST_DIR = r"C:\Users\Republic Of Computer\Downloads\images"


# Get first image
first_image = sorted(list(Path(TEST_DIR).glob('*.JPG')))[-1]


# Run and save
model = YOLO(MODEL)
results = model.predict(first_image, save=True, conf=0.25)

# Show info
print(f"\n Tested: {first_image.name}")
print(f" Detections: {len(results[0].boxes)} person(s)")
print(f" Saved to: runs/pose/predict/{first_image.name}")
print(f"\nOpen that file to see the annotated image!")
