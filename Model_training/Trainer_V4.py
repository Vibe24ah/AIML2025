import os
from ultralytics import YOLO


#model loading
#model = YOLO("yolo11n-obb.yaml")
#model = YOLO("yolo11n-obb.pt")
model = YOLO("yolo11n-obb.pt")

# Paths
output_dir = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/TrainerV4/runs"

# Train the model
results = model.train(
    data="/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/TrainerV4/Output_for_yolo.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name="NID_OBB_3",
    project=output_dir,
    device="cpu",  # Changed to CPU
    workers=8,
    save_period=1,
    save_dir=output_dir,
    exist_ok=True,
    max_det=1,
    patience = 4 # Number of epochs with no improvement after which training will be stopped
)


#validation

metrics= model.val(data="/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/TrainerV4/Output_for_yolo.yaml")