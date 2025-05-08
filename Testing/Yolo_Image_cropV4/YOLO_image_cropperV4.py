from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the trained YOLOv8 model
model = YOLO("../data/runs/NID_OBB_3/weights/best.pt")

# Path to the input image
image_path = '../data/Production_images_true_shifted/puzzle_399.jpeg'

# Run inference on the image
results = model.predict(source=image_path, save=False, conf=0.5)

# Load the image using OpenCV
image = cv2.imread(image_path)

for result in results:
    if result.obb is not None:
        for obb, conf in zip(result.obb.xyxyxyxy, result.obb.conf):  # Iterate through OBBs and their confidence scores
            # Convert the OBB tensor to a NumPy array and reshape it to (4, 2) for the four points
            polygon = np.array(obb.cpu().numpy(), dtype=np.float32).reshape((4, 2))

            # Get the bounding rectangle and rotation angle
            rect = cv2.minAreaRect(polygon)  # Get the minimum area rectangle
            box = cv2.boxPoints(rect)  # Get the four corners of the rectangle
            box = np.int32(box)  # Convert to integer

            # Extract the width, height, and angle of rotation
            width, height = int(rect[1][0]), int(rect[1][1])
            angle = rect[2]

            # Correct the angle for upright rotation
            if width < height:
                angle += 90

            # Get the rotation matrix for the detected region
            center = tuple(map(int, rect[0]))  # Center of the rectangle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Warp the image to align the detected region upright
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # Crop the upright bounding box from the rotated image
            x, y, w, h = cv2.boundingRect(np.int32(cv2.transform(np.array([polygon]), rotation_matrix)[0]))
            cropped_image = rotated_image[y:y+h, x:x+w]

            # Rotate the cropped image 90 degrees to the right
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

            # Save the cropped and rotated image with "_YOLOCroppedRotated" added to the filename
            base_name, ext = os.path.splitext(image_path)
            cropped_image_path = f"{base_name}_YOLOCroppedRotated{ext}"
            cv2.imwrite(cropped_image_path, cropped_image)

            # Print the path of the saved cropped image
            print(f"Cropped and rotated image saved at: {cropped_image_path}")
            print(f"Confidence Score: {conf}")
    else:
        print("No oriented bounding boxes detected.")