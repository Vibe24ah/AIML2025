import os
import cv2
import numpy as np
from tqdm import tqdm

# Define the input and output folders
input_folder = "../Data/Production_images_true"
output_folder = "../data/Production_images_true_shifted"
os.makedirs(output_folder, exist_ok=True)

# Function to apply slight variations to an image
def augment_image(image):
    h, w = image.shape[:2]

    # Step 1: Zoom in slightly (3-5%)
    zoom_factor = np.random.uniform(1.03, 1.05)
    zoomed_h, zoomed_w = int(h * zoom_factor), int(w * zoom_factor)
    zoomed = cv2.resize(image, (zoomed_w, zoomed_h))

    # Random crop to original size after zoom
    x_offset = (zoomed_w - w) // 2
    y_offset = (zoomed_h - h) // 2
    cropped_zoomed = zoomed[y_offset:y_offset + h, x_offset:x_offset + w]

    # Step 2: Random rotation (between -20 and 20 degrees)
    angle = np.random.uniform(-20, 20)
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(cropped_zoomed, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Step 3: Random shift (crop a random side)
    shift_x = np.random.randint(0, int(w * 0.02))  # Max 2% shift horizontally
    shift_y = np.random.randint(0, int(h * 0.02))  # Max 2% shift vertically
    shifted = rotated[shift_y:h, shift_x:w]

    # Resize back to original size to ensure consistency
    final_image = cv2.resize(shifted, (w, h), interpolation=cv2.INTER_AREA)

    return final_image

# Get the list of images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Process each image with a progress bar
for filename in tqdm(image_files, desc="Augmenting images"):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # Read the image
    image = cv2.imread(input_path)

    # Apply augmentation
    augmented_image = augment_image(image)

    # Save the augmented image to the output folder
    cv2.imwrite(output_path, augmented_image)

print("Image augmentation complete! "Shifted" images saved.")
