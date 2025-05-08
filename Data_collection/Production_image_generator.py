from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar

# Define the resize and crop function
def resize_and_crop_to_cover(img, target_w, target_h):
    h, w = img.shape[:2]
    img_aspect = w / h
    target_aspect = target_w / target_h

    # Resize while preserving aspect ratio
    if img_aspect > target_aspect:
        # Image is wider — scale height, then crop width
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h))
        x_offset = (new_w - target_w) // 2
        cropped = resized[:, x_offset:x_offset + target_w]
    else:
        # Image is taller — scale width, then crop height
        scale = target_w / w
        new_h = int(h * scale)
        resized = cv2.resize(img, (target_w, new_h))
        y_offset = (new_h - target_h) // 2
        cropped = resized[y_offset:y_offset + target_h, :]

    return cropped

# Define the crop function to center-crop the image to a square and downscale to 640x640 pixels
def crop_and_downscale_to_640x640(img):
    target_size = 640
    h, w = img.shape[:2]

    # Calculate the crop dimensions to make the image square
    if h > w:
        # Image is taller — crop height
        crop_size = w
        y_offset = (h - crop_size) // 2
        x_offset = 0
    else:
        # Image is wider — crop width
        crop_size = h
        x_offset = (w - crop_size) // 2
        y_offset = 0

    # Perform the crop to make the image square
    cropped = img[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size]

    # Downscale the square image to 640x640 pixels
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized

# Load the template image
template = cv2.imread("/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/IMG_9567.jpeg")

# Coordinates of the four corners of the image area on the puzzle box
# Order: top-left, top-right, bottom-right, bottom-left
dst_points = np.array([
    [926, 1050],
    [1993, 1056],
    [1970, 2640],
    [929, 2624]
], dtype="float32")

# Compute width and height of the replacement image based on distances
width = int(max(
    np.linalg.norm(dst_points[0] - dst_points[1]),
    np.linalg.norm(dst_points[2] - dst_points[3])
))
height = int(max(
    np.linalg.norm(dst_points[0] - dst_points[3]),
    np.linalg.norm(dst_points[1] - dst_points[2])
))

# Source points (corners of the replacement image)
src_points = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Load and replace with images from the folder
replacement_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Fred_pictures"  # Folder with replacement photos
output_folder = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Training_V2/Test_images_2"
os.makedirs(output_folder, exist_ok=True)

# Get the list of replacement images
replacement_files = [f for f in os.listdir(replacement_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in tqdm(replacement_files, desc="Processing images"):
    replacement_img_path = os.path.join(replacement_folder, filename)
    replacement_image = cv2.imread(replacement_img_path)
    
    # Resize replacement to match target area
    resized = resize_and_crop_to_cover(replacement_image, width, height)

    # Add a red box to the resized replacement image (!!!USED FOR ITERATION 1 ONLY!!!)
    #cv2.rectangle(resized, (0, 0), (width - 1, height - 1), (0, 0, 255), 15)  # Red color, thickness 5

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the replacement image (with the red box) to match the puzzle area
    warped = cv2.warpPerspective(resized, matrix, (template.shape[1], template.shape[0]))

    # Create a mask from the warped image to blend properly
    mask = np.zeros_like(template, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(int), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    # Blend the warped image into the original template
    background = cv2.bitwise_and(template, mask_inv)
    combined = cv2.add(background, cv2.bitwise_and(warped, mask))

    # Crop and downscale the final combined image to 640x640 pixels
    final_output = crop_and_downscale_to_640x640(combined)

    # Save final output with the same name as the original file, prefixed with "puzzle_"
    base_name, ext = os.path.splitext(filename)
    output_filename = f"puzzle_{base_name}{ext}"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, final_output)

    # Generate YOLOv8 label file using dst_points
    label_path = os.path.join(output_folder, f"puzzle_{base_name}.txt")
    x_center = (dst_points[:, 0].mean()) / template.shape[1]
    y_center = (dst_points[:, 1].mean()) / template.shape[0]
    box_width = (max(dst_points[:, 0]) - min(dst_points[:, 0])) / template.shape[1]
    box_height = (max(dst_points[:, 1]) - min(dst_points[:, 1])) / template.shape[0]

    # Write the label file
    #with open(label_path, 'w') as f:
     #   f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")  # Class ID is 0

print("✅ Puzzle box images and YOLO labels generated successfully!")
