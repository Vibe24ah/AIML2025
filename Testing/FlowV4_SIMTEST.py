import os
import cv2
import torch
import lpips
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm  # Add this import at the top of the script

# Paths to the folders
folders = {
    "true": {
        "production": '../data/Production_images_true_shifted',
        "comparison": '../data/Images_for_comparison_true'
    },
    "false": {
        "production": '../data/Production_images_false_shifted',
        "comparison": '../data/Images_for_comparison_false'
    }
}

# Load the trained YOLOv8 model
model = YOLO('../Models/runs/NID_OBB_3/weights/best.pt')

# Initialize LPIPS metric
loss_fn = lpips.LPIPS(net='alex')  # or net='vgg'

# CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# Helper function: Compute Homography similarity
def compute_similarity_with_homography(image1_path, image2_path):
    img1_color = cv2.imread(image1_path)
    img2_color = cv2.imread(image2_path)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        inliers = np.sum(matches_mask)
        total = len(good_matches)
        similarity = inliers / total * 100
    else:
        similarity = 0

    return similarity

# Helper function: Process a single pair of images
def process_image_pair(production_path, comparison_path):
    # Load the production image
    image = cv2.imread(production_path)

    # Run YOLO inference
    results = model.predict(source=production_path, save=False)
    cropped_image_path = None  # Initialize variable for cropped image path
    yolo_confidence = None  # Initialize variable for YOLO confidence score

    for result in results:
        if result.obb is not None:
            for obb, conf in zip(result.obb.xyxyxyxy, result.obb.conf):  # Iterate through OBBs and their confidence scores
                yolo_confidence = conf.item()  # Save the confidence score

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

                # Ensure the cropped image is not empty
                if cropped_image is None or cropped_image.size == 0:
                    print("Cropped image is empty. Skipping this result.")
                    continue

                # Save the cropped and rotated image with "_YOLOCroppedRotated" added to the filename
                base_name, ext = os.path.splitext(production_path)
                if not base_name.endswith("_YOLOCroppedRotated"):
                    cropped_image_path = f"{base_name}_YOLOCroppedRotated{ext}"
                    cv2.imwrite(cropped_image_path, cropped_image)
                    print(f"Cropped and rotated image saved at: {cropped_image_path}")
                else:
                    cropped_image_path = production_path  # Use the existing cropped image

                print(f"Confidence Score: {conf}")
        else:
            print("No oriented bounding boxes detected.")

    # Ensure a cropped image exists before proceeding
    if cropped_image_path is None:
        print("No cropped image available for comparison.")
        return None, None, None, None

    # CLIP similarity
    images = [Image.open(cropped_image_path).convert("RGB"), Image.open(comparison_path).convert("RGB")]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model(**inputs)
        embeddings = outputs.pooler_output
    cos_sim = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

    # LPIPS similarity
    pre = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img0 = pre(images[0]).unsqueeze(0)
    img1 = pre(images[1]).unsqueeze(0)
    with torch.no_grad():
        lpips_score = loss_fn(img0, img1).item()

    # Homography similarity
    homography_score = compute_similarity_with_homography(cropped_image_path, comparison_path)

    return cos_sim, lpips_score, homography_score, yolo_confidence

# Helper function to extract numeric ID from a filename
def extract_id(filename):
    return int(''.join(filter(str.isdigit, filename)))

# Main processing loop
results = []
for label, paths in folders.items():
    production_folder = paths["production"]
    comparison_folder = paths["comparison"]

    # Get all production and comparison images
    production_files = [f for f in os.listdir(production_folder) if f.lower().endswith(".jpeg")]
    comparison_files = [f for f in os.listdir(comparison_folder) if f.lower().endswith(".jpeg")]

    # Sort files to ensure consistent order
    production_files.sort(key=extract_id)
    comparison_files.sort(key=extract_id)

    # Create a dictionary for comparison files by ID
    comparison_dict = {extract_id(f): f for f in comparison_files}

    # Process each production image
    for production_file in tqdm(production_files, desc=f"Processing {label} folder"):
        production_id = extract_id(production_file)
        if production_id not in comparison_dict:
            print(f"No matching comparison file for production ID {production_id}. Skipping.")
            continue

        production_path = os.path.join(production_folder, production_file)
        comparison_path = os.path.join(comparison_folder, comparison_dict[production_id])

        # Compute scores
        cos_sim, lpips_score, homography_score, yolo_confidence = process_image_pair(production_path, comparison_path)

        # Append results
        results.append({
            "Label": label,
            "Production Image": production_file,
            "Comparison Image": comparison_dict[production_id],
            "CLIP Score": cos_sim,
            "LPIPS Score": lpips_score,
            "Homography Score": homography_score,
            "YOLO Confidence": yolo_confidence
        })

# Save results to a CSV file
df = pd.DataFrame(results)
output_csv_path = '../data/Final_pipeline/results.csv'
df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")