from ultralytics import YOLO
import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
import lpips
from PIL import Image
import torchvision.transforms as T
import numpy as np

# Load the trained YOLOv11 model
model = YOLO("/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/TrainerV4/runs/NID_OBB_3/weights/best.pt")

# Path to the input image
image_path = '/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/TrainerV4/V4_FinalFlow/Production_images_shifted/puzzle_8.jpeg'

# Extract the ID from the filename (e.g., "Production_1.jpeg" → "1.jpeg")
production_id = os.path.basename(image_path).split("_")[1].split(".")[0]  # Extract "1"
comparison_folder = '/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Final_flow/Images_for_comparison'
comparison_image_path = os.path.join(comparison_folder, f"{production_id}.jpeg")

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

            cropped_image_variable = cropped_image

            # Print the path of the saved cropped image
            print(f"Cropped and rotated image saved at: {cropped_image_path}")
            print(f"Confidence Score: {conf}")
    else:
        print("No oriented bounding boxes detected.")
        


# === CLIP Comparison ===

# 1. Load the CLIP vision model and its processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# 2. Load the cropped image and the comparison image
img_paths = [
    cropped_image_path,  # Cropped image
    comparison_image_path  # Corresponding image from the folder
]

images = [Image.open(p).convert("RGB") for p in img_paths]

# 3. Preprocess both images in a single batch
inputs = processor(images=images, return_tensors="pt")

# 4. Compute the image embeddings
with torch.no_grad():
    outputs = clip_model(**inputs)
    embeddings = outputs.pooler_output   # shape: [2, 512]

# 5. Measure cosine similarity between the two 512-D vectors
emb1, emb2 = embeddings[0], embeddings[1]
cos_sim = torch.nn.functional.cosine_similarity(
    emb1.unsqueeze(0),    # shape [1,512]
    emb2.unsqueeze(0),    # shape [1,512]
).item()

# === LPIPS Comparison ===

# 1. Load the LPIPS metric (AlexNet backbone)
loss_fn = lpips.LPIPS(net='alex')  # or net='vgg'

# 2. Preprocessing: both images to [-1,+1] tensors
pre = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img0 = pre(images[0]).unsqueeze(0)  # Cropped image
img1 = pre(images[1]).unsqueeze(0)  # Comparison image

# 3. Compute LPIPS distance
with torch.no_grad():
    dist = loss_fn(img0, img1).item()

# === Homography Comparison ===

def compute_similarity_with_homography(image1_path, image2_path, output_image_path=None, show_result=False):
    # Load images
    img1_color = cv2.imread(image1_path)
    img2_color = cv2.imread(image2_path)
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    if img1_color is None or img2_color is None:
        raise ValueError("One or both image paths are invalid or images couldn't be loaded.")

    # ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()
        inliers = np.sum(matches_mask)
        total = len(good_matches)

        # Similarity score: how many matches are good under homography
        similarity = inliers / total * 100
    else:
        similarity = 0

    # Visualization
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask if len(good_matches) > 4 else None,
        flags=2,
    )

    img_matches = cv2.drawMatches(img1_color, keypoints1, img2_color, keypoints2, good_matches, None, **draw_params)

    if show_result:
        cv2.imshow("Homography Matches", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_image_path:
        cv2.imwrite(output_image_path, img_matches)

    return similarity

# Compute Homography similarity
homography_similarity = compute_similarity_with_homography(
    cropped_image_path, comparison_image_path, output_image_path=None, show_result=False
)

# === Present Results in a Matrix ===
print("\n=== Image Comparison Results ===")
print(f"{'Metric':<20}{'Score':<10}")
print(f"{'-'*30}")
print(f"{'CLIP-score:':<20}{cos_sim:.4f}")
print(f"{'LPIPS-score:':<20}{dist:.4f}")
print(f"{'Homography-score:':<20}{homography_similarity:.2f}%")

# Interpret Results
if cos_sim > 0.8:
    print("→ CLIP: Images are nearly identical")
else:
    print("→ CLIP: Images differ")

if dist < 0.4:  # Lower LPIPS distance indicates higher similarity
    print("→ LPIPS: Images are nearly identical")
else:
    print("→ LPIPS: Images differ perceptually")

if homography_similarity > 50:  # Threshold for homography similarity
    print("→ Homography: Images are highly similar")
else:
    print("→ Homography: Images differ significantly")