import os
import shutil
import random

# Paths
input_images_folder = "../data/Output_for_yolo/train/images"
input_labels_folder = "../data/Output_for_yolo/val/labels"
output_folder = "../data/Output_for_yolo/Output_for_yolo_final"

# Create train/val folders
train_images = os.path.join(output_folder, "train/images")
train_labels = os.path.join(output_folder, "train/labels")
val_images = os.path.join(output_folder, "val/images")
val_labels = os.path.join(output_folder, "val/labels")

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Step 1: Simplify filenames (remove everything before the first "-" including the "-")
def simplify_filename(filename):
    if "-" in filename:
        return filename.split("-", 1)[1]  # Keep everything after the first "-"
    return filename  # If no "-", return the original filename

# Simplify image and label filenames
simplified_images = {}
simplified_labels = {}

for image_file in os.listdir(input_images_folder):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        new_name = simplify_filename(image_file)
        old_path = os.path.join(input_images_folder, image_file)
        new_path = os.path.join(input_images_folder, new_name)
        os.rename(old_path, new_path)
        simplified_images[new_name] = new_path  # Map simplified name to its path

for label_file in os.listdir(input_labels_folder):
    if label_file.lower().endswith('.txt'):
        new_name = simplify_filename(label_file)
        old_path = os.path.join(input_labels_folder, label_file)
        new_path = os.path.join(input_labels_folder, new_name)
        os.rename(old_path, new_path)
        simplified_labels[new_name] = new_path  # Map simplified name to its path

# Step 2: Match images and labels
matched_data = []
for image_name in simplified_images.keys():
    label_name = os.path.splitext(image_name)[0] + ".txt"  # Replace extension with .txt
    if label_name in simplified_labels:
        matched_data.append((image_name, label_name))

# Debug: Print matched data
print(f"Matched data: {matched_data}")

# Step 3: Split into train and validation sets (80% train, 20% val)
random.shuffle(matched_data)
split_idx = int(len(matched_data) * 0.8)
train_data = matched_data[:split_idx]
val_data = matched_data[split_idx:]

# Step 4: Move files to train and val folders
def move_files(data, dest_images, dest_labels):
    for image_name, label_name in data:
        # Move image
        shutil.copy(simplified_images[image_name], os.path.join(dest_images, image_name))
        # Move label
        shutil.copy(simplified_labels[label_name], os.path.join(dest_labels, label_name))

# Move train data
move_files(train_data, train_images, train_labels)

# Move val data
move_files(val_data, val_images, val_labels)

print("Dataset organized successfully into 'Output_for_yolo'!")