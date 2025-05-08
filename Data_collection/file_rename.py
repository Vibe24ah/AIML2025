import os

# Define the directory containing the images
directory = "/Users/victorberstrand/Desktop/UNI/CBS/AI and machine learning/Exam/Fred_pictures"

# Get a list of all files in the directory
files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg'))]

# Sort the files to ensure consistent renaming order
files.sort()

# Step 1: Rename all files to temporary names to avoid conflicts
temp_files = []
for index, file in enumerate(files, start=1):
    # Get the file extension
    file_extension = os.path.splitext(file)[1]
    
    # Create a temporary filename
    temp_name = f"temp_{index}{file_extension}"
    
    # Get the full paths
    old_path = os.path.join(directory, file)
    temp_path = os.path.join(directory, temp_name)
    
    # Rename the file to the temporary name
    os.rename(old_path, temp_path)
    temp_files.append(temp_name)

# Step 2: Rename temporary files to final sequential names
for index, temp_file in enumerate(temp_files, start=1):
    # Get the file extension
    file_extension = os.path.splitext(temp_file)[1]
    
    # Create the final filename
    final_name = f"{index}{file_extension}"
    
    # Get the full paths
    temp_path = os.path.join(directory, temp_file)
    final_path = os.path.join(directory, final_name)
    
    # Rename the file to the final name
    os.rename(temp_path, final_path)

print("Renaming complete!") 