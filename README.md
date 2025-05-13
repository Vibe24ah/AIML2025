# AIML2025

Explanations of the python files:
**Production_image_generator: **
  Based on manual inspection, we are using the coordinates from the initial mimic production-image, to replace with images from a given folder, and place them in a new. This is also generating YOLOv8 coordinate files, which were used in iterations 1-2. 

**File rename:**
  Simple script to rename files ranging 1-x / in order to mimic a unique ID.

**DS_Variation:**
  Script that "shifts" production images, i.e. rotating them slightly.

**YOLO_Organizer:**
  Script that organizes given files and label-files in the correct order for the YOLO-model to use the data.

**Trainer_V4:**
  The script that trains/fine-tunes the YOLO Model.

**V4_Flow:**
  Contains the final flow, where a nested image is cropped out, and compared to the corresponding comparison image. 

**FlowV4_SIMTEST:**
  Script we made, where we are running the whole pipeline through a series of test images (unseen data) and classifying based on the set parameters. Here we have one dataset with "True" image pairs, and one with "False" image paris - all labeled. The results are stored in a CSV-file, which we can use to evaluate the performance of the pipeline as a whole.
