import os
import random
import shutil
from tqdm.auto import tqdm

'''Curate new datasets by mixing images from Honeydew's shared data directory.
Selects arbitrary classes out of tiny-imagenet.
'''
def curate_dataset(source_folder, destination_folder, num_classes, num_images_per_class):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get the list of classes from the source dir
    all_classes = [subdir for subdir in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, subdir))]
    
    # Select arbitrary classes up to num_classes
    classes = random.sample(all_classes, num_classes)

    for class_name in tqdm(classes, desc="Processing..."):
        class_folder = os.path.join(source_folder, class_name, 'images')
        image_files = [file for file in os.listdir(class_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Select random images up to num_images_per_class in the class_folder
        selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))

        # Copy the selected images to the destination folder
        for image_file in selected_images:
            source_path = os.path.join(class_folder, image_file)
            # print(f"class_folder: {class_folder}")
            # print(f"image_file: {image_file}")
            # print(f"source_path: {source_path}")
            destination_path = os.path.join(destination_folder, f"{image_file}")
            # print(f"destination_path: {destination_path}")
            shutil.copy(source_path, destination_path)
        
        print(f"Copied {len(selected_images)} images from class '{class_name}' to the destination folder.")
    
# Example usage
source_folder = '/data/tiny-imagenet-200/train' #Note: only works on honeydew
destination_path = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/tinyimagenet_notext/'
num_classes = 5
num_images_per_class = 140
# Total: 700 images

curate_dataset(source_folder, destination_path, num_classes, num_images_per_class)