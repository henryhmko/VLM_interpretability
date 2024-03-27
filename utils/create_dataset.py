from PIL import Image, ImageDraw, ImageFont
import random
import os
import math
import argparse
import json
from tqdm.auto import tqdm

'''When given the base dataset path, this script creates a 
dataset where text, position, rotation, and font size is randomized.

Outputs a labels.json file which stores all the labels and randomized
parameters specified above.'''

def get_img_paths(input_dir):
    '''Helper function.
    Returns all image paths in the given directory.'''
    images = os.listdir(input_dir)
    images = [os.path.join(input_dir, img_path) for img_path in images if not img_path.startswith('.')]
    # Sorting images is needed to keep a consistent order
    images = sorted(images)
    return images
    

def get_font_size(img,
                  font_path,
                  rand_word,
                  min_font_size,
                  max_font_size
                  ):
    '''Helper function.
    Returns random font size for the selected word.'''

    rand_font_size = random.randint(min_font_size, max_font_size)
    font = ImageFont.truetype(font_path, rand_font_size)
    draw = ImageDraw.Draw(img)
    text_length = draw.textlength(rand_word, font=font)

    width, _ = img.size
    
    loop_counter = 0
    # if the font size is invalid, try new font sizes until the text fits within the image
    while (width/2.1 - math.ceil(text_length) < 0):
        if loop_counter > 10000:
            raise Exception("Invalid min/max font size bounds. Set wider range for min/max font sizes")
        rand_font_size = random.randint(min_font_size, max_font_size)
        font = ImageFont.truetype(font_path, rand_font_size)
        text_length = draw.textlength(rand_word, font=font)
        loop_counter += 1
            
    return rand_font_size

def get_text_position(img,
                      rand_word,
                      font
                      ):
    '''Helper function.
    Returns random x, y position to place text.'''

    width, _ = img.size
    draw = ImageDraw.Draw(img)
    text_length = draw.textlength(rand_word, font=font)

    # initialize (x,y) to arbitrary large int
    x = width*10
    y = width*10
    
    # sample (x,y) from circle of radius=(width - 2*text_length)/2, so any rotations applied to the text would still fit within the image
    while x**2 + y**2 > ((width - 2*text_length)/2)**2:
        x = random.randint(0, width - int(2*text_length))
        y = random.randint(0, width - int(2*text_length))
    
    # add text_length to center (x,y) to the center of the image
    # math.ceil is used since text_length should be less than calculated maximum
    x += int(text_length)
    y += int(text_length)
    return x, y

def place_text_on_img(img_path,
                      rand_word,
                      rand_position,
                      rand_rotation,
                      rand_font_size,
                      img_size = 400,
                      min_font_size = 30,
                      max_font_size = 100
                      ):
    '''Writes passed in text on the given image. Saves all parameters
    such as rotation degree, position, font size.
    
    Returns:
        img_with_text: PIL's Image object with text written on it.
        labels: Python Dictionary object containing image_id, text label, rotation, etc.'''
    # Check if image exists
    assert os.path.exists(img_path)

    # "RGBA" format used since the alpha layer will be useful for overlaying text on top
    with Image.open(img_path).convert("RGBA") as img:
        # Resize to img_size. Default is (400, 400) to comply with CLIP encoder's restriction size
        img = img.resize((img_size, img_size), resample = Image.Resampling.BICUBIC)
                
        width, height = img.size
        font_path = "font_file.ttf"
        # Check if font file exists
        if not os.path.exists(font_path):
            print("ERROR: Font file missing.")
            print(f"Make sure it is placed in {os.getcwd()}.")
            print("Terminating program. Please run again after fixing errors.")
            return
        
        # Choose font size
        if rand_font_size:
            font_size = get_font_size(img, font_path, rand_word, min_font_size, max_font_size)
        else:
            font_size = 40 # Fix font_size to 40 if not rand
        font = ImageFont.truetype(font_path, font_size) 
        
        # Choose position
        if rand_position:
            x, y = get_text_position(img, rand_word, font)
        else:
            x, y = 100, 100 # Fix position to (100,100) if not rand
        
        # Choose rotation
        if rand_rotation:
            angle = random.randint(0, 360)
        else:
            angle = 0 # Fix angle to 0 if not rand
        
        # Choose color; color is set to white by default.
        r, g, b = 255, 255, 255

        # Overlay new mask to place text on top of
        # Create a new blank canvas
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.text((x,y), rand_word, fill=(r,g,b), font=font, font_size=font_size)

        # Apply the rotation to the text
        overlay = overlay.rotate(angle)
        
        # Paste the rotated overaly onto the original image
        img = Image.alpha_composite(img, overlay)

        # Create labels, a dictionary containing all parameters
        labels = {}
        labels["image_id"] = os.path.basename(img_path) # ex:'img1.jpg'
        labels["label"] = rand_word # ex:"cat"
        labels["position"] = (x, y) # ex: (100, 100)
        labels["rotation"] = angle # ex: 90
        labels["color"] = (r,g,b) # ex: (255, 255, 255)
        labels["font_size"] = font_size # ex: 40

        return img, labels
        
def run(input_dir, 
        output_dir,
        words,
        rand_position = True, # bool for randomizing position
        rand_rotation = True, # bool for randomizing rotation
        rand_font_size = True # bool for randomizing font size
        ):
    # Check input_dir is a valid directory
    cur_dir = os.path.dirname(os.getcwd())
    input_dir = os.path.join(cur_dir, input_dir)
    assert os.path.exists(input_dir), f"Input Path is invalid: {input_dir}"
    assert output_dir[0] != "/", f"Output Path is invalid: Does your path start with '/'?"
    # Check output_dir exists; create output_dir if it does not exist
    new_output_dir = os.path.join(cur_dir, output_dir)
    output_imgdir = os.path.join(new_output_dir, 'images')
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        # Also create dir for the new images
        os.makedirs(output_imgdir)
    # Create array to store all labels that will go into labels.json
    json_arr = []

    # Get all image paths
    img_paths = get_img_paths(input_dir)

    print("Creating Dataset...")
    for img_path in tqdm(img_paths, desc="Processing Images"):
        # Choose random word to place on text
        rand_word = random.choice(words)
        img_with_text, labels_dict = place_text_on_img(img_path, rand_word, rand_position, rand_rotation, rand_font_size)
        
        # Save img_with_text to output_dir and append labels to array storing all labels
        new_img_name = f"with_text_{os.path.basename(img_path)}"
        output_img_path = os.path.join(output_imgdir, new_img_name)
        img_with_text.save(output_img_path)
        json_arr.append(labels_dict)
        
    # Dump all labels array into labels.json
    json_file_name = 'labels.json'
    json_file_path = os.path.join(new_output_dir, json_file_name)
    with open(json_file_path, 'w') as file:
        json.dump(json_arr, file, indent=4)
    print("Creation Complete!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Creates dataset with randomized text positions, color, and rotations"
    )
    
    parser.add_argument(
        "-i",
        "--input_path",
        type = str,
        default = "bunny/serve/examples",
        help = "Input directory where all images are stored"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type = str,
        default="bunny/serve/text_outputs",
        help = "output directory where all output images are stored"
    )
    parser.add_argument(
        "--rand_position",
        type = bool,
        default=True,
        help = "boolean for randomizing position. Set to False if fixed."
    )
    parser.add_argument(
        "--rand_rotation",
        type = bool,
        default=True,
        help = "boolean for randomizing rotation. Set to False if fixed."
    )
    parser.add_argument(
        "--rand_font_size",
        type = bool,
        default=True,
        help = "boolean for randomizing font size. Set to False if fixed."
    )

    args = parser.parse_args()

    words = ["cat", "dog", "fox", "rat", "pet"]
    
    run(args.input_path,
        args.output_path,
        words,
        args.rand_position,
        args.rand_rotation,
        args.rand_font_size)
