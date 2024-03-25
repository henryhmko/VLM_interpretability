from PIL import Image, ImageDraw, ImageFont
import random
import os
import math
import argparse

'''When given the base dataset path, this script creates a 
dataset where text, position, rotation, and font size is randomized.

Outputs a labels.json file which stores all the labels and randomized
parameters specified above.'''

# get input directory
# place text on top of them
# save that image to a new directory and store all labels in labels.json

# write functions for...
    # writing text on single image (single text, img) -> (img with text)
    # save parameters + labels in one directory
    # pick one word at random out of the input words

def get_img_paths(input_dir):
    '''Helper function.
    Returns all image paths in the given directory.'''
    pass

def get_font_size(font_path,
                  rand_word,
                  min_font_size,
                  max_font_size
                  ):
    '''Helper function.
    Returns random font size for the selected word.'''
    pass

def get_text_position(rand_word,
                      font
                      ):
    '''Helper function.
    Returns random x, y position to place text.'''
    pass

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
        
        draw = ImageDraw.Draw(img)
        
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
            font_size = get_font_size(font_path, rand_word, min_font_size, max_font_size)
        else:
            font_size = 40 # Fix font_size to 40 if not rand
        font = ImageFont.truetype(font_path, font_size) 
        
        # Choose position
        if rand_position:
            x, y = get_text_position(rand_word, font)
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
        labels["image_id"] = 

        return img, labels
        
        
        
        

def run(input_dir, 
        words, 
        output_dir, 
        rand_position = True, # bool for randomizing position
        rand_rotation = True, # bool for randomizing rotation
        rand_font_size = True # bool for randomizing font size
        ):
    # Check input_dir is a valid directory
    assert os.path.exists(input_dir)

    # Get all image paths
    img_paths = get_img_paths(input_dir)
    for img_path in img_paths:
        # Choose random word to place on text
        rand_word = random.choice(words)
        img_with_text, labels = place_text_on_img(img_path, rand_word, rand_position, rand_rotation, rand_font_size)
        # Save img_with_text to output_dir and append labels to labels.json

    

if __name__ == "__main__":
    words = [#FILL THE BLANK]
    input_directory = #INPUT GT IMAGE DIR PATH
    output_dir = #output dir #fill out
    run(input_dir, words, output_dir)
