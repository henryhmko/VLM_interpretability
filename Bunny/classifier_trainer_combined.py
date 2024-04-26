import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import warnings
import json
import random
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import transformers
from bunny.constants import WORKER_HEART_BEAT_INTERVAL
from bunny.util.utils import (build_logger, server_error_msg, pretty_print_semaphore)
from bunny.model.builder import load_pretrained_model
from bunny.util.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name, model_type,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            self.model_name = get_model_name_from_path(model_path)
        else:
            self.model_name = model_name

        self.device = device
        print(f"Loading the model {self.model_name} on worker {worker_id} ...")
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, model_type, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = True


def get_image_embeddings(self, params):
    '''Helper function.
    Returns embeddings of a single image that is passed through the encoder+MLP'''
    
    tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

    prompt = params["prompt"]

    images = params.get("images", None)
    # images = torch.cat([images]*40, 0)
    images = process_images(images, image_processor, model.config)
    # images = torch.cat([images]*40, 0)
    images = images.to(self.model.device, dtype=model.dtype)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        self.device)

    emb = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        None,
        None,
        None,
        images=images
    )[4]
    return emb

#@wrap
def setup():
    global worker
    # Directly using the provided command line argument values
    import uuid

    worker_id = str(uuid.uuid4())[:6]
    worker = ModelWorker("http://localhost:10000",  # controller_address
                         "http://localhost:40000",  # worker_address
                         worker_id,                 # worker_id (needs to be defined)
                         True,                      # no_register
                         "../bunny-phi-2-siglip-lora/",  # model_path
                         "../phi-2/",               # model_base
                         None,                      # model_name (passed via command line)
                         "phi-2",                   # model_type
                         False,                     # load_8bit (assuming default as False)
                         False,                     # load_4bit (assuming default as False)
                         "cuda")                    # device


def get_all_labels(input_dict):
    '''Helper function.
    Returns torch tensor containing all the lables in class integers when given the
    json arr containing all labels of the dataset.'''
    # with open(input_fp, 'r') as file:
    #     data = json.load(file)
    # Extract only the labels from all the entries
    labels_strings = [item['label'] for item in input_dict]
    
    # Prepare a mapping from the strings to the class integer values
    label_mapping = {label: idx for idx, label in enumerate(set(labels_strings))}
    # Convert label strings("cat", "fox",...) into class numbers(e.g. 0, 1, 2,...)
    integer_labels = [label_mapping[label] for label in labels_strings]
    # Convert label matrix to tensor
    labels = torch.tensor(integer_labels)
    # return labels
    return labels

# Build model
class LinearClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # Just returning fc layer without sigmoid since we use nn.BCEWithLogits
        x = x.to(self.fc.weight.dtype)
        return self.fc(x)

def get_img_paths(input_dir):
    '''Helper function.
    Returns all image paths in the given directory.'''
    images = os.listdir(input_dir)
    images = [os.path.join(input_dir, img_path) for img_path in images if not img_path.startswith('.')]
    # Sorting images is needed to keep a consistent order
    images = sorted(images)
    return images

def get_font_size(img, font_path, rand_word, min_font_size, max_font_size):
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

def get_text_position(img, rand_word, font):
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

def place_text_on_img(img_path, rand_word, rand_position, rand_rotation, rand_font_size, rand_color, font_path, img_size=400, min_font_size=30, max_font_size=100):
    '''Writes passed in text on the given image. Saves all parameters
    such as rotation degree, position, font size.
    
    Returns:
        img_with_text: PIL's Image object with text written on it.
        labels: Python Dictionary object containing image_id, text label, rotation, etc.'''
    # Check if image exists
    # assert os.path.exists(img_path)
    # print('THIS IS THE IMG PATH WORKING ON:')
    # print(img_path)

    # "RGBA" format used since the alpha layer will be useful for overlaying text on top
    with Image.open(img_path).convert("RGBA") as img:
        # Resize to img_size. Default is (400, 400) to comply with CLIP encoder's restriction size
        img = img.resize((img_size, img_size), resample = Image.Resampling.BICUBIC)
                
        width, height = img.size
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
        if rand_color:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
        else:
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

def run(input_dir, output_dir, words, font_path, rand_position=True, rand_rotation=True, rand_font_size=True, rand_color=True):
    # Check input_dir is a valid directory
    cur_dir = os.path.dirname(os.getcwd())
    input_dir = os.path.join(cur_dir, input_dir)
    assert os.path.exists(input_dir), f"Input Path is invalid: {input_dir}"
    # # Check output_dir exists; create output_dir if it does not exist
    new_output_dir = os.path.join(cur_dir, output_dir)
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
    
    json_arr = []
    embeddings_matrix = torch.Tensor()
    img_paths = get_img_paths(input_dir)
    img_paths = img_paths[:10] #for testing purposes
    print(img_paths)

    print("Creating Dataset and Generating Embeddings...")
    for img_path in tqdm(img_paths, desc="Processing Images"):
        rand_word = random.choice(words)
        img_with_text, labels_dict = place_text_on_img(img_path, rand_word, rand_position, rand_rotation, rand_font_size, rand_color, font_path)
        
        # Generate embedding for the modified image
        args = {'model': 'bunny-phi-2-siglip-lora',
                'prompt': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, and detailed answers to the user's questions. \n\n USER: <image>\nWhat is the one word you see on the image, if any? Dummy prompt \n\n ASSISTANT:",
                'temperature': 0, 'top_p': 0.7, 'max_new_tokens': 128, 'stop': '<|endoftext|>'}
        args['images'] = [img_with_text]
        embedding = get_image_embeddings(worker, args)
        
        # Currently offloading image embeddings to CPU due to VRAM constraints
        # TODO: Do this on the GPU if VRAM is not an issue
        embedding_flatten = embedding.flatten(start_dim=1).to('cpu')
        embeddings_matrix = torch.cat((embeddings_matrix, embedding_flatten), dim=0).to('cpu')

        json_arr.append(labels_dict)

    print("Generated All Image Embeddings.")
    labels_matrix = get_all_labels(json_arr)
    print("Generated All Image Labels.")

    #### LINEAR CLASSIFIER HYPERPARAMETERS ####
    BATCH_SIZE = 64
    LR = 1e-3
    NUM_EPOCHS = 500
    ###########################################
    
    # Train-test split
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_matrix, labels_matrix, test_size=0.2
    )
    
    # Create Dataset
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    # Create Dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        warnings.warn("Training starting on the CPU. Assign available GPUs if this was not intended.", RuntimeWarning)

    # Get num_imgs, embedding_size, total number of classes
    num_imgs = embeddings_matrix.shape[0]
    embedding_size = embeddings_matrix.shape[1]
    num_classes = len(torch.unique(labels_matrix))

    # Create model
    model = LinearClassifier(embedding_size, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogits
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    train_accuracies = []
    val_accuracies = []
    epochs = []

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training model..."):
        model.train() # Revert back to model.train() after validation
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, nn.functional.one_hot(batch_labels, num_classes=num_classes).float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation - Record every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            epochs.append(epoch)
            with torch.no_grad():
                # Compute train accuracy first
                total_train_accuracy = []
                total_val_accuracy = []
                for batch_data, batch_labels in train_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    train_outputs = model(batch_data)
                    train_predicted = torch.argmax(torch.sigmoid(train_outputs), dim=1)
                    train_accuracy = (train_predicted == batch_labels.to(device)).sum().item() / len(batch_labels)
                    total_train_accuracy.append(train_accuracy)
                avg_train_accuracy = np.mean(total_train_accuracy)
                train_accuracies.append(avg_train_accuracy)
                
                # Compute validation accuracy
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device)
                    batch_labels = batch_labels.to(device)

                    val_outputs = model(batch_data)
                    val_predicted = torch.argmax(torch.sigmoid(val_outputs), dim=1)
                    val_accuracy = (val_predicted == batch_labels.to(device)).sum().item() / len(batch_labels)
                    total_val_accuracy.append(val_accuracy)
                avg_val_accuracy = np.mean(total_val_accuracy)
                val_accuracies.append(avg_val_accuracy)
    print("Model Training Complete")

    save_path = os.path.join(new_output_dir, 'classifier_weights.pth')
    torch.save(model.state_dict(), save_path)
    print("Model Weights Saved")

    # Create plot for training and validation accuracies
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt_fig_path = os.path.join(new_output_dir, f"accuracy_plot_{NUM_EPOCHS}epochs_bs{BATCH_SIZE}.png")
    plt.savefig(plt_fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type = str,
        default = "Bunny/bunny/serve/examples",
        help = "Input directory where all images are stored"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type = str,
        default="Bunny/bunny/serve/model_outputs",
        help = "Output directory where the model weights file will be stored"
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
    parser.add_argument(
        "--rand_color",
        type = bool,
        default=True,
        help = "boolean for randomizing color. Set to False if fixed."
    )
    parser.add_argument(
        "--font_path", 
        type = str,
        default="font_file.ttf", 
        help="Path to the font file"
    )
    
    args = parser.parse_args()

    words = ["creative", "notebook", "strategy", "discover", "activity"]  # hard 5 words

    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs.")
    setup()

    run(args.input_path, args.output_path, words, args.font_path, args.rand_position, args.rand_rotation, args.rand_font_size, args.rand_color)