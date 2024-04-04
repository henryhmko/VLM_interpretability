#from pycallcc import *
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


#if __name__ != "__main__":
if True:
    import base64
    from io import BytesIO
    from PIL import Image
    
    import argparse
    import asyncio
    import json
    import time
    import threading
    import uuid
    import requests
    import torch
    import uvicorn
    import transformers
    from PIL import Image
    
    from fastapi import FastAPI, Request, BackgroundTasks
    from fastapi.responses import StreamingResponse
    from functools import partial
    from transformers import TextIteratorStreamer
    from threading import Thread
    
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


if 1:
    def get_image_embeddings(self, params):
        '''Helper function.
        Returns embeddings of a single image that is passed through the encoder+MLP'''
        
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]

        images = params.get("images", None)
        images = process_images(images, image_processor, model.config)
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

    # delete generate_stream
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=model.dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=model.dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False




        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                        "error_code": 0}).encode() + b"\0"
            return

        ## This is how you get the embeddings
        emb = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            images=images
        )[4]
        
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ''

        for new_text in streamer:
            if generated_text and not generated_text.endswith(' '):
                generated_text += ' '
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]

        # Make dict for json file
        json_dict = {}
        # json_dict["prompt"] = params["prompt"]
        # json_dict["question"] = params["question"]
        json_dict["img_path"] = params["img_path"]
        json_dict["output"] = generated_text

        # # Save hyperparameters to a separate dictionary
        # hyperparam_dict = {}
        # hyperparam_dict["temperature"] = temperature
        # hyperparam_dict["top_p"] = top_p
        # hyperparam_dict["max_context_length"] = max_context_length
        # hyperparam_dict["max_new_tokens"] = max_new_tokens
        # json_dict["model_hyperparams"] = hyperparam_dict

        # save info of dataset by reading json file
        img_path = params["img_path"]
        # get path to json file containing text hyperparams
        text_hyperparms_json_path = os.path.join(os.path.dirname(img_path), "info.json")

        with open(text_hyperparms_json_path, 'r') as file:
            data = json.load(file)

        json_dict["text_hyperparams"] = data

        print(f"Original prompt: {ori_prompt}")
        print(generated_text)
        
        return json_dict


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


import base64
from io import BytesIO
from PIL import Image
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def get_all_labels(input_fp):
    '''Helper function.
    Returns torch tensor containing all the lables in class integers when given the
    filepath to the labels.json of the dataset.'''
    with open(input_fp, 'r') as file:
        data = json.load(file)
    # Extract only the labels from all the entries
    labels_strings = [item['label'] for item in data]
    
    # Prepare a mapping from the strings to the class integer values
    label_mapping = {label: idx for idx, label in enumerate(set(labels_strings))}
    # Convert label strings("cat", "fox",...) into class numbers(e.g. 0, 1, 2,...)
    integer_labels = [label_mapping[label] for label in labels_strings]
    # Convert label matrix to tensor
    labels = torch.tensor(integer_labels)
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

#@wrap
def run(input_path):
    print("Going")

    # make sure to only select image files in the directory (no .json files)
    img_dir_path = os.path.join(input_path, 'images')
    if not os.path.exists(img_dir_path):
        print(f"No such path exists: {img_dir_path}. Check input path again.")
        return None
    all_file_paths = os.listdir(img_dir_path)
    img_paths = []
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for file in all_file_paths:
        _, extension = os.path.splitext(file)
        if extension.lower() in img_extensions:
            img_paths.append(file)
            
    img_paths = [os.path.join(img_dir_path, img_path) for img_path in img_paths if not img_path.startswith('.')] # Concatenate input_path to each file name (i.e. create full img path names)
    img_paths.sort() # And sort them

    question = "What is the one word you see on the image, if any? Dummy prompt"
    args = {'model': 'bunny-phi-2-siglip-lora', 
    'prompt': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, and detailed answers to the user's questions. \n\n USER: <image>\n%s \n\n ASSISTANT:"%question, 
    'temperature': 0, 'top_p': 0.7, 'max_new_tokens': 128, 'stop': '<|endoftext|>'}

    args['question'] = question

    # Init embeddings matrix + labels vector
    lables_vec = []
    embeddings_matrix = torch.Tensor()

    for img_path in tqdm(img_paths, desc="Generating All Image Embeddings"):
        img = Image.open(img_path).resize((400,400))
        args['images'] = [img]

        # Generate embedding for single image
        embedding = get_image_embeddings(worker, args) #embedding.shape = [1, 785, 2560]
        embedding_flatten = embedding.flatten(start_dim=1).to('cpu') #embedding_flatten.shape = [1, 2009600]
        embeddings_matrix = torch.cat((embeddings_matrix, embedding_flatten), dim=0).to('cpu')

    print("Generated All Image Embeddings.")
    
    # get all labels
    labels_fp = os.path.join(input_path, 'labels.json')
    labels_matrix = get_all_labels(labels_fp)
    print("Generated All Image Labels.")

    # Separate train-test split
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_matrix, labels_matrix, test_size=0.2
    )
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32) #bs=32 is 1.31s/it; bs=256 is 1.61s/it; 
    val_loader = DataLoader(val_dataset, batch_size=32)

             
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training starting on {device}.")
    torch.cuda.empty_cache() #clear memory


    # Set the hyperparameters
    num_imgs = embeddings_matrix.shape[0]
    embedding_size = embeddings_matrix.shape[1]

    num_classes = len(torch.unique(labels_matrix))
    learning_rate = 0.001
    # lr = 0.003
    num_epochs = 31

    model = LinearClassifier(embedding_size, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_accuracies = []
    val_accuracies = []
    epochs = []

    for epoch in tqdm(range(num_epochs), desc="Training model..."):
        model.train() # Revert back to model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, nn.functional.one_hot(batch_labels, num_classes=num_classes).float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation - only every 10 epochs
        if epoch % 10 == 0:
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

                # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {avg_train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
    print("Model Training Complete")

    # Create plot for training and validation accuracies
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("accuracy_plot.png")

    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs.")
    setup()
    input_path = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/tinyimagenet_long_words/'
    run(input_path)