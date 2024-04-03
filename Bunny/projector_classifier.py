#from pycallcc import *
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
        # images = process_images(images, image_processor, model.config)
        images = process_images(images, image_processor, self.model.modules.config)
        # Holy this would be so slow. There is 0 batching of images being done -> #TODO: err fix later when i get the pipeline working first
        images = images.to(dtype=model.dtype)
        # images = images.to(self.model.device, dtype=model.dtype)

        # self.model = Model(input_size, output_size)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
            # model.to(device)

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
            # print(f"new_text: {new_text}")

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
    labels = torch.tensor(integer_labels).to(torch.float16)
    print(f"label dtype: {labels.dtype}")
    return labels[:350]
    # return labels[:350].to(torch.float16)


#@wrap
def run(input_path):
    print("Going")

    # make sure to only select image files in the directory (no .json files)
    img_dir_path = os.path.join(input_path, 'images')
    print(f"img_dir_path is: {img_dir_path}")
    if not os.path.exists(img_dir_path):
        print(f"No such path exists: {img_dir_path}. Check input path again.")
        return None
    all_file_paths = os.listdir(img_dir_path)
    print(f"item from all_file_paths: {all_file_paths[0]}")
    img_paths = []
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for file in all_file_paths:
        _, extension = os.path.splitext(file)
        if extension.lower() in img_extensions:
            img_paths.append(file)
            
    img_paths = [os.path.join(img_dir_path, img_path) for img_path in img_paths if not img_path.startswith('.')] # Concatenate input_path to each file name (i.e. create full img path names)
    img_paths.sort() # And sort them
    img_paths = img_paths[:350] # Half them
    print(f"Number of images: {len(img_paths)}") # Print number of images as sanity check

    question = "What is the one word you see on the image, if any? Dummy prompt"
    args = {'model': 'bunny-phi-2-siglip-lora', 
    'prompt': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, and detailed answers to the user's questions. \n\n USER: <image>\n%s \n\n ASSISTANT:"%question, 
    'temperature': 0, 'top_p': 0.7, 'max_new_tokens': 128, 'stop': '<|endoftext|>'}

    args['question'] = question
    # args['images'] = [img] #huh where does img come from? Keep for now

    # Init embeddings matrix + labels vector
    embeddings_lst = []
    lables_vec = []

    # img_paths = img_paths[:350]
    # print(f"new num imgs is; {len(img_paths)}")

    for img_path in tqdm(img_paths, desc="Generating All Image Embeddings"):
        img = Image.open(img_path).resize((400,400))
        args['images'] = [img]

        # Generate embedding for single image
        embedding = get_image_embeddings(worker, args)
        # Append each embedding to the total embeddings matrix
        embeddings_lst.append(embedding)

    embeddings_lst = embeddings_lst
    # labels[:350].to(torch.float16)d
    # embeddings_lst = embeddings_lst.to(torch.float16)
    #convert embeddigns lst into a tensor
    embeddings_matrix = torch.cat(embeddings_lst, dim=0)
    print(f"embedding dtype: {embeddings_matrix.dtype}")
    print("Generated All Image Embeddings.")
    
    # get all labels
    labels_fp = os.path.join(input_path, 'labels.json')
    labels_matrix = get_all_labels(labels_fp)
    print("Generated All Image Labels.")

    # Separate train-test split
    # labels_dtype = labels_matrix.dtype
    # embeddings_matrix = embeddings_matrix.to(labels_dtype)
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_matrix, labels_matrix, test_size=0.2
    )

    # Build model
    class LinearClassifier(nn.Module):
        def __init__(self, embedding_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            # Just returning fc layer without sigmoid since we use nn.BCEWithLogits
            print(f"input dtype is: {x.dtype}")
            print(f"weight dtype is: {self.fc.weight.dtype}")
            # x = x.to(self.fc.weight.dtype)
            self.fc.weight = self.fc.weight.to(x.dtype)
            self.fc.bias = self.fc.bias.to(x.dtype)
            return self.fc(x)
            


    # # NEW SCHEME
    # # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Set the hyperparameters
    # embedding_size = embeddings_matrix.shape[1]
    # num_classes = len(torch.unique(labels_matrix))
    # learning_rate = 0.003
    # num_epochs = 1500

    # # Split the embeddings and labels into chunks
    # chunk_size = 350  # 2070ti could hold ~376.72 chunks
    # train_embeddings_chunks = torch.split(train_embeddings, chunk_size)
    # val_embeddings_chunks = torch.split(val_embeddings, chunk_size)
    # train_labels_chunks = torch.split(train_labels, chunk_size)
    # val_labels_chunks = torch.split(val_labels, chunk_size)

    # # Create a separate model for each chunk
    # models = []
    # for _ in range(len(train_embeddings_chunks)):
    #     model = LinearClassifier(embedding_size, num_classes).to(device)
    #     models.append(model)

    # # Define the loss function and optimizer for each model
    # criterion = nn.BCEWithLogitsLoss()
    # optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    # # Training loop
    # train_accuracies = []
    # val_accuracies = []
    # epochs = []

    # for epoch in range(num_epochs):
    #     for model in models:
    #         model.train()  # Set the model to training mode

    #     for i in range(len(train_embeddings_chunks)):
    #         train_embeddings_chunk = train_embeddings_chunks[i].to(device)
    #         train_labels_chunk = train_labels_chunks[i].to(device)
    #         model = models[i]
    #         optimizer = optimizers[i]

    #         # Forward pass
    #         outputs = model(train_embeddings_chunk)
    #         loss = criterion(outputs, nn.functional.one_hot(train_labels_chunk, num_classes=num_classes).float())

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     if epoch % 50 == 0:  # Record validation accuracy for every 50 epochs
    #         epochs.append(epoch)
    #         train_accuracy = 0.0
    #         val_accuracy = 0.0

    #         with torch.no_grad():
    #             for i in range(len(train_embeddings_chunks)):
    #                 train_embeddings_chunk = train_embeddings_chunks[i].to(device)
    #                 train_labels_chunk = train_labels_chunks[i].to(device)
    #                 model = models[i]
    #                 model.eval()  # Set the model to evaluation mode

    #                 train_outputs = model(train_embeddings_chunk)
    #                 train_predicted = torch.argmax(torch.sigmoid(train_outputs), dim=1)
    #                 train_accuracy += (train_predicted == train_labels_chunk).sum().item()

    #             train_accuracy /= len(train_labels)
    #             train_accuracies.append(train_accuracy)

    #             for i in range(len(val_embeddings_chunks)):
    #                 val_embeddings_chunk = val_embeddings_chunks[i].to(device)
    #                 val_labels_chunk = val_labels_chunks[i].to(device)
    #                 model = models[i]
    #                 model.eval()  # Set the model to evaluation mode

    #                 outputs = model(val_embeddings_chunk)
    #                 predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
    #                 val_accuracy += (predicted == val_labels_chunk).sum().item()

    #             val_accuracy /= len(val_labels)
    #             val_accuracies.append(val_accuracy)

    #             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # # END OF NEW SCHEME
             
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() #clear memory


    # Set the hyperparameters
    embedding_size = embeddings_matrix.shape[1]
    num_classes = len(torch.unique(labels_matrix))
    learning_rate = 0.003
    num_epochs = 1500

    model = LinearClassifier(embedding_size, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_accuracies = []
    val_accuracies = []
    epochs = []
    
    for epoch in range(num_epochs):
        # Reset model back to training mode
        model.train() 

        # Forward pass
        outputs = model(train_embeddings.to(device))
        loss = criterion(outputs, nn.functional.one_hot(train_labels.to(device), num_classes=num_classes).float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate training accuracy
        model.eval()  # Set the model to evaluation mode

        if epoch % 50 == 0: # Record validation accuracy for every 50 epochs
            epochs.append(epoch)
            with torch.no_grad():
                train_outputs = model(train_embeddings.to(device))
                train_predicted = torch.argmax(torch.sigmoid(train_outputs), dim=1)
                train_accuracy = (train_predicted == train_labels.to(device)).sum().item() / len(train_labels)
                train_accuracies.append(train_accuracy)

                # Evaluate the model on the validation set
                outputs = model(val_embeddings.to(device))
                predicted = torch.argmax(torch.sigmoid(outputs), dim=1)
                val_accuracy = (predicted == val_labels.to(device)).sum().item() / len(val_labels)
                val_accuracies.append(val_accuracy)

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print("Model Training Complete.")

    # Create plot for training and validation accuracies
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("accuracy_plot.png")
    
# if __name__ == "__main__":
#     input_path = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/tinyimagenet_long_words/'
#     print(input_path)
#     run(input_path)

    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    setup()
    input_path = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/tinyimagenet_long_words/'
    run(input_path)