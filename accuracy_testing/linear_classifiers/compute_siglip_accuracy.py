import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image
from transformers import AutoProcessor, AutoModel
import os
from tqdm.auto import tqdm
import json

'''See at how text is recognized by looking at embeddings of an image with text on them. 
1. Feed images through SigLip encoder
2. Feed these embeddings to a linear classifier and see how it performs
'''

def get_image_embeddings(input_path, model, processor):
    '''Helper function.
    Returns embeddings of a single image when given the input path.'''
    image = Image.open(input_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings

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

def run(input_dir):
    '''Computes accuracy of linear classifier on siglip embeddings.'''
    # Initialize siglip model and processor
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    print("Model Loading Complete.")
    
    # Init embeddings matrix + labels vector
    embeddings_lst = []
    labels_vec = []
    
    # Get all image paths in the given directory
    img_dir = os.path.join(input_dir, 'images')
    img_paths = os.listdir(img_dir)
    img_paths = sorted(img_paths)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths]
    for img_path in tqdm(img_paths, desc="Generating All Image Embeddings"):
        # Generate embedding
        embedding = get_image_embeddings(img_path, model, processor)
        # Append each embedding to the total embeddings matrix
        embeddings_lst.append(embedding)
    
    # Convert embeddings list into tensor
    embeddings_matrix = torch.cat(embeddings_lst, dim=0)
    print("Generated All Image Embeddings.")

    # Get all labels
    labels_fp = os.path.join(input_dir, 'labels.json')
    labels_matrix = get_all_labels(labels_fp)
    print("Generated All Image Labels.")

    # Build and train linear classifier
    # Err..would be good to separate these into separate files later..
    
    # Separate train-test split
    train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(
        embeddings_matrix, labels_matrix, test_size=0.2
    )

    # Build model
    class LinearClassifier(nn.Module):
        def __init__(self, embedding_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            return torch.sigmoid(self.fc(x))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the hyperparameters
    embedding_size = embeddings_matrix.shape[1]
    num_classes = len(torch.unique(labels_matrix))
    learning_rate = 0.003
    num_epochs = 50

    model = LinearClassifier(embedding_size, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    # for epoch in tqdm(range(num_epochs), desc="Model Training..."):
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_embeddings.to(device))
        loss = criterion(outputs, nn.functional.one_hot(train_labels.to(device), num_classes=num_classes).float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print Loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Model Training Complete.")
    print("Starting Evaluation.")

    # Evaluate the model on the validation set
    with torch.no_grad():
        outputs = model(val_embeddings.to(device))
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == val_labels.to(device)).sum().item() / len(val_labels)
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    

if __name__ == "__main__":
    input_path = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/doggos_randomized_sorted/'
    run(input_path)

# image_file = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/doggos_randomized/images/with_text_ILSVRC2012_val_00001968.png'
# image_file = '/home/ko.hyeonmok/local_testing/VLM_interpretability/data/doggos_randomized/images/with_text_n02102040_334.png'


# embeddings = get_image_embeddings(image_file)

# print(embeddings)
# print(embeddings.shape)
# print(embeddings.size())