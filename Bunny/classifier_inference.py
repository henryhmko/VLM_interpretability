import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np

# Define the same model architecture as in the training script
class LinearClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        return self.fc(x)

def load_model(weights_path, embedding_size, num_classes, device):
    model = LinearClassifier(embedding_size, num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def run_inference(model, data_loader, device, loss):
    with torch.no_grad():
        total_val_accuracy = []
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            val_outputs = model(batch_data)
            if loss.lower() == "bce": #model is trained with BCELoss
                val_predicted = torch.argmax(torch.sigmoid(val_outputs), dim=1)
            elif loss.lower() == "ce": #model is trained with CE Loss
                val_predicted = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predicted == batch_labels).sum().item() / len(batch_labels)
            total_val_accuracy.append(val_accuracy)

        avg_val_accuracy = np.mean(total_val_accuracy)
        print("Validation accuracy", avg_val_accuracy)
        print(total_val_accuracy)
    return

def setup(weights_path, data_path, labels_path, embedding_size, num_classes, batch_size, loss):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model weights
    model = load_model(weights_path, embedding_size, num_classes, device)

    # Load and preprocess new data
    new_data = torch.load(data_path)  
    labels = torch.load(labels_path) 
    dataset = TensorDataset(new_data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    run_inference(model, data_loader, device, loss)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=".../bunny/serve/model_outputs/model_xe.pt",
                        help="Path to the saved model weights")
    parser.add_argument("--data_path", type=str, default='.../bunny/serve/model_outputs/embedding_matrix_notext.pt',
                        help="Path to the new data file")
    parser.add_argument("--labels_path", type=str, default='.../bunny/serve/model_outputs/labels_matrix_notext.pt',
                        help="Path to the labels file")
    parser.add_argument("--embedding_size", type=int, default=839808,
                        help="Size of the embedding")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--loss", type=str, default="ce",
                        help="Loss function used to train the linear classifier")
    
    args = parser.parse_args()
    
    setup(args.weights_path, args.data_path, args.labels_path, args.embedding_size, args.num_classes, args.batch_size, args.loss)