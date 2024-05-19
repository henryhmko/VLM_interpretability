import torch
import torch.nn as nn
import argparse

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

def get_weight_matrix(model):
    return model.fc.weight.detach()

def run_experiment_1(model, new_data, labels, device):
    # Get model's weights
    weights_matrix = get_weight_matrix(model) # [5, hidden_dim] since linear model is W @ x.T + b

    # Work with just 1 embedding for now # NOTE: Generalize this to many examples
    embed = new_data[0:1, :] # [1, hidden_dim]
    embed_gt_label = labels[0].long() # Convert fp to int since label should be a class number
    
    embed = new_data[1:2, :]
    embed_gt_label = labels[1].long()

    # Sanity checking dimensions
    print(f"embed shape: {embed.shape}")
    print(f"weights_matrix shape: {weights_matrix.shape}")
    
    logits = embed @ weights_matrix.T
    print(f"logits shape: {logits.shape}")
    print(f"logits: {logits}")

    print(f"ground truth label: {embed_gt_label}")
    
    w_row_0 = weights_matrix[0, :] # [1, hidden_dim]
    w_row_1 = weights_matrix[1, :] # [1, hidden_dim]

    embed2 = embed - w_row_0 + w_row_1

    new_logits = embed2 @ weights_matrix.T
    new_label = torch.argmax(new_logits)

    print(f"new_logits: {new_logits}")
    print(f"new label: {new_label}")
        
    return

def setup(weights_path, data_path, labels_path, embedding_size, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model weights
    model = load_model(weights_path, embedding_size, num_classes, device)

    # Load and preprocess new data
    new_data = torch.load(data_path)  
    labels = torch.load(labels_path)

    run_experiment_1(model, new_data, labels, device)

    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str,
                        help="Path to the saved model weights")
    parser.add_argument("--data_path", type=str,
                        help="Path to the new data file")
    parser.add_argument("--labels_path", type=str,
                        help="Path to the labels file")
    parser.add_argument("--embedding_size", type=int, default=839808,
                        help="Size of the embedding")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of classes")
    
    args = parser.parse_args()

    setup(args.weights_path, args.data_path, args.labels_path, args.embedding_size, args.num_classes)
    