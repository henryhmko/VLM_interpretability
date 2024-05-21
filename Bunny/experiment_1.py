import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

def run_experiment_1(model, new_data, labels):
    # Get model's weights
    weights_matrix = get_weight_matrix(model) # [5, hidden_dim] since linear model is W @ x.T + b

    # Get 500 examples for now (can be changed later)
    n = 500
    embed = new_data[:n, :] #[n, hidden_dim]
    embed_gt_labels = labels[:n].long() #[n]
    
    # Original
    original_logits = embed @ weights_matrix.T #[n, 5]
    original_preds = torch.argmax(original_logits, dim=1)
    
    # Switch with arbitrary row
    embed2 = embed.clone()
    for i in range(n):
        gt_label = embed_gt_labels[i].item()

        # Get top 2 indices
        top2_indices = torch.topk(original_logits[i], k=2, dim=0, largest=True, sorted=True).indices
        switch_label = top2_indices[1]
        
        # switch_label = (gt_label+1) % weights_matrix.shape[0] # num_classes
        embed2[i] = embed[i] - weights_matrix[gt_label, :] + weights_matrix[switch_label, :]

    # modified logits and preds
    modified_logits = embed2 @ weights_matrix.T #[n, 5]
    modified_preds = torch.argmax(modified_logits, dim=1)

    # construct confusion matrices
    original_cm = confusion_matrix(embed_gt_labels.cpu().numpy(), original_preds.cpu().numpy())
    modified_cm = confusion_matrix(embed_gt_labels.cpu().numpy(), modified_preds.cpu().numpy())
    
    # plot it
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_cm, cmap='Blues')
    plt.title("Original Confusion Matrix (n=500)")
    # plt.colorbar(label="Percentage")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add legend to the original confusion matrix
    legend_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.Blues(i / 4)) for i in range(5)]
    plt.subplot(1, 2, 2)
    plt.imshow(modified_cm, cmap='Blues')
    plt.title("Modified Confusion Matrix (n=500)")
    # plt.colorbar(label="Percentage")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1.05, 1), title='Percentage Range')

    plt.tight_layout()
    plt.savefig("Experiment_1_top_2.png")

    return 

    # # below is when you want to test with just a single embedding.
    # row_num = 20
    # embed = new_data[row_num:row_num+1, :] # [1, hidden_dim]
    # embed_gt_label = labels[row_num].long() # Convert fp to int since label should be a class number
    
    # # embed = new_data[1:2, :]
    # # embed_gt_label = labels[1].long()

    # # Sanity checking dimensions
    # print(f"embed shape: {embed.shape}")
    # print(f"weights_matrix shape: {weights_matrix.shape}")
    
    # logits = embed @ weights_matrix.T
    # print(f"logits shape: {logits.shape}")
    # print(f"logits: {logits}")

    # print(f"ground truth label: {embed_gt_label}")
    
    # w_row_0 = weights_matrix[4, :] # [1, hidden_dim]
    # w_row_1 = weights_matrix[0, :] # [1, hidden_dim]

    # embed2 = embed - w_row_0 + w_row_1

    # new_logits = embed2 @ weights_matrix.T
    # new_label = torch.argmax(new_logits)

    # print(f"new_logits: {new_logits}")
    # print(f"new label: {new_label}")
        
    # return

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
    
    args = parser.parse_args()

    setup(args.weights_path, args.data_path, args.labels_path, args.embedding_size, args.num_classes)
    