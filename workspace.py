import torch
from dataset import ShapesBoundingBoxDataset  
from model import DualHeadCNN  
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from label_embedding import EmbeddingCNN

# Custom collate function to handle batches with variable-sized data
def custom_collate(batch):
    # Filter out invalid samples (empty patches)
    valid_samples = [sample for sample in batch if len(sample[0]) > 0]
    
    # If no valid samples, return empty tensors
    if len(valid_samples) == 0:
        empty_patches = torch.empty(0, 3, 75, 75)  
        return empty_patches, torch.tensor([]), torch.tensor([])

    # Aggregate patches, shape labels, and color labels from valid samples
    all_patches = []
    all_shape_labels = []
    all_color_labels = []
    for patches, shape_labels, color_labels, _ in valid_samples:
        all_patches.extend(patches)
        all_shape_labels.extend(shape_labels)
        all_color_labels.extend(color_labels)
    
    # Convert lists to tensors
    all_patches = torch.stack(all_patches)
    all_shape_labels = torch.tensor(all_shape_labels)
    all_color_labels = torch.tensor(all_color_labels)
    
    return all_patches, all_shape_labels, all_color_labels

def main():
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to dataset and CSV file
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()  # Transform to convert images to tensors

    # Initialize the dataset
    dataset = ShapesBoundingBoxDataset(csv_file, dataset_dir, transform=transform)

    # Determine the number of shapes and colors from the dataset
    num_shapes = len(dataset.shapes)  # e.g., ["triangle", "square", "circle"]
    num_colors = len(dataset.colors)  # e.g., ["red", "blue", "green"]

    # Create a data loader with the custom collate function
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    # Initialize the model with dynamic shape and color counts
    model = EmbeddingCNN(num_shapes=num_shapes, num_colors=num_colors, embed_dim=64).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Track total loss for the epoch
        count_samples = 0  # Track the number of processed samples

        for patches, shape_labels, color_labels in dataloader:
            # Skip empty batches
            if patches.size(0) == 0:
                continue

            # Move data to the appropriate device
            patches = patches.to(device)
            shape_labels = shape_labels.to(device)
            color_labels = color_labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            shape_logits, color_logits = model(patches)

            # Compute losses for shape and color predictions
            loss_shape = criterion(shape_logits, shape_labels)
            loss_color = criterion(color_logits, color_labels)
            loss = loss_shape + loss_color  # Total loss

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Accumulate loss and sample count
            epoch_loss += loss.item() * patches.size(0)
            count_samples += patches.size(0)

        # Compute average loss for the epoch
        avg_loss = epoch_loss / count_samples if count_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model's state
    torch.save(model.state_dict(), "embedding_cnn.pth")

if __name__ == '__main__':
    main()