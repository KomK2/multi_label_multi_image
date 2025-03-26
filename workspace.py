import torch
from dataset import ShapesBoundingBoxDataset  # Make sure this import path is correct
from model import DualHeadCNN  # Make sure this import path is correct
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

def custom_collate(batch):
    # Filter out samples that don't have any patches
    valid_samples = [sample for sample in batch if len(sample[0]) > 0]
    
    # If no sample in the batch has patches, return empty tensors.
    if len(valid_samples) == 0:
        empty_patches = torch.empty(0, 3, 75, 75)  # adjust if using a different fixed size
        return empty_patches, torch.tensor([]), torch.tensor([])

    all_patches = []
    all_shape_labels = []
    all_color_labels = []
    for patches, shape_labels, color_labels, _ in valid_samples:
        all_patches.extend(patches)
        all_shape_labels.extend(shape_labels)
        all_color_labels.extend(color_labels)
    
    all_patches = torch.stack(all_patches)
    all_shape_labels = torch.tensor(all_shape_labels)
    all_color_labels = torch.tensor(all_color_labels)
    
    return all_patches, all_shape_labels, all_color_labels


# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    for images, shape_labels, color_labels in dataloader:
        # If this batch has no patches, skip it.
        if images.size(0) == 0:
            continue
        
        images = images.to(device)
        shape_labels = shape_labels.to(device)
        color_labels = color_labels.to(device)
        
        optimizer.zero_grad()
        shape_logits, color_logits = model(images)
        loss_shape = criterion(shape_logits, shape_labels)
        loss_color = criterion(color_logits, color_labels)
        loss = loss_shape + loss_color
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        batch_count += images.size(0)
        
    return total_loss / batch_count if batch_count > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust these paths to your local setup.
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()
    
    dataset = ShapesBoundingBoxDataset(csv_file, dataset_dir, transform=transform)
    
    dataset = ShapesBoundingBoxDataset(csv_file, dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    
    model = DualHeadCNN(num_shapes=3, num_colors=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), "dual_head_cnn.pth")

if __name__ == '__main__':
    main()