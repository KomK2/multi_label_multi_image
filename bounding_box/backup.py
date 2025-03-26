import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from fast_rcnn_dataset import ShapesFRCNNDataset


def collate_fn(batch):
    # Filter out samples with no bounding boxes
    filtered_batch = [item for item in batch if item[1]["boxes"].numel() > 0]
    if len(filtered_batch) == 0:
        # Return empty lists so that we can check in the training loop.
        return [], []
    return tuple(zip(*filtered_batch))


# Function to create the model with a custom number of classes
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one (num_classes includes background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training loop function
def train(model, dataloader, device, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            # Skip if the batch is empty
            if len(images) == 0:
                continue

            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1
            progress_bar.set_postfix(loss=epoch_loss/num_batches if num_batches > 0 else 0)
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No valid batches found.")


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths and transform
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()

    # Load dataset and create DataLoader
    dataset = ShapesFRCNNDataset(csv_file, dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Number of classes: unique shapes + 1 for background
    num_shapes = len(dataset.shape_to_id)
    num_classes = num_shapes + 1

    # Get model and move it to device
    model = get_model(num_classes)
    model.to(device)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train the model
    num_epochs = 10
    train(model, dataloader, device, optimizer, num_epochs)

    # Save the fine-tuned model
    torch.save(model.state_dict(), "fasterrcnn_shapes.pth")
