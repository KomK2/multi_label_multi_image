import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import wandb

from fast_rcnn_dataset import ShapesFRCNNDataset

def collate_fn(batch):
    filtered_batch = [item for item in batch if item[1]["boxes"].numel() > 0]
    if len(filtered_batch) == 0:
        return [], []
    return tuple(zip(*filtered_batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for images, targets in progress_bar:
            if len(images) == 0:
                continue
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            num_batches += 1
            progress_bar.set_postfix(loss=val_loss/num_batches if num_batches > 0 else 0)
    return val_loss / num_batches if num_batches > 0 else None

def train(model, train_loader, val_loader, device, optimizer, num_epochs, output_dir):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            if len(images) == 0:
                continue
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1
            progress_bar.set_postfix(loss=epoch_loss/num_batches if num_batches > 0 else 0)

        # Save model checkpoint after every epoch
        checkpoint_path = f"{output_dir}/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        avg_train_loss = epoch_loss/num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Run validation and log both train and validation loss to wandb
        val_loss = validate(model, val_loader, device)
        if val_loss is not None:
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": val_loss})
        


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb for experiment tracking
    wandb.init(project="bounding_box_project")
    
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()
    
    dataset = ShapesFRCNNDataset(csv_file, dataset_dir, transform=transform)
    
    # Split dataset: 85% training, 15% validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Define number of classes (shapes + background)
    num_shapes = len(dataset.shape_to_id)
    num_classes = num_shapes + 1
    
    model = get_model(num_classes)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 10
    output_dir = "/home/kiran/projects/github/test/bounding_box/outputs"
    
    train(model, train_loader, val_loader, device, optimizer, num_epochs, output_dir)
