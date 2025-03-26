import os
import torch
import torchvision
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image

from fast_rcnn_dataset import ShapesFRCNNDataset  # adjust if needed

# --- Define get_model ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset and model paths ---
train_csv = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
test_csv = "/home/kiran/projects/github/test/all-shapes-and-colors/test.csv"
dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
checkpoint_path = "/home/kiran/projects/github/test/bounding_box/outputs/model_epoch_1.pth"

# --- Load training dataset to get label mappings ---
transform = transforms.ToTensor()
train_dataset = ShapesFRCNNDataset(train_csv, dataset_dir, transform=transform)
num_shapes = len(train_dataset.shape_to_id)
num_classes = num_shapes + 1

# --- Load model ---
model = get_model(num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# --- Load test CSV ---
test_df = pd.read_csv(test_csv)

# --- Inference settings ---
confidence_threshold = 0.5

# --- Run inference on test images ---
for idx, row in test_df.iterrows():
    image_rel_path = row["image_path"]
    image_path = os.path.join(dataset_dir, image_rel_path)

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # Convert image for display
    image_np = image_tensor.cpu().permute(1, 2, 0).numpy()

    # Plot predictions
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image_np)

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Label: {label}, {score:.2f}",
                color="yellow", fontsize=12)

    ax.set_title(f"Prediction: {image_rel_path}")
    ax.axis("off")
    plt.show()
