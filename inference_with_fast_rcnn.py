import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from bounding_box.fast_rcnn_dataset import ShapesFRCNNDataset
from model import DualHeadCNN  # adjust if needed
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# 1) Define Faster R-CNN and DualHeadCNN Models
def get_model(num_classes):
    """Returns a Faster R-CNN model with a custom predictor layer."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Label mappings for DualHeadCNN
idx2shape = {0: "triangle", 1: "square", 2: "circle"}
idx2color = {0: "red", 1: "blue", 2: "green"}


# 2) Setup Paths and Device
train_csv       = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
test_csv        = "/home/kiran/projects/github/test/all-shapes-and-colors/test.csv"
dataset_dir     = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
checkpoint_path = "/home/kiran/projects/github/test/bounding_box/outputs/model_epoch_1.pth"
model_path      = "dual_head_cnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3) Load Datasets, Determine Num Classes
train_dataset = ShapesFRCNNDataset(train_csv, dataset_dir, transform=transforms.ToTensor())
num_shapes = len(train_dataset.shape_to_id)
num_classes = num_shapes + 1  # +1 for background


# 4) Initialize and Load the Detection Model
detection_model = get_model(num_classes)
detection_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
detection_model.to(device)
detection_model.eval()


# 5) Initialize and Load the Classification Model
classification_model = DualHeadCNN(num_shapes=3, num_colors=3).to(device)
classification_model.load_state_dict(torch.load(model_path, map_location=device))
classification_model.eval()


# 6) Define Transforms
# For the detection model: no forced 75×75 resize here
detection_transform = transforms.ToTensor()

# For cropping out patches and feeding into DualHeadCNN
classification_transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])


# 7) Inference on Test CSV
test_df = pd.read_csv(test_csv)
confidence_threshold = 0.5

for idx, row in test_df.iterrows():
    image_rel_path = row["image_path"]
    image_path = os.path.join(dataset_dir, image_rel_path)

    # --- Load and transform image for detection ---
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image_pil).to(device)

    # --- Faster R-CNN inference ---
    with torch.no_grad():
        prediction = detection_model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    # --- Crop patches for DualHeadCNN ---
    patches_list = []
    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        patch_pil = image_pil.crop((x1, y1, x2, y2))
        patch_tensor = classification_transform(patch_pil)
        patches_list.append(patch_tensor)

    if not patches_list:
        print(f"Image: {image_path}")
        print("Predicted shape-color pairs: [] (no detections above threshold)")
        continue

    # --- Classify patches with DualHeadCNN ---
    input_batch = torch.stack(patches_list).to(device)
    with torch.no_grad():
        shape_logits, color_logits = classification_model(input_batch)
        shape_preds = torch.argmax(shape_logits, dim=1).cpu().numpy()
        color_preds = torch.argmax(color_logits, dim=1).cpu().numpy()

    # --- Decode and print results ---
    results = list(set((idx2shape[s], idx2color[c]) for s, c in zip(shape_preds, color_preds)))
    print(f"Image: {image_path}")
    print("Predicted shape-color pairs:", results)
