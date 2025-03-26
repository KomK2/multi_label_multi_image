import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image

from bounding_box.fast_rcnn_dataset import ShapesFRCNNDataset
from model import DualHeadCNN  # adjust if needed
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Define Faster R-CNN model ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Paths ---
train_csv       = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
test_csv        = "/home/kiran/projects/github/test/all-shapes-and-colors/test.csv"
dataset_dir     = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
checkpoint_path = "/home/kiran/projects/github/test/bounding_box/outputs/model_epoch_1.pth"
dual_head_path  = "dual_head_cnn.pth"

# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load training dataset to determine number of classes ---
train_dataset   = ShapesFRCNNDataset(train_csv, dataset_dir, transform=transforms.ToTensor())
num_shapes      = len(train_dataset.shape_to_id)
num_classes     = num_shapes + 1  # +1 for background

# --- Initialize and load the detection model ---
detection_model = get_model(num_classes)
detection_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
detection_model.to(device)
detection_model.eval()

# --- Initialize and load the classification model ---
classification_model = DualHeadCNN(num_shapes=3, num_colors=3).to(device)
classification_model.load_state_dict(torch.load(dual_head_path, map_location=device))
classification_model.eval()

# --- Transforms ---
detection_transform = transforms.ToTensor()
classification_transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])

# --- Label decoding ---
idx2shape = {0: "triangle", 1: "square", 2: "circle"}
idx2color = {0: "red", 1: "blue", 2: "green"}

# --- Load test CSV ---
test_df = pd.read_csv(test_csv)
confidence_threshold = 0.5

# --- Store predictions for CSV output ---
predictions_list = []

for _, row in test_df.iterrows():
    image_rel_path = row["image_path"]
    image_path = os.path.join(dataset_dir, image_rel_path)

    # Load and transform image for detection
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image_pil).to(device)

    # Run Faster R-CNN inference
    with torch.no_grad():
        prediction = detection_model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Prepare patches for DualHeadCNN
    patches_list = []
    for box, score in zip(boxes, scores):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        patch_pil = image_pil.crop((x1, y1, x2, y2))
        patch_tensor = classification_transform(patch_pil)
        patches_list.append(patch_tensor)

    # If no patches above threshold
    if not patches_list:
        predictions_list.append({
            "image_path": image_rel_path,
            "label": "[]"
        })
        continue

    # DualHeadCNN inference
    input_batch = torch.stack(patches_list).to(device)
    with torch.no_grad():
        shape_logits, color_logits = classification_model(input_batch)
        shape_preds = torch.argmax(shape_logits, dim=1).cpu().numpy()
        color_preds = torch.argmax(color_logits, dim=1).cpu().numpy()

    # Decode shape-color pairs
    results = list(set((idx2shape[s], idx2color[c]) for s, c in zip(shape_preds, color_preds)))
    
    # Store for CSV
    predictions_list.append({
        "image_path": image_rel_path,
        "label": str(results)
    })

# --- Write output CSV ---
submission_df = pd.DataFrame(predictions_list, columns=["image_path", "label"])
submission_df.to_csv("submission_1.csv", index=False)
