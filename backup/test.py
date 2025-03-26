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

from bounding_box.fast_rcnn_dataset import ShapesFRCNNDataset
from model import DualHeadCNN  # adjust if needed

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
model_path = "dual_head_cnn.pth"

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

# --- Extract patches ---
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])


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


    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    patches_list = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        patch = image.crop((x1, y1, x2, y2))
        patch_tensor = transform(patch)
        patches_list.append(patch_tensor)

    
    if not patches_list:
        print("No valid patches found.")
        exit()

    # --- Stack patches and run model ---
    input_batch = torch.stack(patches_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCNN(num_shapes=3, num_colors=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        shape_logits, color_logits = model(input_batch.to(device))
        shape_preds = torch.argmax(shape_logits, dim=1).cpu().numpy()
        color_preds = torch.argmax(color_logits, dim=1).cpu().numpy()

    # --- Label decoding ---
    idx2shape = {0: "triangle", 1: "square", 2: "circle"}
    idx2color = {0: "red", 1: "blue", 2: "green"}
    results = list(set((idx2shape[s], idx2color[c]) for s, c in zip(shape_preds, color_preds)))

    print(f"Image: {image_path}")
    print("Predicted shape-color pairs:", results)




