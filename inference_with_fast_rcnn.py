# Import necessary libraries
import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image

# Import custom modules
from bounding_box.fast_rcnn_dataset import ShapesFRCNNDataset  
from label_embedding import EmbeddingCNN 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import ShapesBoundingBoxDataset  

# Function to get the Faster R-CNN model with a custom number of classes
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Define paths for training, testing, and dataset directories
train_csv       = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
test_csv        = "/home/kiran/projects/github/test/all-shapes-and-colors/test.csv"
dataset_dir     = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"

# Define paths for model checkpoints
checkpoint_path = "/home/kiran/projects/github/test/bounding_box/outputs/model_epoch_1.pth"
embedding_cnn_path  = "embedding_cnn.pth"

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training dataset for Faster R-CNN
train_dataset   = ShapesFRCNNDataset(train_csv, dataset_dir, transform=transforms.ToTensor())
num_shapes      = len(train_dataset.shape_to_id)  # Number of shape classes in detection
num_classes     = num_shapes + 1                  # +1 for background in Faster R-CNN

# Load the detection model and its weights
detection_model = get_model(num_classes)
detection_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
detection_model.to(device)
detection_model.eval()

# Load the dummy dataset for classification
dummy_dataset = ShapesBoundingBoxDataset(csv_file=train_csv, dataset_dir=dataset_dir)
num_shapes_clf = len(dummy_dataset.shapes)  # Number of shape classes for classification
num_colors_clf = len(dummy_dataset.colors)  # Number of color classes for classification

# Load the classification model and its weights
classification_model = EmbeddingCNN(
    num_shapes=num_shapes_clf,
    num_colors=num_colors_clf,
    embed_dim=64
).to(device)
classification_model.load_state_dict(torch.load(embedding_cnn_path, map_location=device))
classification_model.eval()

# Define transformations for detection and classification
detection_transform = transforms.ToTensor()
classification_transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])

# Read the test dataset CSV
test_df = pd.read_csv(test_csv)
confidence_threshold = 0.5  # Confidence threshold for detection

# Initialize a list to store predictions
predictions_list = []

# Iterate through each row in the test dataset
for _, row in test_df.iterrows():
    image_rel_path = row["image_path"]  # Get the relative image path
    image_path = os.path.join(dataset_dir, image_rel_path)  # Construct the full image path

    # Load the image and convert it to RGB
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = detection_transform(image_pil).to(device)

    # Perform object detection
    with torch.no_grad():
        detection_output = detection_model([image_tensor])[0] 
    
    # Extract bounding boxes, scores, and labels
    boxes = detection_output["boxes"].cpu().numpy()
    scores = detection_output["scores"].cpu().numpy()
    labels = detection_output["labels"].cpu().numpy()  
    patches_list = []  # List to store cropped image patches
    box_indices = []   # List to store indices of valid boxes

    # Iterate through detected boxes and filter by confidence threshold
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)  # Get box coordinates
        patch_pil = image_pil.crop((x1, y1, x2, y2))  # Crop the image patch
        patch_tensor = classification_transform(patch_pil)  # Apply classification transform
        patches_list.append(patch_tensor)
        box_indices.append(i)

    # If no valid patches, append an empty label and continue
    if not patches_list:
        predictions_list.append({
            "image_path": image_rel_path,
            "label": "[]"
        })
        continue

    # Stack patches into a batch and perform classification
    input_batch = torch.stack(patches_list).to(device)
    with torch.no_grad():
        shape_logits, color_logits = classification_model(input_batch)

        # Get predicted shape and color indices
        shape_preds = torch.argmax(shape_logits, dim=1).cpu().numpy()
        color_preds = torch.argmax(color_logits, dim=1).cpu().numpy()

    # Map indices to shape and color names
    results = []
    for shp_idx, clr_idx in zip(shape_preds, color_preds):
        shp_name = dummy_dataset.shapes[shp_idx]
        clr_name = dummy_dataset.colors[clr_idx]
        results.append((shp_name, clr_name))

    # Remove duplicate predictions
    results = list(set(results))

    # Append predictions to the list
    predictions_list.append({
        "image_path": image_rel_path,
        "label": str(results)
    })

# Create a DataFrame from predictions and save to CSV
submission_df = pd.DataFrame(predictions_list, columns=["image_path", "label"])
submission_df.to_csv("submission_2.csv", index=False)
print("Wrote predictions to submission_2.csv")
