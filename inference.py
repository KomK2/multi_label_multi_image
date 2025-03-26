import torch
from torchvision import transforms
from model import DualHeadCNN
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

# --- Paths ---
csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/test.csv"
dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
model_path = "dual_head_cnn.pth"

# --- Load CSV and get first image path ---
df = pd.read_csv(csv_file)
image_rel_path = df.iloc[55]["image_path"]
image_path = os.path.join(dataset_dir, image_rel_path)

# --- Load and preprocess image ---
image_cv = cv2.imread(image_path)
if image_cv is None:
    raise ValueError(f"Image not found: {image_path}")
hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

# --- Color ranges from dataset ---
color_ranges = {
    'red': ((0, 100, 100), (10, 255, 255)),
    'blue': ((100, 150, 0), (140, 255, 255)),
    'green': ((40, 70, 70), (80, 255, 255))
}

def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    sides = len(approx)
    if sides == 3:
        return "triangle"
    elif sides == 4:
        return "square"
    else:
        return "circle"

# --- Extract patches ---
area_threshold = 100
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor()
])

patches = []
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            shape = detect_shape(contour)
            x, y, w, h = cv2.boundingRect(contour)
            patch_cv = image_cv[y:y+h, x:x+w]
            patch_rgb = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB)
            patch_pil = Image.fromarray(patch_rgb)
            patch_tensor = transform(patch_pil)
            patches.append(patch_tensor)
            # import pdb; pdb.set_trace()

if not patches:
    print("No valid patches found.")
    exit()

# --- Stack patches and run model ---
input_batch = torch.stack(patches)

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
