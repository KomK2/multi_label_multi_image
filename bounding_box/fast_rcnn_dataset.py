from matplotlib import pyplot as plt
import pandas as pd
import ast
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.patches as patches


class ShapesFRCNNDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, transform=None, area_threshold=100):
        self.df = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.area_threshold = area_threshold

        self.color_ranges = {
            'red': ((0, 100, 100), (10, 255, 255)),
            'blue': ((100, 150, 0), (140, 255, 255)),
            'green': ((40, 70, 70), (80, 255, 255))
        }

        # Dynamically extract unique shape names
        all_labels = self.df["label"].apply(ast.literal_eval)
        unique_shapes = sorted(list({s.lower() for row in all_labels for s, _ in row}))
        self.shape_to_id = {s: i+1 for i, s in enumerate(unique_shapes)}  # 1-based class IDs

    def __len__(self):
        return len(self.df)

    def detect_shape(self, contour):
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        sides = len(approx)
        if sides == 3:
            return "triangle"
        elif sides == 4:
            return "square"
        else:
            return "circle"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_dir, row["image_path"])
        label_str = row["label"]

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gt_shape_color = [(s.lower(), c.lower()) for s, c in ast.literal_eval(label_str)]

        boxes, labels = [], []

        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.area_threshold:
                    continue
                detected_shape = self.detect_shape(contour)
                if (detected_shape, color_name) in gt_shape_color:
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append([x, y, x+w, y+h])
                    labels.append(self.shape_to_id[detected_shape])

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = self.transform(image_pil)

        return image_tensor, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }


if __name__ == '__main__':
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()

    dataset = ShapesFRCNNDataset(csv_file=csv_file, dataset_dir=dataset_dir, transform=transform)

    # Loop through a few samples to test dataset usage
    for i in range(146,154):
        image_tensor, target = dataset[i]
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()

        # Convert image tensor to numpy (H, W, C) for display
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Plot the image and overlay the bounding boxes
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(image_np)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # ax.text(x1, y1 - 5, f"Shape: {label}", color='yellow', fontsize=12, weight='bold')
        ax.set_title(f"Sample {i} with Bounding Boxes")
        ax.axis('off')
        plt.show()