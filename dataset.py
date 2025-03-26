import pandas as pd
import ast
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

class ShapesBoundingBoxDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, transform=None, area_threshold=100, use_matching=False):
        """
        Args:
            csv_file (str): Path to CSV file with columns: image_path, label.
            dataset_dir (str): Directory where the images are stored.
            transform (callable, optional): Transform to apply to each cropped patch.
            area_threshold (int): Minimum area for a contour to be considered a shape.
            use_matching (bool): (Unused in this approach)
        """
        self.df = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir

        # Exclude specific images by number
        excluded_ids = {'40', '1655', '3145', '1168', '286', '148', '2652', '2321', '3331', '3533', '3716', '4408', '4883', '4965'}
        def is_excluded(image_path):
            filename = os.path.splitext(os.path.basename(image_path))[0]
            if filename.startswith("img_"):
                image_num = filename.split("_")[1]
                return image_num in excluded_ids
            return False
        self.df = self.df[~self.df['image_path'].apply(is_excluded)].reset_index(drop=True)

        transform = transforms.Compose([
            transforms.Resize((75, 75)),
            transforms.ToTensor()
        ])

        self.transform = transform
        self.area_threshold = area_threshold
        self.use_matching = use_matching

        self.color_ranges = {
            'red': ((0, 100, 100), (10, 255, 255)),
            'blue': ((100, 150, 0), (140, 255, 255)),
            'green': ((40, 70, 70), (80, 255, 255))
        }

        self.shapes = ["triangle", "square", "circle"]
        self.colors = ["red", "blue", "green"]
        self.shape2idx = {s: i for i, s in enumerate(self.shapes)}
        self.color2idx = {c: i for i, c in enumerate(self.colors)}

    def detect_shape(self, contour):
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        sides = len(approx)
        if sides == 3:
            return "triangle"
        elif sides == 4:
            return "square"
        else:
            return "circle"

    def find_bounding_boxes(self, hsv, color_name):
        lower, upper = self.color_ranges[color_name]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > self.area_threshold:
                shape = self.detect_shape(contour)
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append({
                    'box': (x, y, w, h),
                    'detected_shape': shape,
                    'color': color_name
                })
        return boxes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_img_path = row["image_path"]
        img_path = os.path.join(self.dataset_dir, rel_img_path)

        image_cv = cv2.imread(img_path)
        if image_cv is None:
            raise ValueError(f"Image {img_path} not found")
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        detected_boxes = []
        for color_name in self.color_ranges.keys():
            boxes = self.find_bounding_boxes(hsv, color_name)
            detected_boxes.extend(boxes)

        label_str = row["label"]
        csv_labels = ast.literal_eval(label_str)
        csv_labels = [(s.lower(), c.lower()) for s, c in csv_labels]

        patches = []
        shape_labels = []
        color_labels = []
        for csv_shape, csv_color in csv_labels:
            match_found = False
            for i, box_info in enumerate(detected_boxes):
                if box_info['detected_shape'] == csv_shape and box_info['color'] == csv_color:
                    match_found = True
                    matched_box = detected_boxes.pop(i)
                    x, y, w, h = matched_box['box']
                    patch_cv = image_cv[y:y+h, x:x+w]
                    patch_rgb = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB)
                    patch_pil = Image.fromarray(patch_rgb)
                    if self.transform:
                        patch_pil = self.transform(patch_pil)
                    patches.append(patch_pil)
                    shape_labels.append(self.shape2idx[csv_shape])
                    color_labels.append(self.color2idx[csv_color])
                    break
            if not match_found:
                raise ValueError(f"No matching detection for CSV label {(csv_shape, csv_color)} in image {img_path}")

        return patches, shape_labels, color_labels, img_path

if __name__ == '__main__':
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"
    transform = transforms.ToTensor()

    dataset = ShapesBoundingBoxDataset(csv_file, dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (patches, shape_labels, color_labels, img_path) in enumerate(dataloader):
        print(f"Image: {img_path}")
        print(f"Shape labels: {shape_labels}")
        print(f"Color labels: {color_labels}")

        for j, patch in enumerate(patches):
            print(f"patch size {patch.size()}")
            if isinstance(patch, torch.Tensor):
                patch_np = patch.squeeze(0).permute(1, 2, 0).numpy()
            else:
                patch_np = np.array(patch)
            plt.imshow(patch_np)
            plt.title(f"Patch {j} - Shape: {shape_labels[j]}, Color: {color_labels[j]}")
            plt.axis('off')
            plt.show()

        if i >= 2:
            break
