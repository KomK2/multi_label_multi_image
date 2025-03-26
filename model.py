import torch.nn as nn
import torch.nn.functional as F

class DualHeadCNN(nn.Module):
    def __init__(self, num_shapes=3, num_colors=3):
        super().__init__()
        # Shared feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Separate classification heads
        self.shape_head = nn.Linear(64, num_shapes)
        self.color_head = nn.Linear(64, num_colors)

    def forward(self, x):
        # Shared features
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        
        # Separate predictions
        shape_logits = self.shape_head(features)
        color_logits = self.color_head(features)
        
        return shape_logits, color_logits