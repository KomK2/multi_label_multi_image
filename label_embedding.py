import torch
import torch.nn as nn

class LabelEmbeddingHead(nn.Module):
    """
    A flexible 'head' that stores an embedding (dictionary) for each label (shape or color).
    The final logits for each label come from dot products with the CNN feature vector.
    """
    def __init__(self, num_labels: int, embed_dim: int):
        super().__init__()
        # shape: (num_labels, embed_dim)
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, embed_dim)  # random init
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [batch_size, embed_dim]
        returns:  [batch_size, num_labels]
        We do a matrix multiply of the form (B x E) x (E x L) -> (B x L).
        """
        # Transpose label_embeddings to shape (embed_dim, num_labels)
        return features @ self.label_embeddings.T

class EmbeddingCNN(nn.Module):
    """
    CNN feature extractor + label embedding heads for shapes and colors.
    """
    def __init__(self, num_shapes: int, num_colors: int, embed_dim=64):
        super().__init__()
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
            nn.AdaptiveAvgPool2d((1, 1))  # final shape: [batch_size, 64, 1, 1]
        )
        self.embed_dim = embed_dim

        self.shape_head = LabelEmbeddingHead(num_shapes, embed_dim)
        self.color_head = LabelEmbeddingHead(num_colors, embed_dim)


    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, 3, H, W]
        returns: (shape_logits, color_logits)
        """
        # 1) Extract CNN feature
        features = self.conv_layers(x)                 # shape: [batch_size, 64, 1, 1]
        features = features.view(features.size(0), -1) # shape: [batch_size, 64]

        # 2) Dot product with the shape and color embeddings
        shape_logits = self.shape_head(features)  # [batch_size, num_shapes]
        color_logits = self.color_head(features)  # [batch_size, num_colors]

        return shape_logits, color_logits
