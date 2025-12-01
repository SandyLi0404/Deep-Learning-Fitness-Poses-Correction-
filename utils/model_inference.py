# model_inference.py - Updated version matching new_version_feedback_image.ipynb

import torch
import torch.nn as nn
import pickle
import numpy as np


# Residual Block used inside the AngleResNet classifier
class ResidualBlock(nn.Module):
    """
    A residual block with two linear layers, batch norm, ReLU, and dropout.
    """
    def __init__(self, dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        out = out + x  # Residual connection
        out = self.act(out)
        return out


# Main model architecture
class AngleResNet(nn.Module):
    """
    Residual MLP for classifying yoga poses based on angle features.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=256, num_blocks=3, dropout=0.5):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.blocks(h)
        logits = self.head(h)
        return logits


# Load trained model + metadata
def load_model():
    """
    Load the trained yoga pose classifier and associated meta-information.
    
    Returns
    -------
    model : torch.nn.Module
        PyTorch model in eval() mode.
        
    meta : dict
        Metadata containing class_names, feature_cols, angle_stats, etc.
    """
    # Load metadata
    with open("models/yoga_angle_resnet_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # Retrieve model parameters from metadata
    class_names = meta.get("class_names", ["downdog", "goddess", "plank", "tree", "warrior2"])
    feature_cols = meta.get("feature_cols", [])
    in_dim = len(feature_cols)
    num_classes = len(class_names)
    hidden_dim = meta.get("hidden_dim", 256)

    # Initialize model architecture
    model = AngleResNet(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_blocks=3,
        dropout=0.5
    )

    # Load trained weights
    device = torch.device("cpu")
    state_dict = torch.load("models/yoga_angle_resnet.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, meta


# Global model load (executed once at import time)
try:
    model, meta = load_model()
    class_names = meta.get("class_names", ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"])
    feature_cols = meta.get("feature_cols", [])
    angle_stats = meta.get("angle_stats", {})

except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None
    meta = {}
    class_names = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]
    feature_cols = []
    angle_stats = {}


def predict_pose(angle_dict):
    """
    Predict yoga pose class from a dictionary of joint angles.

    Parameters
    ----------
    angle_dict : dict
        Dictionary containing angle measurements with keys matching feature_cols.

    Returns
    -------
    pose_name : str
        Predicted pose label.

    confidence_pct : float
        Confidence score (0–100%).

    probabilities : np.ndarray
        Full probability distribution across all classes.
    """
    if model is None or not feature_cols:
        return "Model not loaded", 0.0, np.zeros(5)

    # Build feature vector from angle dictionary
    feats = [angle_dict.get(name, 0.0) for name in feature_cols]
    
    # Convert to tensor
    device = torch.device("cpu")
    x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        
        # Apply temperature scaling to reduce overconfidence
        # Temperature > 1 makes the model less confident
        temperature = 1.5
        scaled_logits = logits / temperature
        
        probabilities = torch.softmax(scaled_logits, dim=1)[0].cpu().numpy()
        
        cls_idx = int(np.argmax(probabilities))
        pose_name = class_names[cls_idx]
        confidence_pct = probabilities[cls_idx] * 100

    return pose_name, confidence_pct, probabilities