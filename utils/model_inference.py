# utils/model_inference.py

import torch
import torch.nn as nn
import pickle
import numpy as np


# Residual Block used inside the Residual MLP classifier
class ResidualBlock(nn.Module):
    """
    A basic fully-connected residual block:
    - Linear layer → BatchNorm → ReLU
    - Optional shortcut projection when input/output dims differ
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

        # Shortcut connection (projection if dimension mismatch)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out += residual
        return self.relu(out)


# ---------------------------------------------------------
# Residual MLP Model (copied from training notebook)
# ---------------------------------------------------------
class ResidualMLP(nn.Module):
    """
    Multi-layer perceptron with stacked residual blocks.
    Used for classifying yoga poses based on angle features.
    """
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []

        # First block: input_dim → hidden_dims[0]
        layers.append(ResidualBlock(input_dim, hidden_dims[0]))

        # Intermediate hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1]))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Load trained model + metadata
def load_model():
    """
    Load the trained yoga pose classifier and associated meta-information.
    Returns:
        - model: PyTorch model in eval() mode
        - meta: dict containing input_dim, pose names, scaler statistics, etc.
    """
    # Load metadata (input_dim, hidden_dims, scaler stats, etc.)
    with open("models/yoga_angle_resnet_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # Retrieve parameters from metadata
    input_dim = meta.get("input_dim", 12)                  # Adjust based on number of angle features
    hidden_dims = meta.get("hidden_dims", [64, 128, 64])   # Architecture used during training
    num_classes = meta.get("num_classes", 5)

    # Initialize model architecture
    model = ResidualMLP(input_dim, hidden_dims, num_classes)

    # Load trained weights
    state_dict = torch.load("models/yoga_angle_resnet.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, meta


# Global model load (executed once at import time)
try:
    model, meta = load_model()
    pose_names = meta.get("pose_names",
                          ["downdog", "goddess", "plank", "tree", "warrior2"])
    scaler_mean = meta.get("scaler_mean", None)
    scaler_std = meta.get("scaler_std", None)

except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None
    meta = {}
    pose_names = ["downdog", "goddess", "plank", "tree", "warrior2"]
    scaler_mean = None
    scaler_std = None


# Pose prediction function
def predict_pose(angles):
    """
    Predict yoga pose class from a vector of joint angles.

    Parameters
    ----------
    angles : list or np.ndarray
        List/array of angle features extracted from landmarks.

    Returns
    -------
    pose_name : str
        Predicted pose label.

    confidence_pct : float
        Confidence score (0–100%).

    probabilities : np.ndarray
        Full probability distribution across all classes.
    """
    if model is None:
        return "Model not loaded", 0.0, np.zeros(5)

    # Convert list to numpy array
    if isinstance(angles, list):
        angles = np.array(angles)

    # Normalize using saved training-time scaler (if available)
    if scaler_mean is not None and scaler_std is not None:
        angles = (angles - scaler_mean) / scaler_std

    # Convert to tensor
    angles_tensor = torch.FloatTensor(angles).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(angles_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    pose_name = pose_names[predicted.item()]
    confidence_pct = confidence.item() * 100

    return pose_name, confidence_pct, probabilities[0].numpy()
