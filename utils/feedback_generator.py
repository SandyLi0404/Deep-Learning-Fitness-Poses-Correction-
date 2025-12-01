# utils/feedback_generator.py

import pandas as pd
import numpy as np


# Load per-pose statistical angle data (mean & std)
def load_pose_stats():
    """
    Load statistical angle summaries (mean and std) for each yoga pose.
    
    Returns
    -------
    stats : dict
        Dictionary mapping pose names to their statistical angle data:
        - mean: average angle values across the dataset
        - std: standard deviation of angle values
        - columns: angle names (if available)
    """
    stats = {}
    
    # Supported pose names (must match filenames in angles/ folder)
    poses = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
    
    for pose in poses:
        csv_path = f'angles/{pose}.csv'   # Adjust if your folder uses different casing

        try:
            df = pd.read_csv(csv_path)
            
            # Select numeric columns only (ignore filenames or metadata)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            stats[pose] = {
                'mean': df[numeric_columns].mean().values,
                'std': df[numeric_columns].std().values,
                'columns': list(numeric_columns)
            }
        
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")
            
            # Use fallback values if dataset is missing or corrupted
            stats[pose] = {
                'mean': np.zeros(12),
                'std': np.ones(12),
                'columns': []
            }
    
    return stats


# Load global pose statistics at import time
pose_stats = load_pose_stats()


# Generate personalized feedback for a predicted pose
def generate_feedback(angles, pose_name):
    """
    Generate personalized corrective feedback based on joint angle deviations.
    
    Parameters
    ----------
    angles : list or np.ndarray
        Angle vector for the input image (from angle extractor).
    
    pose_name : str
        Name of the predicted pose (e.g., 'tree', 'plank').
    
    Returns
    -------
    str
        A human-readable feedback message describing joint misalignment.
    """
    
    if pose_name not in pose_stats:
        return "⚠️ Unknown pose type (statistics unavailable)."
    
    stats = pose_stats[pose_name]
    mean_angles = stats['mean']
    std_angles = stats['std']
    
    # Validate angle vector length
    if len(angles) != len(mean_angles):
        return f"⚠️ Angle length mismatch: expected {len(mean_angles)}, received {len(angles)}"
    
    feedback = []
    
    # Assign human-readable angle names when available
    if stats['columns']:
        angle_names = stats['columns']
    else:
        angle_names = [f'Angle {i+1}' for i in range(len(angles))]
    
    # Evaluate angle deviation from pose-specific mean
    for i, (angle, mean, std) in enumerate(zip(angles, mean_angles, std_angles)):
        
        if std == 0:   # Avoid division by zero
            continue
        
        deviation = abs(angle - mean)
        
        # Flag angles above 2 standard deviations as problematic
        if deviation > 2 * std:
            angle_name = angle_names[i] if i < len(angle_names) else f'Joint {i}'
            
            if angle > mean:
                feedback.append(
                    f"⚠️ {angle_name}: Too extended ({angle:.1f}° vs ideal {mean:.1f}°)"
                )
            else:
                feedback.append(
                    f"⚠️ {angle_name}: Too compressed ({angle:.1f}° vs ideal {mean:.1f}°)"
                )
    
    # If no issue found, return a positive message
    if not feedback:
        return "✅ Great form! All joints fall within ideal alignment ranges."
    
    return "\n".join(feedback)
