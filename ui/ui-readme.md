# 🧘 FitCheck – Yoga Pose Feedback System

**Real-time yoga pose analysis with AI-powered feedback and visual angle comparison**

---

## Overview

FitCheck is an intelligent yoga pose correction system that uses computer vision and deep learning to:
- Detect body landmarks using MediaPipe Pose
- Classify poses with a trained ResNet model
- Generate feedback based on **model statistics** (not hardcoded values)
- Display interactive radar charts comparing your form to typical expert form

**Supported Poses:** Downdog, Goddess, Plank, Tree, Warrior II

---

## Quick Start

### 1. Setup Environment

```bash
cd /path/to/Deep-Learning-Fitness-Poses-Correction

# Create virtual environment
python3 -m venv yoga_env

# Activate it
source yoga_env/bin/activate          # macOS/Linux
# OR
yoga_env\Scripts\activate              # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
- `gradio` - Web UI framework
- `torch` - Deep learning model
- `mediapipe` - Pose detection
- `opencv-python` - Image processing
- `plotly` - Interactive radar charts
- `numpy`, `pillow` - Data handling

### 3. Launch Application

```bash
python app.py
```

**Expected output:**
```
Starting FitCheck application...
✓ All modules loaded successfully
✓ Available poses in angle_stats: ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']
Launching Gradio interface...
Running on local URL: http://127.0.0.1:7860
```

### 4. Open Browser

Navigate to: **http://127.0.0.1:7860**

---

## Using the Interface

### Step 1: Upload Image
- Click "📤 Upload Your Yoga Pose"
- Choose an image file or drag & drop
- Supported formats: JPG, PNG

### Step 2: Analyze
- Click "🔍 Analyze Pose" button
- Wait for processing (~2-3 seconds)

### Step 3: Review Results

The interface displays:

#### **Left Panel:**
1. **Detected Landmarks** - Your uploaded image with pose skeleton overlay
2. **Predicted Pose** - Classified pose name
3. **Confidence (%)** - Model prediction confidence

#### **Bottom Section:**
1. **Detailed Feedback** - Text-based corrections and good alignments
   - ⚠️ Corrections needed (mild/moderate/severe)
   - ✅ Good alignment points
   - Overall score (0-100)

2. **Radar Chart** - Visual angle comparison
   - 🟢 Green area: Typical expert form (from model statistics)
   - 🔵 Blue area: Your actual form
   - Overlap = good alignment
   - Gaps = areas needing correction

---

## ⚙️ Advanced Settings

Click "⚙️ Advanced Settings" to adjust:

### Sensitivity (std_factor)
- **Range:** 0.5 - 3.0
- **Default:** 1.0
- **Lower value** (0.5-0.9): Stricter feedback, flags more deviations
- **Higher value** (1.5-3.0): More lenient, only flags major issues

### Minimum Deviation (degrees)
- **Range:** 5° - 30°
- **Default:** 20°
- Minimum angle difference required to trigger correction

---

## 📂 Project Structure

```
Deep-Learning-Fitness-Poses-Correction/
├── app.py                          # Main Gradio UI application
├── requirements.txt                # Python dependencies
├── ui-readme.md                    # This file
│
├── utils/                          # Core functionality modules
│   ├── pose_extraction.py          # MediaPipe landmark detection
│   ├── angle_calculation.py        # Joint angle computation (0-360°)
│   ├── model_inference.py          # ResNet pose classification
│   └── feedback_generator.py       # Statistics-based feedback
│
├── models/                         # Trained model files
│   ├── yoga_angle_resnet.pt        # Model weights
│   └── yoga_angle_resnet_meta.pkl  # Metadata with angle statistics
│
├── test/                           # Sample test images
│   ├── test1.jpg
│   ├── test2.jpg
│   └── test3.jpg
│
└── feedback_image_complete.ipynb   # Jupyter notebook version
```

---

## How It Works

### Architecture Overview

```
Image Input
    ↓
MediaPipe Pose Detection (33 landmarks)
    ↓
Angle Calculation (12 joint angles, 0-360°)
    ↓
ResNet Classification (5 pose classes)
    ↓
Statistical Comparison (model mean/std)
    ↓
Feedback Generation + Radar Chart
```

### Key Technical Details

#### **Angle Representation**
- **Calculation Layer:** 0-360° (preserves full geometric information)
- **Display Layer:** 0-180° (human-readable format)
- **Conversion:** `normalize_to_180()` for visualization only

#### **Circular Difference**
- Formula: `(a - b + 180) % 360 - 180`
- Handles angle wrapping correctly
- Example: 350° vs 10° = 20° difference (not 340°)

#### **Dynamic Thresholds**
- Each joint has different natural variation
- Threshold = `max(min_deg, std_factor × model_std)`
- Based on actual training data distribution

#### **Model Statistics Source**
- File: `models/yoga_angle_resnet_meta.pkl`
- Contains: `angle_stats[pose_name]["mean"]` and `["std"]`
- Computed from expert-labeled training dataset
- **No hardcoded ideal angles!**

---

## Understanding the Radar Chart

### Visual Elements

```
         Left Shoulder
              ↑
             /|\
            / | \
   Left → /  |  \ → Right
    Hip  /   |   \   Hip
        /    |    \
       /     ↓     \
  Left Knee    Right Knee
```

### Interpretation

| Observation | Meaning |
|-------------|---------|
| Blue inside green | Your angle is smaller than typical |
| Blue outside green | Your angle is larger than typical |
| Perfect overlap | Excellent alignment |
| Large gap | Significant deviation needing correction |

### Radial Axis
- Range: 0° to 180°
- Tick marks every 30°
- All angles converted to 0-180° for readability

---

## Example Feedback

### Good Form Example
```
Feedback Analysis for Downdog

Overall Alignment Score: 88.9 / 100
(Checked 9 joints against model statistics)

Corrections Needed:
  ⚠️ (mild) Increase angle of left Left Knee. 
     (Left Knee: 178.8°, Typical: 179.2°, Diff: 1.8°)

✅ Good Alignment:
  ✅ Right Knee: 172.2° (Avg: 179.9°)
  ✅ Left Hip: 60.6° (Avg: 144.4°)
  ✅ Right Hip: 179.5° (Avg: 177.4°)
  ✅ Left Shoulder: 178.8° (Avg: 179.2°)
  ✅ Right Shoulder: 179.5° (Avg: 177.4°)
  ✅ Left Elbow: 178.8° (Avg: 179.2°)
  ✅ Right Elbow: 172.2° (Avg: 179.9°)
  ✅ Neck: 101.6° (Avg: 161.8°)
================================================================
```

---

## Troubleshooting

### Issue: "No pose detected"
**Causes:**
- Image doesn't show full body
- Person too small or far from camera
- Poor lighting/contrast

**Solutions:**
- Use images with clear full-body view
- Ensure person occupies >30% of frame
- Use well-lit photos

---

### Issue: Plotly radar chart not displaying
**Cause:** Missing Plotly dependency

**Solution:**
```bash
pip install --upgrade plotly
```

---

### Issue: Low confidence predictions
**Causes:**
- Unusual pose variation
- Partial occlusion
- Ambiguous pose

**Solutions:**
- Try standard pose variations
- Ensure all limbs are visible
- Check if pose is in supported list

---

### Issue: Import errors
**Cause:** Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install gradio torch mediapipe opencv-python plotly numpy pillow
```

---

### Issue: Gradio theme error (old version)
**Error:** `TypeError: BlockContext.__init__() got an unexpected keyword argument 'theme'`

**Solution:**
```bash
pip install --upgrade gradio
```

If still failing, edit `app.py` line ~233:
```python
# Change from:
with gr.Blocks(theme=gr.themes.Soft()) as demo:

# To:
with gr.Blocks() as demo:
```

---

## Stopping the Application

### Stop the server
Press `Ctrl + C` in the terminal

### Deactivate virtual environment
```bash
deactivate
```


