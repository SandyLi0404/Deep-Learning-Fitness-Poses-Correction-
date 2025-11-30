# FitCheck – Angle-Based Yoga Pose Feedback

FitCheck is a deep-learning project that gives **pose feedback from a single yoga image**.  
It uses **MediaPipe Pose** to extract landmarks, computes **joint angles**, classifies the pose using a **PyTorch model**, and then highlights which joints deviate from “typical” form.

Supported poses (currently):

- **Downward Dog** (`downdog`)
- **Goddess** (`goddess`)
- **Plank** (`plank`)
- **Tree** (`tree`)
- **Warrior II** (`warrior2`)

---

## 1. Overview

### Pipeline

1. **Image → Landmarks**  
   Use [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) to detect 33 body joints.

2. **Landmarks → Joint Angles**  
   Compute a set of meaningful angles (hips, knees, shoulders, wrists, neck, etc.) from the landmarks.

3. **Angles → Pose Classification (PyTorch)**  
   Feed the angle vector into a **Residual MLP** to predict one of the 5 yoga poses.

4. **Pose + Angles → Feedback**  
   For the predicted pose, compare each joint angle to the **typical mean angle** (and standard deviation) for that pose.  
   If a joint deviates more than a threshold, generate human-readable feedback such as:

   > “Lower your left hip slightly; it’s higher than the usual alignment.”

5. **UI (Notebook)**  
   A Jupyter notebook that:
   - shows predicted pose + probability
   - overlays the skeleton on the image
   - prints joint-level coaching tips.

---

## 2. Repository Structure
```text
.
├── data/
│   ├── downdog/
|       ├── Images/
|       ├── Landmarks/
│   ├── goddess/
|       ├── Images/
|       ├── Landmarks/
│   ├── plank/
|       ├── Images/
|       ├── Landmarks/
│   ├── tree/
|       ├── Images/
|       ├── Landmarks/
│   └── warrior2/
|       ├── Images/
|       ├── Landmarks/
│
├── angles/
│   ├── downdog.csv
│   ├── goddess.csv
│   ├── plank.csv
│   ├── tree.csv
│   ├── warrior2.csv
│   └── all_angles.csv
│
├── models/
│   ├── yoga_angle_resnet.pt
│   └── yoga_angle_resnet_meta.pkl
│
├── angles.ipynb
├── feedback_image.ipynb
├── landmarks.ipynb
├── train.ipynb
│
├── test/
│   └── test1.jpg
|   └── test2.jpg
|   └── test3.jpg
│
└── README.md
