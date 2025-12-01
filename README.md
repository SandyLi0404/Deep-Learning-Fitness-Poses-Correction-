# FitCheck – Angle-Based Yoga Pose Feedback

FitCheck is a deep-learning computer vision project that gives **yoga pose feedback from image input**.  

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

5. **An interactive website that:**
   - Takes an input image from upload
   - Predict pose + confidence in prediction
   - Identify joints in the image
   - Provides corrections to the pose joint angles

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
```
**Key artifacts**:
- `angles/all_angles.csv` – merged angle features + labels for all poses.
- `models/yoga_angle_resnet.pt` – trained PyTorch model.
- `models/yoga_angle_resnet_meta.pkl` – metadata: class names, feature names, angle stats.

## 3. Data
- Source: [Yoga Pose Classification](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) from Kaggle.
- Poses used: `downdog`, `goddess`, `plank`, `tree`, `warrior2`.

For each image:
1. Run MediaPipe Pose to get 33 landmarks.
2. Compute a fixed set of joint angles, including examples like:
   - `left_elbow_angle`, `right_elbow_angle`
   - `left_shoulder_angle`, `right_shoulder_angle`
   - `left_hip_angle`, `right_hip_angle`
   - `left_knee_angle`, `right_knee_angle`
   - `left_wrist_angle_bk`, `right_wrist_angle_bk`
   - `neck_angle_uk`
3. Save each pose’s angles to a CSV:
   ```text
   image_path, pose_label, left_elbow_angle, right_elbow_angle, ...
   ```
4. Concatenate all per-pose CSVs into Results/all_angles.csv.

## 4. Installation
```bash
# (optional) create a conda env
conda create -n fitcheck python=3.10
conda activate fitcheck

# core dependencies
pip install numpy pandas matplotlib scikit-learn opencv-python mediapipe

# install PyTorch
pip install torch torchvision torchaudio
```

## 5. Angle Extraction
Notebook: `angles.ipynb`

1. Run MediaPipe Pose on each image in `data/<pose>/Images/`.
2. From the 33 landmarks, compute a fixed set of angles (hips, knees, shoulders, wrists, neck, etc.).
3. Save one CSV per pose in `angles/` and then merge them into:
   ```text
   angles/all_angles.csv
   ```

This merged file is the input to the training step.

## 6. Training the Pose Classifier

Notebook: `train.ipynb`
1. Load `angles/all_angles.csv`.
2. Use all angle columns as features and `pose_label` as the target.
3. Split into train / validation / test (e.g. 70 / 15 / 15) with stratification.
4. Train a **Residual MLP (AngleResNet)** in PyTorch:
   - several fully connected layers with BatchNorm, ReLU, Dropout, and skip connections
   - optimized with Adam + CrossEntropyLoss
   - early stopping on validation accuracy
5. Evaluate on the test set and save:
   - `models/yoga_angle_resnet.pt` (weights)
   - `models/yoga_angle_resnet_meta.pkl` (class names, feature names, angle means/stds).

## 7. Single-Image Pose Feedback
Notebook: `feedback_image.ipynb`

Given a new image:
1. Run MediaPipe Pose → get landmarks → compute angles.
2. Build a feature vector in the same order as `feature_cols`.
3. Use the trained model to predict the pose.
4. Compare each **pose-relevant** angle to its mean and std (from `angle_stats`), using a circular difference on angles.
5. If a joint deviates beyond a threshold, generate a short coaching message
(e.g. “Lower your left hip slightly; it’s higher than the usual alignment”)
and display it along with the image and pose skeleton.

## 8. Possible Future Work
- Extend to more poses and other exercise types (e.g. squats, lunges, push-ups).
- Add a webcam / live video mode for real-time feedback.
- Wrap the system in a simple web UI (Streamlit or Gradio).
- Explore 2D skeleton models such as GCNs or Transformers on landmark sequences.
- Calibrate thresholds against a curated set of expert-labeled “ideal” poses.

## 9. Acknowledgements
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) for fast, robust pose estimation.
- [Yoga Pose Classification](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) for training and evaluation data.
- Prior angle-based yoga pose projects (e.g. [Manoj-2702
Yoga_Poses-Dataset](https://github.com/Manoj-2702/Yoga_Poses-Dataset)) for inspiration on angle engineering and feedback ideas.