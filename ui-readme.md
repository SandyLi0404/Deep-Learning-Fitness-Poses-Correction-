# 🧘 FitCheck – Yoga Pose Feedback UI Guide

This guide explains how to launch and use the FitCheck web interface for yoga pose analysis.

## 1. Setup

Create and activate a virtual environment

```Shell
cd /path/to/Deep-Learning-Fitness-Poses-Correction
python3 -m venv yoga_env
```

macOS/Linux:

`source yoga_env/bin/activate`

Windows:

`yoga_env\Scripts\activate`

Install dependencies
```Shell
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Launch the Web UI

Start the application:

`python app.py`

If successful, the terminal will show:

```
Starting FitCheck application...
Launching Gradio interface...
Running on local URL: http://127.0.0.1:7860
```

Open your browser

Go to:
`http://127.0.0.1:7860`

You should now see the FitCheck interface.

## 3. How to Use the Application

### Step 1 — Upload an Image

Click “Upload Your Yoga Pose”

Choose an image from your computer

Or simply drag & drop an image

### Step 2 — Analyze

Click “Analyze Pose 🔍”

The system will detect landmarks and classify the pose

### Step 3 — View the Results

The interface will display:

Predicted Pose

Confidence (%)

Feedback & Corrections

Image with detected landmarks

## 4. Project Structure (Simplified)

```
Deep-Learning-Fitness-Poses-Correction/
├── app.py                 # Main UI application
├── requirements.txt       # Dependencies
├── utils/                 # Landmark extraction, angles, inference, feedback
├── models/                # Trained model state_dict + metadata
├── angles/                # Per-pose angle statistics
└── test/                  # Sample images
```

## 5. Stopping the Application

To stop the running app:

`Ctrl + C`

To exit the virtual environment:

`deactivate`
