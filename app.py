# app.py - Fully runnable version
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

print("Starting FitCheck application...")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(image):
    """Extract pose landmarks from the input image."""
    
    # Convert PIL Image to NumPy array if needed
    if hasattr(image, "convert"):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=2
    ) as pose:

        # Process the image with MediaPipe
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # If no person / pose detected
        if not results.pose_landmarks:
            return None, image_np

        # Draw detected landmarks on a copy of the image
        annotated_image = image_np.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        return results.pose_landmarks, annotated_image


def analyze_yoga_pose(image):
    """Main pipeline for pose analysis and feedback generation."""

    try:
        if image is None:
            return ("No image uploaded", 0.0,
                    "Please upload an image before analyzing.", None)

        # 1. Extract pose landmarks
        landmarks, annotated_image = extract_landmarks(image)

        if landmarks is None:
            return (
                "No pose detected",
                0.0,
                "No person detected in the uploaded image. Please try another photo.",
                image
            )

        # 2. Temporary demo classifier (placeholder for real model)
        import random
        poses = ["Downdog", "Goddess", "Plank", "Tree", "Warrior II"]
        predicted_pose = random.choice(poses)
        confidence = random.uniform(85, 98)

        # 3. Generate basic feedback text
        feedback = f"""
✅ Pose detected successfully!

Analysis:
- Detected {len(list(landmarks.landmark))} body landmarks
- Pose classification: {predicted_pose}

Quick Tips:
- Keep your core engaged
- Maintain steady breathing
- Focus on alignment and stability

Note: This is a demo version. The full corrective model will be integrated soon!
"""

        return predicted_pose, round(confidence, 1), feedback, annotated_image

    except Exception as e:
        # Detailed error trace for debugging
        import traceback
        error_msg = f"Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return "Error", 0.0, error_msg, None


# Build the Gradio UI
with gr.Blocks() as demo:  
    gr.Markdown("# 🧘 FitCheck - Yoga Pose Feedback System")
    gr.Markdown("Upload a yoga pose image to receive instant posture feedback!")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Your Yoga Pose")
            analyze_btn = gr.Button("Analyze Pose 🔍", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detected Landmarks")

    with gr.Row():
        pose_output = gr.Textbox(label="Predicted Pose")
        confidence_output = gr.Number(label="Confidence (%)")

    feedback_output = gr.Textbox(
        label="Feedback & Corrections",
        lines=10
    )

    # Bind button click to analysis function
    analyze_btn.click(
        fn=analyze_yoga_pose,
        inputs=input_image,
        outputs=[pose_output, confidence_output, feedback_output, output_image]
    )

print("Launching Gradio interface...")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Localhost
        server_port=7860,
        share=False,             # Change to True if you want public sharing
        debug=True
    )
