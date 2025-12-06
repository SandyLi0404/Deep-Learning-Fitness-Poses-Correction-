import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Extract pose landmarks from an input image
def extract_landmarks(image):
  
    # Convert PIL Image to NumPy array if necessary
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        # Convert RGB to BGR for MediaPipe
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run pose estimation
        results = pose.process(image_bgr)

        # If no person or landmarks detected
        if not results.pose_landmarks:
            return None, image

        # Get image dimensions for denormalization
        h, w, _ = image_bgr.shape
        
        # Convert normalized landmarks to pixel coordinates as tuples
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((
                int(lm.x * w),  # x in pixels
                int(lm.y * h),  # y in pixels
                lm.z * w        # z (depth, relative)
            ))

        # Draw detected landmarks on a copy of the image
        annotated_image = image_bgr.copy()
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS
        )
        
        # Convert back to RGB for Gradio display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Return in correct order: landmarks first, then annotated image
        return landmarks, annotated_image