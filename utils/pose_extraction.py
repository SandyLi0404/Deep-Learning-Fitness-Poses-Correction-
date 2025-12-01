import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(image):
    """
    Extract pose landmarks from an input image and return both:
    - a dictionary of landmark coordinates
    - an annotated image with skeleton drawn
    
    Parameters
    ----------
    image : PIL.Image or numpy.ndarray
        The input image containing a person.
    
    Returns
    -------
    landmarks_dict : dict or None
        Dictionary containing normalized landmark coordinates.
        Returns None if no pose landmarks are detected.
    
    annotated_image : numpy.ndarray
        The image with pose landmarks drawn on top.
    """

    # Convert PIL Image to NumPy array if necessary
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=2
    ) as pose:

        # Run pose estimation inference
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # If no person or landmarks detected
        if not results.pose_landmarks:
            return None, image

        # Draw detected landmarks on a copy of the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Convert MediaPipe landmarks into a clean Python dictionary
        landmarks_dict = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks_dict[idx] = {
                "x": landmark.x,                # Normalized x-coordinate
                "y": landmark.y,                # Normalized y-coordinate
                "z": landmark.z,                # Depth (relative)
                "visibility": landmark.visibility  # Detection confidence
            }

        return landmarks_dict, annotated_image
