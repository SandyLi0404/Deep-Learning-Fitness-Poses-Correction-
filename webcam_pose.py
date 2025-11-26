import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    # initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # create display window
    window_name = "Workout Pose Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # initialize MediaPipe pose detector
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            # capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            # convert frame to RGB and process with pose detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # draw pose landmarks on frame if detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            cv2.imshow(window_name, frame) # display

            # exit conditions
            key = cv2.waitKey(1) & 0xFF
            # press q or ESC inside the video window
            if key == ord('q') or key == 27:
                break
            # or click X
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
