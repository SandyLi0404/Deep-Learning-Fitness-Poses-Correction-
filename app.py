# app.py - Improved layout with better image sizing

import gradio as gr
import numpy as np
from PIL import Image

print("Starting FitCheck application...")

# Import our utility modules
try:
    from utils.pose_extraction import extract_landmarks
    from utils.angle_calculation import angles_finder
    from utils.model_inference import predict_pose, class_names, feature_cols, angle_stats
    from utils.feedback_generator import generate_feedback
    
    print("✓ All modules loaded successfully")
    print(f"✓ Available poses in angle_stats: {list(angle_stats.keys())}")
except Exception as e:
    print(f"⚠️ Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    raise


def analyze_yoga_pose(image, std_factor=1.0, min_deg=20.0):
    """
    Main pipeline for pose analysis and feedback generation.
    
    Parameters
    ----------
    image : PIL.Image or numpy.ndarray
        Input yoga pose image.
    std_factor : float
        Multiplier on standard deviation for flagging abnormal angles.
    min_deg : float
        Minimum absolute angle difference (in degrees) to flag.
    
    Returns
    -------
    tuple
        (pose_name, confidence, feedback_text, annotated_image)
    """
    try:
        if image is None:
            return (
                "No image uploaded",
                0.0,
                "Please upload an image before analyzing.",
                None
            )

        print(f"Processing image of type: {type(image)}")
        
        # 1. Extract pose landmarks
        landmarks, annotated_image = extract_landmarks(image)

        if landmarks is None:
            return (
                "No pose detected",
                0.0,
                "⚠️ No person detected in the uploaded image. Please try another photo with a clear view of the full body.",
                image if isinstance(image, np.ndarray) else np.array(image)
            )

        print(f"✓ Detected {len(landmarks)} landmarks")
        
        # 2. Calculate joint angles
        angles = angles_finder(landmarks)
        print(f"✓ Calculated {len(angles)} angles")
        
        # 3. Predict pose using trained model
        pose_name, confidence_pct, probabilities = predict_pose(angles)
        print(f"✓ Predicted pose: {pose_name} ({confidence_pct:.1f}%)")
        
        # 4. Generate detailed feedback
        feedback = generate_feedback(
            angle_dict=angles,
            pose_name=pose_name,
            angle_stats=angle_stats,
            feature_cols=feature_cols,
            std_factor=std_factor,
            min_deg=min_deg
        )
        
        # Format pose name nicely
        pose_display = pose_name.title()
        
        return pose_display, round(confidence_pct, 1), feedback, annotated_image

    except Exception as e:
        # Detailed error trace for debugging
        import traceback
        error_msg = f"❌ Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return "Error", 0.0, error_msg, None


# Build the Gradio UI with improved layout
with gr.Blocks() as demo:
    gr.Markdown("# 🧘 FitCheck - Yoga Pose Feedback System")
    gr.Markdown(
        "Upload a yoga pose image to receive instant posture analysis and corrective feedback. "
        "Supported poses: Downdog, Goddess, Plank, Tree, Warrior II"
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", 
                label="📤 Upload Your Yoga Pose",
                height=400
            )
            
            analyze_btn = gr.Button(
                "Analyze Pose 🔍", 
                variant="primary", 
                size="lg",
                scale=1
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.Markdown("**Adjust feedback sensitivity:**")
                std_factor = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Sensitivity (std_factor)",
                    info="Lower = stricter, Higher = more lenient"
                )
                min_deg = gr.Slider(
                    minimum=5.0,
                    maximum=30.0,
                    value=20.0,
                    step=1.0,
                    label="Minimum deviation (degrees)",
                    info="Minimum angle difference to flag"
                )

        with gr.Column(scale=1):
            output_image = gr.Image(
                type="numpy", 
                label="🔍 Detected Landmarks",
                height=400
            )
            
            with gr.Row():
                pose_output = gr.Textbox(
                    label="Predicted Pose", 
                    scale=3,
                    interactive=False
                )
                confidence_output = gr.Number(
                    label="Confidence (%)", 
                    scale=1,
                    interactive=False
                )

    # Feedback section - full width
    gr.Markdown("---")
    feedback_output = gr.Textbox(
        label="📝 Detailed Feedback & Corrections",
        lines=22,
        max_lines=35,
        interactive=False
    )

    # Bind button click to analysis function
    analyze_btn.click(
        fn=analyze_yoga_pose,
        inputs=[input_image, std_factor, min_deg],
        outputs=[pose_output, confidence_output, feedback_output, output_image]
    )

print("Launching Gradio interface...")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )