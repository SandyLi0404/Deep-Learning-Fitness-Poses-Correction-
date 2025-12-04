# app.py - Enhanced with Radar Chart Visualization

import gradio as gr
import numpy as np
from PIL import Image
import plotly.graph_objects as go

print("Starting FitCheck application...")

# Import our utility modules
try:
    from utils.pose_extraction import extract_landmarks
    from utils.angle_calculation import angles_finder
    from utils.model_inference import predict_pose, class_names, feature_cols, angle_stats
    from utils.feedback_generator import generate_feedback, normalize_to_180, pretty_name, POSE_ANGLE_WHITELIST
    
    print("✓ All modules loaded successfully")
    print(f"✓ Available poses in angle_stats: {list(angle_stats.keys())}")
except Exception as e:
    print(f"⚠️ Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    raise


def create_radar_chart(angle_dict, pose_name, angle_stats, feature_cols):
    """
    Create a radar chart comparing user's angles to model typical angles.
    
    Parameters
    ----------
    angle_dict : dict
        User's calculated angles (0-360).
    pose_name : str
        Predicted pose name.
    angle_stats : dict
        Model statistics containing mean and std for each pose.
    feature_cols : list
        List of feature names.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Radar chart figure.
    """
    pose_lower = pose_name.lower()
    
    # Get model statistics
    stats = angle_stats.get(pose_name)
    if not stats:
        for key in angle_stats.keys():
            if key.lower() == pose_lower:
                stats = angle_stats[key]
                break
    
    if not stats:
        return None
    
    mean_angles = stats["mean"]
    whitelist = POSE_ANGLE_WHITELIST.get(pose_lower, [])
    
    # Collect data for radar chart
    categories = []
    user_values = []
    model_values = []
    
    for i, name in enumerate(feature_cols):
        if name not in whitelist:
            continue
        
        # Raw values (0-360) for calculation
        raw_user_val = angle_dict.get(name, 0.0)
        raw_model_mean = mean_angles[i]
        
        # Display values (0-180) for visualization
        display_user_val = normalize_to_180(raw_user_val)
        display_model_val = normalize_to_180(raw_model_mean)
        
        categories.append(pretty_name(name))
        user_values.append(display_user_val)
        model_values.append(display_model_val)
    
    if not categories:
        return None
    
    # Create radar chart
    fig = go.Figure()
    
    # Add model typical angles
    fig.add_trace(go.Scatterpolar(
        r=model_values,
        theta=categories,
        fill='toself',
        name='Typical Form',
        line=dict(color='rgb(34, 197, 94)', width=3),  # Green
        fillcolor='rgba(34, 197, 94, 0.15)'
    ))
    
    # Add user's angles
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Form',
        line=dict(color='rgb(59, 130, 246)', width=3),  # Blue
        fillcolor='rgba(59, 130, 246, 0.15)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 180],
                tickmode='linear',
                tick0=0,
                dtick=30,
                showticklabels=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(128, 128, 128, 0.2)'
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        title={
            'text': f"<b>Angle Comparison - {pose_name.title()}</b><br><sub>Comparison between your form (blue) and typical form (green)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=550,
        width=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11)
    )
    
    return fig


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
        (pose_name, confidence, feedback_text, annotated_image, radar_chart)
    """
    try:
        if image is None:
            return (
                "No image uploaded",
                0.0,
                "Please upload an image before analyzing.",
                None,
                None  # No radar chart
            )

        print(f"Processing image of type: {type(image)}")
        
        # 1. Extract pose landmarks
        landmarks, annotated_image = extract_landmarks(image)

        if landmarks is None:
            return (
                "No pose detected",
                0.0,
                "⚠️ No person detected in the uploaded image. Please try another photo with a clear view of the full body.",
                image if isinstance(image, np.ndarray) else np.array(image),
                None  # No radar chart
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
        
        # 5. Create radar chart
        radar_fig = create_radar_chart(
            angle_dict=angles,
            pose_name=pose_name,
            angle_stats=angle_stats,
            feature_cols=feature_cols
        )
        
        # Format pose name nicely
        pose_display = pose_name.title()
        
        return pose_display, round(confidence_pct, 1), feedback, annotated_image, radar_fig

    except Exception as e:
        # Detailed error trace for debugging
        import traceback
        error_msg = f"❌ Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return "Error", 0.0, error_msg, None, None


# Build the Gradio UI with radar chart
with gr.Blocks() as demo:
    gr.Markdown("# 🧘 FitCheck - Yoga Pose Feedback System")
    gr.Markdown(
        "Upload a yoga pose image to receive instant posture analysis, corrective feedback, and visual comparison. "
        "Supported poses: **Downdog, Goddess, Plank, Tree, Warrior II**"
    )
    
    with gr.Row(equal_height=False):
        # Left column - Input
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", 
                label="📤 Upload Your Yoga Pose",
                height=400
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Pose", 
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
        
        # Right column - Results
        with gr.Column(scale=1):
            output_image = gr.Image(
                type="numpy", 
                label="📍 Detected Landmarks",
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
    
    # Full-width sections
    gr.Markdown("---")
    
    with gr.Row():
        # Feedback text
        with gr.Column(scale=1):
            feedback_output = gr.Textbox(
                label=" Detailed Feedback & Corrections",
                lines=20,
                max_lines=30,
                interactive=False
            )
        
        # Radar chart
        with gr.Column(scale=1):
            radar_output = gr.Plot(
                label="📊 Visual Angle Comparison"
            )
    
   

    # Bind button click to analysis function
    analyze_btn.click(
        fn=analyze_yoga_pose,
        inputs=[input_image, std_factor, min_deg],
        outputs=[pose_output, confidence_output, feedback_output, output_image, radar_output]
    )

print("Launching Gradio interface...")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )
