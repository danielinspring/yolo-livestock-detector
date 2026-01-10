"""
Inference Page - Run YOLO detection on images and videos
"""

import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

st.set_page_config(page_title="Inference", page_icon="📸", layout="wide")

st.title("📸 Object Detection Inference")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Scan for available models
def get_available_models():
    """Scan models directory for all available .pt files"""
    models_dir = Path("models")
    available_models = []
    
    if models_dir.exists():
        # Find all .pt files recursively
        for pt_file in models_dir.rglob("*.pt"):
            available_models.append(str(pt_file))
    
    # Sort by modification time (newest first)
    available_models.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return available_models

available_models = get_available_models()

# Model selection
if available_models:
    model_path = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        format_func=lambda x: f"📦 {x}",
        help="Choose a trained model version"
    )
else:
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/best.pt",
        help="Path to YOLO model weights"
    )

# Check if model exists
if not model_path or not Path(model_path).exists():
    st.sidebar.error(f"❌ Model not found")
    st.error("Please train a model first or provide a valid model path")
    st.info("Train a model using: `python scripts/train.py`")
    st.stop()

st.sidebar.success("✓ Model loaded")

# Display model training info
try:
    from model_info import display_model_info_compact
    display_model_info_compact(model_path)
except ImportError:
    pass

# Detection parameters
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence for detections"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="IoU threshold for Non-Maximum Suppression"
)

# Class colors
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for cowtail
    1: (255, 0, 0),    # Blue for ride
}

CLASS_NAMES = {
    0: "cowtail",
    1: "ride",
}

# Tabs
tab1, tab2 = st.tabs(["📷 Image Inference", "🎥 Video Inference"])

# Image Inference Tab
with tab1:
    st.markdown("### Upload Images for Detection")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="image_upload"
    )

    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} image(s) uploaded")

        if st.button("🚀 Run Detection", key="run_image_detection"):
            # Load model
            with st.spinner("Loading model..."):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()

            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"---")
                st.markdown(f"#### Image {idx + 1}: {uploaded_file.name}")

                col1, col2 = st.columns(2)

                # Read image
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                # Original image
                with col1:
                    st.markdown("**Original Image**")
                    st.image(image, width='stretch')

                # Run inference
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    start_time = time.time()

                    results = model.predict(
                        img_array,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )

                    inference_time = time.time() - start_time

                # Draw results
                result_img = img_array.copy()
                detections = []

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Draw box
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        cv2.rectangle(
                            result_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2
                        )

                        # Draw label
                        label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(
                            result_img,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)),
                            color,
                            -1
                        )
                        cv2.putText(
                            result_img,
                            label,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                        detections.append({
                            "class": CLASS_NAMES[class_id],
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })

                # Detection result
                with col2:
                    st.markdown("**Detection Result**")
                    st.image(result_img, width='stretch')

                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Detections", len(detections))
                with col_m2:
                    ride_count = sum(1 for d in detections if d["class"] == "ride")
                    st.metric("Ride", ride_count)
                with col_m3:
                    cowtail_count = sum(1 for d in detections if d["class"] == "cowtail")
                    st.metric("Cowtail", cowtail_count)

                st.info(f"⏱️ Inference time: {inference_time:.3f}s")

                # Detection details
                if detections:
                    with st.expander("📋 Detection Details"):
                        for i, det in enumerate(detections):
                            st.write(f"{i+1}. **{det['class'].upper()}** - Confidence: {det['confidence']:.2%}")

                # Download button
                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    result_pil.save(tmp_file.name)
                    with open(tmp_file.name, "rb") as f:
                        st.download_button(
                            label="💾 Download Result",
                            data=f,
                            file_name=f"result_{uploaded_file.name}",
                            mime="image/jpeg"
                        )

# Video Inference Tab
with tab2:
    st.markdown("### Upload Video for Detection")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video:
        st.success(f"✓ Video uploaded: {uploaded_video.name}")

        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            video_path = tmp_video.name

        # Display video info
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("FPS", fps)
        with col2:
            st.metric("Frames", total_frames)
        with col3:
            st.metric("Resolution", f"{width}x{height}")
        with col4:
            st.metric("Duration", f"{duration:.1f}s")

        # Processing options
        process_every_n = st.slider(
            "Process every N frames (higher = faster)",
            min_value=1,
            max_value=30,
            value=1,
            help="Process every Nth frame to speed up processing"
        )

        if st.button("🚀 Process Video", key="run_video_detection"):
            # Load model
            with st.spinner("Loading model..."):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.stop()

            # Process video
            cap = cv2.VideoCapture(video_path)

            # Create output video
            output_path = tempfile.mktemp(suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()

            frame_count = 0
            total_detections = {"ride": 0, "cowtail": 0}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process only every Nth frame
                if frame_count % process_every_n == 0:
                    # Run inference
                    results = model.predict(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )

                    # Draw detections
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])

                            color = CLASS_COLORS.get(class_id, (255, 255, 255))
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                            label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            total_detections[CLASS_NAMES[class_id]] += 1

                # Write frame
                out.write(frame)

                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")

                # Display current frame (every 10th frame to reduce overhead)
                if frame_count % 10 == 0:
                    frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                       caption=f"Frame {frame_count}",
                                       width='stretch')

            cap.release()
            out.release()

            st.success("✅ Video processing complete!")

            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Ride Detections", total_detections["ride"])
            with col3:
                st.metric("Cowtail Detections", total_detections["cowtail"])

            # Download processed video
            with open(output_path, "rb") as f:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=f,
                    file_name=f"detected_{uploaded_video.name}",
                    mime="video/mp4"
                )

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.markdown("""
- **Confidence Threshold**: Higher values = fewer but more certain detections
- **IoU Threshold**: Controls how overlapping boxes are merged
- **Video Processing**: Use "Process every N frames" to speed up processing for long videos
""")
