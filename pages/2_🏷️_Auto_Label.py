"""
Auto-Label Page - Automatically label images using trained model
"""

import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tempfile
import zipfile
import io
import json

st.set_page_config(page_title="Auto-Label", page_icon="🏷️", layout="wide")

st.title("🏷️ Automatic Image Labeling")

st.markdown("""
Upload images to automatically generate YOLO format labels using your trained model.
Perfect for quickly labeling new data!
""")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Model selection
model_path = st.sidebar.text_input(
    "Model Path",
    value="models/best.pt",
    help="Path to YOLO model weights"
)

# Check if model exists
if not Path(model_path).exists():
    st.sidebar.error(f"❌ Model not found: {model_path}")
    st.error("Please train a model first or provide a valid model path")
    st.stop()

st.sidebar.success("✓ Model loaded")

# Detection parameters
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence for auto-labeling"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="IoU threshold for NMS"
)

# Visualization option
show_visualization = st.sidebar.checkbox(
    "Show Visualizations",
    value=True,
    help="Display detection boxes on images"
)

# Class info
st.sidebar.markdown("### Class Mapping")
st.sidebar.code("""
0: ride
1: cowtail
""")

CLASS_NAMES = {0: "ride", 1: "cowtail"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}

# Main content
uploaded_files = st.file_uploader(
    "📤 Upload images to auto-label",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    help="Select multiple images to label at once"
)

if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} image(s) uploaded")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Options")

        output_format = st.radio(
            "Output Format",
            ["YOLO TXT", "JSON", "Both"],
            help="Choose label format"
        )

        include_images = st.checkbox(
            "Include Images",
            value=True,
            help="Include images in download package"
        )

    with col2:
        st.markdown("### Preview")
        st.info(f"""
        **Settings:**
        - Model: {model_path}
        - Confidence: {conf_threshold}
        - IoU: {iou_threshold}
        - Images: {len(uploaded_files)}
        - Output: {output_format}
        """)

    if st.button("🚀 Generate Labels", type="primary", use_container_width=True):
        # Load model
        with st.spinner("Loading model..."):
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Storage for results
        labels_data = {}
        images_data = {}
        stats = {"total_images": len(uploaded_files), "labeled_images": 0, "total_detections": 0, "class_counts": {"ride": 0, "cowtail": 0}}

        # Process each image
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{len(uploaded_files)})")

            # Read image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]

            # Run inference
            results = model.predict(
                img_array,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            # Process detections
            annotations_yolo = []
            annotations_json = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Convert to YOLO format (normalized xywh)
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # YOLO format
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    annotations_yolo.append(yolo_line)

                    # JSON format
                    annotations_json.append({
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": conf,
                        "bbox": {
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height
                        },
                        "bbox_pixels": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    })

                    # Update stats
                    stats["total_detections"] += 1
                    stats["class_counts"][CLASS_NAMES[class_id]] += 1

                stats["labeled_images"] += 1

            # Store labels
            file_stem = Path(uploaded_file.name).stem

            if output_format in ["YOLO TXT", "Both"]:
                labels_data[f"{file_stem}.txt"] = "\n".join(annotations_yolo) if annotations_yolo else ""

            if output_format in ["JSON", "Both"]:
                labels_data[f"{file_stem}.json"] = json.dumps({
                    "image": uploaded_file.name,
                    "width": img_width,
                    "height": img_height,
                    "annotations": annotations_json
                }, indent=2)

            # Store image if needed
            if include_images:
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                images_data[uploaded_file.name] = img_bytes.getvalue()

            # Visualization
            if show_visualization and annotations_yolo:
                viz_img = img_array.copy()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])

                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    cv2.rectangle(viz_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
                    cv2.putText(viz_img, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display in expander
                with st.expander(f"👁️ {uploaded_file.name} - {len(annotations_yolo)} detections"):
                    st.image(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))

        status_text.empty()
        progress_bar.empty()

        # Display statistics
        st.success("✅ Labeling complete!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", stats["total_images"])
        with col2:
            st.metric("Labeled Images", stats["labeled_images"])
        with col3:
            st.metric("Ride Detections", stats["class_counts"]["ride"])
        with col4:
            st.metric("Cowtail Detections", stats["class_counts"]["cowtail"])

        # Create download package
        st.markdown("### 📦 Download Package")

        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add labels
            for filename, content in labels_data.items():
                zip_file.writestr(f"labels/{filename}", content)

            # Add images
            if include_images:
                for filename, content in images_data.items():
                    zip_file.writestr(f"images/{filename}", content)

            # Add stats
            zip_file.writestr("stats.json", json.dumps(stats, indent=2))

            # Add README
            readme = f"""# Auto-Labeling Results

Generated with YOLO Combined Model

## Statistics
- Total Images: {stats['total_images']}
- Labeled Images: {stats['labeled_images']}
- Total Detections: {stats['total_detections']}
- Ride Detections: {stats['class_counts']['ride']}
- Cowtail Detections: {stats['class_counts']['cowtail']}

## Settings
- Model: {model_path}
- Confidence Threshold: {conf_threshold}
- IoU Threshold: {iou_threshold}
- Output Format: {output_format}

## Directory Structure
```
labels/          - YOLO format labels
{'images/         - Original images' if include_images else ''}
stats.json       - Labeling statistics
README.md        - This file
```

## Class Mapping
0: ride
1: cowtail
"""
            zip_file.writestr("README.md", readme)

        # Download button
        st.download_button(
            label="💾 Download Labels Package",
            data=zip_buffer.getvalue(),
            file_name="auto_labels.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

        # Show sample labels
        with st.expander("📝 Sample Labels"):
            sample_files = list(labels_data.keys())[:3]
            for filename in sample_files:
                st.markdown(f"**{filename}**")
                st.code(labels_data[filename][:500] + ("..." if len(labels_data[filename]) > 500 else ""))

else:
    # Instructions
    st.info("👆 Upload images to get started")

    st.markdown("### 📖 How it works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **1. Upload Images**
        - Select multiple images
        - Supports JPG, PNG, BMP
        - Any resolution
        """)

    with col2:
        st.markdown("""
        **2. Configure Settings**
        - Adjust confidence threshold
        - Choose output format
        - Enable visualizations
        """)

    with col3:
        st.markdown("""
        **3. Download Results**
        - YOLO TXT format
        - JSON format
        - Includes statistics
        """)

    st.markdown("### ⚙️ Output Formats")

    tab1, tab2 = st.tabs(["YOLO TXT Format", "JSON Format"])

    with tab1:
        st.markdown("""
        **YOLO TXT Format** (one line per detection):
        ```
        <class_id> <x_center> <y_center> <width> <height>
        ```

        Example:
        ```
        0 0.456789 0.523456 0.234567 0.345678
        1 0.678901 0.234567 0.123456 0.234567
        ```

        All values are normalized (0-1 range)
        """)

    with tab2:
        st.markdown("""
        **JSON Format**:
        ```json
        {
          "image": "example.jpg",
          "width": 640,
          "height": 384,
          "annotations": [
            {
              "class_id": 0,
              "class_name": "ride",
              "confidence": 0.95,
              "bbox": {
                "x_center": 0.456,
                "y_center": 0.523,
                "width": 0.234,
                "height": 0.345
              }
            }
          ]
        }
        ```
        """)

# Footer
st.markdown("---")
st.markdown("### 💡 Tips")
st.markdown("""
- **Quality Control**: Review visualizations to ensure accurate labeling
- **Confidence Threshold**: Lower values label more objects but may include false positives
- **Batch Processing**: Upload multiple images at once for efficiency
- **Re-labeling**: Fine-tune the model and re-run auto-labeling for better results
""")
