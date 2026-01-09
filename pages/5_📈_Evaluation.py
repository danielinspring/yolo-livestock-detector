"""
YOLO Model Evaluation Page
Post-training evaluation and visualization using Supervision
"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Evaluation - YOLO Combined Model",
    page_icon="📈",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("# 📈 Model Evaluation")
st.markdown("Evaluate trained models and visualize predictions using Supervision.")
st.markdown("---")

# Check for required packages
try:
    import supervision as sv
    from ultralytics import YOLO
    SUPERVISION_AVAILABLE = True
except ImportError as e:
    SUPERVISION_AVAILABLE = False
    st.error(f"❌ Required packages not installed: {e}")
    st.info("Install with: `pip install supervision ultralytics`")
    st.stop()

# Check for trained model
model_path = Path("models/best.pt")
results_model_path = Path("results/train/combined_model/weights/best.pt")

if model_path.exists():
    active_model_path = model_path
elif results_model_path.exists():
    active_model_path = results_model_path
else:
    active_model_path = None

if active_model_path is None:
    st.warning("⚠️ No trained model found")
    st.info("Train a model first using the 🚀 Training page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 📈 Evaluation")
    st.markdown("""
    Evaluate your trained YOLO model:
    
    **Features:**
    - Sample predictions
    - Confusion matrix
    - Class distribution
    - Performance metrics
    
    **Powered by:**
    - [Supervision](https://github.com/roboflow/supervision)
    - [Ultralytics YOLO](https://docs.ultralytics.com)
    """)
    
    st.markdown("---")
    
    # Model selection
    st.markdown("### Model")
    st.success(f"✓ {active_model_path}")
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    # IOU threshold
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="IoU threshold for NMS"
    )

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))

model = load_model(active_model_path)

# Get class names
class_names = model.names

# Display model info
st.markdown("### 🎯 Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Path", active_model_path.name)
with col2:
    size_mb = active_model_path.stat().st_size / (1024 * 1024)
    st.metric("Model Size", f"{size_mb:.1f} MB")
with col3:
    st.metric("Classes", len(class_names))

# Display class names
with st.expander("📋 Class Names"):
    for idx, name in class_names.items():
        st.markdown(f"- **{idx}**: {name}")

st.markdown("---")

# Main evaluation tabs
tab1, tab2, tab3 = st.tabs(["🔍 Sample Predictions", "📊 Validation Metrics", "🖼️ Batch Evaluation"])

with tab1:
    st.markdown("### Sample Predictions")
    st.markdown("Run inference on sample images from the validation set.")
    
    # Check for validation images
    val_images_dir = Path("data/processed/images/val")
    
    if not val_images_dir.exists():
        st.warning("⚠️ Validation images not found at `data/processed/images/val`")
        st.info("Upload an image to test the model:")
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Run Detection"):
                with st.spinner("Running inference..."):
                    results = model.predict(image, conf=confidence, iou=iou_threshold)
                    result = results[0]
                    
                    # Convert to supervision detections
                    detections = sv.Detections.from_ultralytics(result)
                    
                    # Annotate
                    box_annotator = sv.BoxAnnotator()
                    label_annotator = sv.LabelAnnotator()
                    
                    annotated = np.array(image)
                    annotated = box_annotator.annotate(annotated, detections=detections)
                    
                    # Create labels
                    labels = [
                        f"{class_names[class_id]} {conf:.2f}"
                        for class_id, conf in zip(detections.class_id, detections.confidence)
                    ]
                    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
                    
                    st.image(annotated, caption="Detection Results", use_container_width=True)
                    
                    st.markdown(f"**Detections:** {len(detections)}")
                    for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                        st.markdown(f"- {class_names[class_id]}: {conf:.2%}")
    else:
        # Get list of validation images
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        
        if len(val_images) == 0:
            st.warning("No images found in validation directory")
        else:
            st.success(f"✓ Found {len(val_images)} validation images")
            
            # Number of samples
            n_samples = st.slider("Number of samples", 1, min(10, len(val_images)), 3)
            
            if st.button("🔍 Run Sample Predictions", type="primary"):
                # Random sample
                sample_images = random.sample(val_images, n_samples)
                
                cols = st.columns(min(n_samples, 3))
                
                for idx, img_path in enumerate(sample_images):
                    with cols[idx % 3]:
                        with st.spinner(f"Processing {img_path.name}..."):
                            # Load image
                            image = Image.open(img_path)
                            
                            # Run inference
                            results = model.predict(image, conf=confidence, iou=iou_threshold, verbose=False)
                            result = results[0]
                            
                            # Convert to supervision
                            detections = sv.Detections.from_ultralytics(result)
                            
                            # Annotate
                            box_annotator = sv.BoxAnnotator()
                            label_annotator = sv.LabelAnnotator()
                            
                            annotated = np.array(image)
                            annotated = box_annotator.annotate(annotated, detections=detections)
                            
                            labels = [
                                f"{class_names[class_id]} {conf:.2f}"
                                for class_id, conf in zip(detections.class_id, detections.confidence)
                            ]
                            annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
                            
                            st.image(annotated, caption=img_path.name, use_container_width=True)
                            st.caption(f"Detections: {len(detections)}")

with tab2:
    st.markdown("### Validation Metrics")
    st.markdown("Run full validation on the dataset to compute metrics.")
    
    # Check for training results
    results_dir = Path("results/train/combined_model")
    
    if results_dir.exists():
        st.success("✓ Training results found")
        
        # Check for confusion matrix
        confusion_matrix_path = results_dir / "confusion_matrix.png"
        confusion_matrix_normalized_path = results_dir / "confusion_matrix_normalized.png"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if confusion_matrix_path.exists():
                st.markdown("**Confusion Matrix**")
                st.image(str(confusion_matrix_path), use_container_width=True)
            else:
                st.info("Confusion matrix not found")
        
        with col2:
            if confusion_matrix_normalized_path.exists():
                st.markdown("**Confusion Matrix (Normalized)**")
                st.image(str(confusion_matrix_normalized_path), use_container_width=True)
            else:
                st.info("Normalized confusion matrix not found")
        
        st.markdown("---")
        
        # Other metrics plots
        metrics_files = {
            "F1 Curve": "F1_curve.png",
            "Precision Curve": "P_curve.png",
            "Recall Curve": "R_curve.png",
            "PR Curve": "PR_curve.png",
        }
        
        st.markdown("### Performance Curves")
        cols = st.columns(2)
        
        for idx, (name, filename) in enumerate(metrics_files.items()):
            metric_path = results_dir / filename
            with cols[idx % 2]:
                if metric_path.exists():
                    st.markdown(f"**{name}**")
                    st.image(str(metric_path), use_container_width=True)
                else:
                    st.info(f"{name} not found")
        
        # Results CSV
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            st.markdown("---")
            st.markdown("### Training Metrics History")
            import pandas as pd
            df = pd.read_csv(results_csv)
            # Clean column names
            df.columns = [c.strip() for c in df.columns]
            st.dataframe(df, use_container_width=True)
            
            # Plot key metrics
            if 'metrics/mAP50(B)' in df.columns:
                st.markdown("**mAP50 Progress**")
                st.line_chart(df['metrics/mAP50(B)'])
    else:
        st.info("No training results found. Run training first to see metrics.")
        
        if st.button("🔄 Run Validation"):
            with st.spinner("Running validation..."):
                try:
                    results = model.val(conf=confidence, iou=iou_threshold)
                    
                    st.success("Validation complete!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("mAP50", f"{results.box.map50:.3f}")
                    with col2:
                        st.metric("mAP50-95", f"{results.box.map:.3f}")
                    with col3:
                        st.metric("Precision", f"{results.box.mp:.3f}")
                    with col4:
                        st.metric("Recall", f"{results.box.mr:.3f}")
                except Exception as e:
                    st.error(f"Validation failed: {e}")

with tab3:
    st.markdown("### Batch Evaluation")
    st.markdown("Evaluate the model on a batch of images and view statistics.")
    
    val_images_dir = Path("data/processed/images/val")
    val_labels_dir = Path("data/processed/labels/val")
    
    if not val_images_dir.exists():
        st.warning("Validation images directory not found")
    else:
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        
        if len(val_images) == 0:
            st.warning("No validation images found")
        else:
            n_eval = st.slider("Images to evaluate", 10, min(100, len(val_images)), 20)
            
            if st.button("📊 Run Batch Evaluation", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                all_detections = []
                class_counts = {name: 0 for name in class_names.values()}
                total_detections = 0
                
                sample_images = val_images[:n_eval]
                
                for idx, img_path in enumerate(sample_images):
                    status.text(f"Processing {idx + 1}/{n_eval}: {img_path.name}")
                    progress.progress((idx + 1) / n_eval)
                    
                    # Run inference
                    results = model.predict(img_path, conf=confidence, iou=iou_threshold, verbose=False)
                    result = results[0]
                    
                    detections = sv.Detections.from_ultralytics(result)
                    
                    total_detections += len(detections)
                    for class_id in detections.class_id:
                        class_counts[class_names[class_id]] += 1
                
                status.empty()
                progress.empty()
                
                st.success(f"✓ Evaluated {n_eval} images")
                
                # Display statistics
                st.markdown("### Results Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Detections", total_detections)
                    st.metric("Avg Detections/Image", f"{total_detections/n_eval:.1f}")
                
                with col2:
                    st.markdown("**Class Distribution**")
                    for class_name, count in class_counts.items():
                        st.markdown(f"- **{class_name}**: {count}")
                
                # Bar chart
                import plotly.express as px
                fig = px.bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    labels={'x': 'Class', 'y': 'Count'},
                    title='Detection Class Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
