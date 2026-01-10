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

# Scan for available dataset versions
def get_dataset_versions():
    """Scan data/dataset directory for available processed datasets"""
    dataset_dir = Path("data/dataset")
    versions = []

    if dataset_dir.exists():
        # Find all processed_* folders
        for folder in dataset_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("processed_"):
                versions.append(folder)

    # Sort by modification time (newest first)
    versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return versions

# Scan for available model versions
def get_model_versions():
    """Scan models directory for model versions (folders containing .pt files)"""
    models_dir = Path("models")
    versions = {}

    if models_dir.exists():
        # Find all .pt files recursively
        for pt_file in models_dir.rglob("*.pt"):
            # Skip symlinks
            if pt_file.is_symlink():
                continue

            # Get the parent folder as version name
            parent = pt_file.parent
            if parent == models_dir:
                version_name = "root"
            else:
                version_name = str(parent.relative_to(models_dir))

            if version_name not in versions:
                versions[version_name] = []
            versions[version_name].append(pt_file)

    # Also check results directory
    results_model_path = Path("results/train/combined_model/weights/best.pt")
    if results_model_path.exists():
        version_name = "results/train"
        if version_name not in versions:
            versions[version_name] = []
        versions[version_name].append(results_model_path)

    # Sort each version's files (best.pt first, then by timestamp in filename - newest first)
    import re
    def extract_time_from_name(path):
        """Extract time portion from filename like combined_model_155359.pt -> 155359"""
        match = re.search(r'_(\d{6})\.pt$', path.name)
        if match:
            return match.group(1)
        return "000000"  # Default for files without timestamp
    
    for version in versions:
        versions[version].sort(key=lambda x: (
            0 if x.name == "best.pt" else 1,  # best.pt always first
            extract_time_from_name(x)  # Then by time in filename
        ), reverse=False)
        # Re-sort: best.pt first, then others by time descending
        best_files = [f for f in versions[version] if f.name == "best.pt"]
        other_files = [f for f in versions[version] if f.name != "best.pt"]
        other_files.sort(key=lambda x: extract_time_from_name(x), reverse=True)  # Newest first
        versions[version] = best_files + other_files

    return versions

model_versions = get_model_versions()

if not model_versions:
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

    # Sort versions by modification time (newest first)
    sorted_versions = sorted(
        model_versions.keys(),
        key=lambda v: max(f.stat().st_mtime for f in model_versions[v]),
        reverse=True
    )

    selected_version = st.selectbox(
        "Model Version",
        options=sorted_versions,
        format_func=lambda x: f"📁 {x}" if x != "root" else "📁 models/",
        help="Choose a model version folder"
    )

    # Show available weights in that version
    version_files = model_versions[selected_version]
    if len(version_files) > 1:
        active_model_path = Path(st.selectbox(
            "Weight File",
            options=[str(f) for f in version_files],
            format_func=lambda x: f"📦 {Path(x).name}",
            help="Choose weight file (best.pt recommended)"
        ))
    else:
        active_model_path = version_files[0]
        st.info(f"📦 {version_files[0].name}")
    
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

    st.markdown("---")

    # Dataset selection
    st.markdown("### Dataset")
    
    dataset_versions = get_dataset_versions()
    
    if dataset_versions:
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=dataset_versions,
            format_func=lambda x: f"📁 {x.name}",
            help="Choose a processed dataset for evaluation"
        )
        st.caption(f"Path: `{selected_dataset}`")
    else:
        st.warning("⚠️ No processed datasets found")
        selected_dataset = None

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
    val_images_dir = selected_dataset / "images" / "val" if selected_dataset else Path("data/processed/images/val")
    
    if not val_images_dir or not val_images_dir.exists():
        st.warning(f"⚠️ Validation images not found at `{val_images_dir}`")
        st.info("Upload an image to test the model:")
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
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
                    
                    st.image(annotated, caption="Detection Results", width='stretch')
                    
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
                            
                            st.image(annotated, caption=img_path.name, width='stretch')
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
                st.image(str(confusion_matrix_path), width='stretch')
            else:
                st.info("Confusion matrix not found")
        
        with col2:
            if confusion_matrix_normalized_path.exists():
                st.markdown("**Confusion Matrix (Normalized)**")
                st.image(str(confusion_matrix_normalized_path), width='stretch')
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
                    st.image(str(metric_path), width='stretch')
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
            st.dataframe(df, width='stretch')
            
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
    st.markdown("Evaluate the model on validation images with **Ground Truth comparison**.")
    
    val_images_dir = selected_dataset / "images" / "val" if selected_dataset else Path("data/processed/images/val")
    val_labels_dir = selected_dataset / "labels" / "val" if selected_dataset else Path("data/processed/labels/val")
    
    # Helper functions for evaluation
    def parse_yolo_label(label_path, img_width, img_height):
        """Parse YOLO format label file and return bounding boxes in xyxy format."""
        boxes = []
        class_ids = []
        
        if not label_path.exists():
            return np.array([]), np.array([])
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(class_id)
        
        return np.array(boxes), np.array(class_ids)
    
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_predictions_to_gt(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh=0.5):
        """Match predictions to ground truth using IoU threshold."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if pred_classes[i] == gt_classes[j]:  # Same class requirement
                    iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
        
        # Greedy matching
        matched_pairs = []
        matched_preds = set()
        matched_gts = set()
        
        while True:
            max_iou = iou_matrix.max()
            if max_iou < iou_thresh:
                break
            
            pred_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_pairs.append((pred_idx, gt_idx, max_iou))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
            
            # Zero out matched rows/columns
            iou_matrix[pred_idx, :] = 0
            iou_matrix[:, gt_idx] = 0
        
        unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
        unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
        
        return matched_pairs, unmatched_preds, unmatched_gts
    
    if not val_images_dir.exists():
        st.warning("Validation images directory not found")
    elif not val_labels_dir.exists():
        st.warning(f"⚠️ Validation labels directory not found at `{val_labels_dir}`")
        st.info("Ground Truth labels are required for proper evaluation.")
    else:
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        
        if len(val_images) == 0:
            st.warning("No validation images found")
        else:
            st.success(f"✓ Found {len(val_images)} validation images with labels")
            
            col1, col2 = st.columns(2)
            with col1:
                n_eval = st.slider("Images to evaluate", 10, min(100, len(val_images)), min(30, len(val_images)))
            with col2:
                eval_iou_thresh = st.slider("IoU Threshold for matching", 0.3, 0.9, 0.5, 0.05,
                                            help="IoU threshold to consider a prediction as True Positive")
            
            show_visual = st.checkbox("Show visual comparison (slower)", value=False)
            
            if st.button("📊 Run Batch Evaluation", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                
                # Metrics tracking
                total_tp = 0
                total_fp = 0
                total_fn = 0
                
                # Per-class metrics
                class_metrics = {name: {"tp": 0, "fp": 0, "fn": 0} for name in class_names.values()}
                
                # For confidence-based metrics
                all_confidences = []
                all_matches = []  # True if TP, False if FP
                
                sample_images = val_images[:n_eval]
                comparison_images = []
                
                for idx, img_path in enumerate(sample_images):
                    status.text(f"Processing {idx + 1}/{n_eval}: {img_path.name}")
                    progress.progress((idx + 1) / n_eval)
                    
                    # Load image to get dimensions
                    image = Image.open(img_path)
                    img_width, img_height = image.size
                    
                    # Load GT labels
                    label_path = val_labels_dir / (img_path.stem + ".txt")
                    gt_boxes, gt_classes = parse_yolo_label(label_path, img_width, img_height)
                    
                    # Run inference
                    results = model.predict(img_path, conf=confidence, iou=iou_threshold, verbose=False)
                    result = results[0]
                    detections = sv.Detections.from_ultralytics(result)
                    
                    pred_boxes = detections.xyxy if len(detections) > 0 else np.array([])
                    pred_classes = detections.class_id if len(detections) > 0 else np.array([])
                    pred_confs = detections.confidence if len(detections) > 0 else np.array([])
                    
                    # Match predictions to GT
                    matched, unmatched_preds, unmatched_gts = match_predictions_to_gt(
                        pred_boxes, pred_classes, gt_boxes, gt_classes, eval_iou_thresh
                    )
                    
                    # Count TP, FP, FN
                    tp = len(matched)
                    fp = len(unmatched_preds)
                    fn = len(unmatched_gts)
                    
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    
                    # Track confidences for AP calculation
                    for i, conf_score in enumerate(pred_confs):
                        all_confidences.append(conf_score)
                        all_matches.append(i not in unmatched_preds)
                    
                    # Per-class metrics
                    for pred_idx, gt_idx, _ in matched:
                        class_name = class_names[pred_classes[pred_idx]]
                        class_metrics[class_name]["tp"] += 1
                    
                    for pred_idx in unmatched_preds:
                        class_name = class_names[pred_classes[pred_idx]]
                        class_metrics[class_name]["fp"] += 1
                    
                    for gt_idx in unmatched_gts:
                        class_name = class_names[gt_classes[gt_idx]]
                        class_metrics[class_name]["fn"] += 1
                    
                    # Visual comparison (first 3 images only)
                    if show_visual and len(comparison_images) < 3:
                        annotated = np.array(image)
                        
                        # Draw GT boxes in green
                        for box in gt_boxes:
                            x1, y1, x2, y2 = map(int, box)
                            import cv2
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw predictions in red (FP) or blue (TP)
                        for i, (box, class_id, conf_score) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
                            x1, y1, x2, y2 = map(int, box)
                            color = (0, 0, 255) if i in unmatched_preds else (255, 0, 0)  # Red=FP, Blue=TP
                            label = "FP" if i in unmatched_preds else "TP"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, f"{label} {conf_score:.2f}", (x1, y2+15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        comparison_images.append((img_path.name, annotated, tp, fp, fn))
                
                status.empty()
                progress.empty()
                
                st.success(f"✓ Evaluated {n_eval} images")
                
                # Calculate overall metrics
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Calculate AP@50 (simplified)
                if len(all_confidences) > 0:
                    sorted_indices = np.argsort(all_confidences)[::-1]
                    sorted_matches = np.array(all_matches)[sorted_indices]
                    
                    cumsum = np.cumsum(sorted_matches)
                    precisions_at_k = cumsum / np.arange(1, len(sorted_matches) + 1)
                    recalls_at_k = cumsum / (total_tp + total_fn) if (total_tp + total_fn) > 0 else np.zeros_like(cumsum)
                    
                    # AP using all-point interpolation
                    ap = 0
                    for i in range(len(sorted_matches)):
                        if sorted_matches[i]:
                            ap += precisions_at_k[i]
                    ap = ap / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                else:
                    ap = 0
                
                # Display Overall Metrics
                st.markdown("---")
                st.markdown("### 📊 Overall Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", f"{precision:.1%}", help="TP / (TP + FP)")
                with col2:
                    st.metric("Recall", f"{recall:.1%}", help="TP / (TP + FN)")
                with col3:
                    st.metric("F1 Score", f"{f1_score:.1%}", help="Harmonic mean of Precision and Recall")
                with col4:
                    st.metric("AP@50", f"{ap:.1%}", help="Average Precision at IoU=0.5")
                
                # TP/FP/FN breakdown
                st.markdown("---")
                st.markdown("### 🎯 Detection Breakdown")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ True Positives", total_tp, help="Correct detections")
                with col2:
                    st.metric("❌ False Positives", total_fp, help="Wrong detections (no matching GT)")
                with col3:
                    st.metric("⚠️ False Negatives", total_fn, help="Missed detections (GT not detected)")
                
                # Per-class performance table
                st.markdown("---")
                st.markdown("### 📋 Per-Class Performance")
                
                import pandas as pd
                class_data = []
                for class_name, metrics in class_metrics.items():
                    tp = metrics["tp"]
                    fp = metrics["fp"]
                    fn = metrics["fn"]
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    class_data.append({
                        "Class": class_name,
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "Precision": f"{prec:.1%}",
                        "Recall": f"{rec:.1%}",
                        "F1": f"{f1:.1%}"
                    })
                
                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Visual comparison
                if show_visual and comparison_images:
                    st.markdown("---")
                    st.markdown("### 🖼️ Visual Comparison")
                    st.caption("🟢 Green = Ground Truth | 🔵 Blue = True Positive | 🔴 Red = False Positive")
                    
                    cols = st.columns(min(3, len(comparison_images)))
                    for i, (name, img, tp, fp, fn) in enumerate(comparison_images):
                        with cols[i]:
                            st.image(img, caption=f"{name}\nTP:{tp} FP:{fp} FN:{fn}", use_container_width=True)
                
                # Class distribution chart
                st.markdown("---")
                st.markdown("### 📈 Detection Distribution")
                
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[
                    go.Bar(name='True Positive', x=list(class_metrics.keys()), 
                           y=[m["tp"] for m in class_metrics.values()], marker_color='#28a745'),
                    go.Bar(name='False Positive', x=list(class_metrics.keys()), 
                           y=[m["fp"] for m in class_metrics.values()], marker_color='#dc3545'),
                    go.Bar(name='False Negative', x=list(class_metrics.keys()), 
                           y=[m["fn"] for m in class_metrics.values()], marker_color='#ffc107')
                ])
                fig.update_layout(barmode='group', title='Detection Results by Class',
                                  xaxis_title='Class', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
