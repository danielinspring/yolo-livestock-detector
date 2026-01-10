"""
Dataset Info Page - View dataset statistics and visualizations
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json
import subprocess

st.set_page_config(page_title="Dataset Info", page_icon="📊", layout="wide")

st.title("📊 Dataset Information & Statistics")

# Class names
CLASS_NAMES = {0: "cowtail", 1: "ride"}


def get_dataset_versions():
    """Scan data/dataset directory for available processed datasets"""
    dataset_dir = Path("data/dataset")
    versions = []

    if dataset_dir.exists():
        # Find all processed_* folders
        for folder in dataset_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("processed_"):
                # Check if it has the expected structure (images or labels folder)
                if (folder / "images").exists() or (folder / "labels").exists():
                    versions.append(folder)

    # Sort by modification time (newest first)
    versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return versions

def analyze_dataset(data_dir):
    """Analyze dataset and return statistics"""
    data_dir = Path(data_dir)

    stats = {
        "splits": {},
        "class_distribution": defaultdict(lambda: {"ride": 0, "cowtail": 0}),
        "total_images": 0,
        "total_annotations": 0,
        "images_per_split": {},
        "annotations_per_split": {},
    }

    # Check each split
    for split in ["train", "val", "test"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            continue

        # Count images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        num_images = len(image_files)
        stats["images_per_split"][split] = num_images
        stats["total_images"] += num_images

        # Analyze labels
        label_files = list(labels_dir.glob("*.txt"))
        num_annotations = 0

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        class_id = int(float(line.split()[0]))
                        class_name = CLASS_NAMES.get(class_id, "unknown")
                        stats["class_distribution"][split][class_name] += 1
                        num_annotations += 1

        stats["annotations_per_split"][split] = num_annotations
        stats["total_annotations"] += num_annotations

        stats["splits"][split] = {
            "images": num_images,
            "annotations": num_annotations,
            "avg_annotations_per_image": num_annotations / num_images if num_images > 0 else 0
        }

    return stats


def analyze_background_images(data_dir):
    """Analyze background vs labeled images in the dataset"""
    data_dir = Path(data_dir)
    
    stats = {
        "labeled": 0,      # Images with at least one annotation
        "background": 0,   # Images with no annotations (empty label files)
        "per_split": {}
    }
    
    # Check each split
    for split in ["train", "val", "test"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split
        
        if not images_dir.exists():
            continue
        
        split_stats = {"labeled": 0, "background": 0}
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + \
                      list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.PNG"))
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                # Check if label file has any content
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        split_stats["labeled"] += 1
                        stats["labeled"] += 1
                    else:
                        split_stats["background"] += 1
                        stats["background"] += 1
            else:
                # No label file = background image
                split_stats["background"] += 1
                stats["background"] += 1
        
        stats["per_split"][split] = split_stats
    
    return stats


# Get available datasets
dataset_versions = get_dataset_versions()

if not dataset_versions:
    st.warning("⚠️ No processed dataset found in `data/dataset/`")
    st.info("Run preprocessing first: `python scripts/preprocess_data.py`")
    st.stop()

# Sidebar for dataset selection
with st.sidebar:
    st.markdown("## 📊 Dataset Selection")

    selected_dataset = st.selectbox(
        "Select Dataset",
        options=dataset_versions,
        format_func=lambda x: f"📁 {x.name}",
        help="Choose a processed dataset folder"
    )

    st.markdown("---")
    st.markdown(f"**Path:** `{selected_dataset}`")
    
    # Archive button
    st.markdown("---")
    if st.button("📦 Archive Dataset", type="secondary", use_container_width=True):
        import shutil
        archived_dir = Path("data/archived")
        archived_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = archived_dir / selected_dataset.name
        
        if dest_path.exists():
            st.error(f"❌ Archive already exists: `{dest_path}`")
        else:
            try:
                shutil.move(str(selected_dataset), str(dest_path))
                st.success(f"✅ Archived to `{dest_path}`")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Archive failed: {e}")

processed_dir = selected_dataset

# Check if data is split
train_dir = processed_dir / "images" / "train"

if not train_dir.exists():
    st.warning("⚠️ Dataset not split yet")
    st.info("This dataset needs to be split into train/val/test sets before viewing statistics.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Split Options")
        train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7, 0.05)
        val_ratio = st.slider("Validation Ratio", 0.05, 0.3, 0.2, 0.05)
        test_ratio = round(1.0 - train_ratio - val_ratio, 2)
        st.info(f"Test Ratio: {test_ratio:.0%}")

        if test_ratio < 0:
            st.error("Train + Val ratios exceed 100%. Please adjust.")
        else:
            if st.button("🔀 Split Dataset", type="primary", use_container_width=True):
                with st.spinner("Splitting dataset..."):
                    cmd = [
                        "python", "scripts/split_data.py",
                        "--data-dir", str(processed_dir),
                        "--train-ratio", str(train_ratio),
                        "--val-ratio", str(val_ratio),
                        "--test-ratio", str(test_ratio)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        st.success("✅ Dataset split completed!")
                        st.code(result.stdout)
                        st.rerun()
                    else:
                        st.error("❌ Split failed!")
                        st.code(result.stderr)
    st.stop()

# Analyze dataset
with st.spinner("Analyzing dataset..."):
    stats = analyze_dataset(processed_dir)

if stats["total_images"] == 0:
    st.error("No images found in dataset")
    st.stop()

# Overview metrics
st.markdown("## 📈 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Images", stats["total_images"])
with col2:
    st.metric("Total Annotations", stats["total_annotations"])
with col3:
    total_ride = sum(stats["class_distribution"][split]["ride"] for split in stats["splits"])
    st.metric("Ride Detections", total_ride)
with col4:
    total_cowtail = sum(stats["class_distribution"][split]["cowtail"] for split in stats["splits"])
    st.metric("Cowtail Detections", total_cowtail)

st.markdown("---")

# Background Images Analysis
st.markdown("## 🖼️ Background vs Labeled Images")

with st.spinner("Analyzing background images..."):
    bg_stats = analyze_background_images(processed_dir)

total_images_bg = bg_stats["labeled"] + bg_stats["background"]

if total_images_bg > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratio distribution pie chart
        st.markdown("### Distribution")
        
        fig_bg = go.Figure(data=[go.Pie(
            labels=["Labeled Images", "Background Images"],
            values=[bg_stats["labeled"], bg_stats["background"]],
            hole=0.3,
            marker_colors=['#3498db', '#95a5a6']
        )])
        
        fig_bg.update_layout(
            showlegend=True,
            height=350
        )
        
        st.plotly_chart(fig_bg, use_container_width=True)
    
    with col2:
        # Statistics
        st.markdown("### Statistics")
        
        bg_ratio = bg_stats["background"] / bg_stats["labeled"] * 100 if bg_stats["labeled"] > 0 else 0
        
        st.metric("Labeled Images", bg_stats["labeled"])
        st.metric("Background Images", bg_stats["background"])
        st.metric("Background Ratio", f"{bg_ratio:.1f}%")
        
        st.markdown("**Per Split Breakdown:**")
        for split, split_stats in bg_stats["per_split"].items():
            total_split = split_stats["labeled"] + split_stats["background"]
            bg_pct = split_stats["background"] / total_split * 100 if total_split > 0 else 0
            st.markdown(f"- **{split.capitalize()}**: {split_stats['background']}/{total_split} background ({bg_pct:.1f}%)")
else:
    st.info("No image data available")

st.markdown("---")

# Split information
st.markdown("## 📁 Dataset Splits")

# Create DataFrame for splits
split_data = []
for split in ["train", "val", "test"]:
    if split in stats["splits"]:
        split_info = stats["splits"][split]
        split_data.append({
            "Split": split.capitalize(),
            "Images": split_info["images"],
            "Annotations": split_info["annotations"],
            "Avg per Image": f"{split_info['avg_annotations_per_image']:.2f}",
            "Percentage": f"{split_info['images'] / stats['total_images'] * 100:.1f}%"
        })

df_splits = pd.DataFrame(split_data)
st.dataframe(df_splits, use_container_width=True, hide_index=True)

# Visualizations
st.markdown("---")
st.markdown("## 📊 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Images per split pie chart
    st.markdown("### Images Distribution")

    fig_images = go.Figure(data=[go.Pie(
        labels=[s.capitalize() for s in stats["images_per_split"].keys()],
        values=list(stats["images_per_split"].values()),
        hole=0.3,
        marker_colors=['#2ecc71', '#3498db', '#e74c3c']
    )])

    fig_images.update_layout(
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_images, use_container_width=True)

with col2:
    # Annotations per split pie chart
    st.markdown("### Annotations Distribution")

    fig_annotations = go.Figure(data=[go.Pie(
        labels=[s.capitalize() for s in stats["annotations_per_split"].keys()],
        values=list(stats["annotations_per_split"].values()),
        hole=0.3,
        marker_colors=['#2ecc71', '#3498db', '#e74c3c']
    )])

    fig_annotations.update_layout(
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_annotations, use_container_width=True)

# Class distribution (including background images)
st.markdown("### Class Distribution per Split")

class_dist_data = []
for split in stats["splits"].keys():
    class_dist_data.append({
        "Split": split.capitalize(),
        "Class": "Ride",
        "Count": stats["class_distribution"][split]["ride"]
    })
    class_dist_data.append({
        "Split": split.capitalize(),
        "Class": "Cowtail",
        "Count": stats["class_distribution"][split]["cowtail"]
    })
    # Add background images count
    if split in bg_stats["per_split"]:
        class_dist_data.append({
            "Split": split.capitalize(),
            "Class": "Background",
            "Count": bg_stats["per_split"][split]["background"]
        })

df_class_dist = pd.DataFrame(class_dist_data)

fig_class = px.bar(
    df_class_dist,
    x="Split",
    y="Count",
    color="Class",
    barmode="group",
    color_discrete_map={"Ride": "#2ecc71", "Cowtail": "#e74c3c", "Background": "#95a5a6"},
    height=400
)

fig_class.update_layout(
    xaxis_title="Dataset Split",
    yaxis_title="Count",
    legend_title="Type"
)

st.plotly_chart(fig_class, use_container_width=True)

# Overall class balance
st.markdown("### Overall Class Balance")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    total_ride = sum(stats["class_distribution"][split]["ride"] for split in stats["splits"])
    total_cowtail = sum(stats["class_distribution"][split]["cowtail"] for split in stats["splits"])

    fig_balance = go.Figure(data=[go.Pie(
        labels=["Ride", "Cowtail"],
        values=[total_ride, total_cowtail],
        hole=0.4,
        marker_colors=['#2ecc71', '#e74c3c']
    )])

    fig_balance.update_layout(
        showlegend=True,
        height=400
    )

    st.plotly_chart(fig_balance, use_container_width=True)

    # Balance ratio
    ratio = total_ride / total_cowtail if total_cowtail > 0 else 0
    if abs(ratio - 1.0) < 0.2:
        balance_status = "✅ Well balanced"
        balance_color = "green"
    elif abs(ratio - 1.0) < 0.5:
        balance_status = "⚠️ Slightly imbalanced"
        balance_color = "orange"
    else:
        balance_status = "❌ Imbalanced"
        balance_color = "red"

    st.markdown(f"**Balance Status:** :{balance_color}[{balance_status}]")
    st.markdown(f"**Ride:Cowtail Ratio:** {ratio:.2f}:1")

# Detailed statistics
st.markdown("---")
st.markdown("## 📋 Detailed Statistics")

with st.expander("📊 View Detailed Stats"):
    for split in stats["splits"].keys():
        st.markdown(f"### {split.capitalize()} Set")

        split_stats = stats["splits"][split]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Images", split_stats["images"])
        with col2:
            st.metric("Annotations", split_stats["annotations"])
        with col3:
            st.metric("Avg per Image", f"{split_stats['avg_annotations_per_image']:.2f}")

        # Class breakdown
        ride_count = stats["class_distribution"][split]["ride"]
        cowtail_count = stats["class_distribution"][split]["cowtail"]

        st.markdown("**Class Distribution:**")
        st.markdown(f"- Ride: {ride_count} ({ride_count / split_stats['annotations'] * 100:.1f}%)")
        st.markdown(f"- Cowtail: {cowtail_count} ({cowtail_count / split_stats['annotations'] * 100:.1f}%)")

        st.markdown("---")

# Export statistics
st.markdown("## 💾 Export Statistics")

if st.button("📥 Download Statistics as JSON"):
    stats_json = {
        "dataset_path": str(processed_dir),
        "total_images": stats["total_images"],
        "total_annotations": stats["total_annotations"],
        "splits": {},
        "class_distribution": {}
    }

    for split in stats["splits"].keys():
        stats_json["splits"][split] = stats["splits"][split]
        stats_json["class_distribution"][split] = dict(stats["class_distribution"][split])

    st.download_button(
        label="💾 Download JSON",
        data=json.dumps(stats_json, indent=2),
        file_name="dataset_statistics.json",
        mime="application/json"
    )

# Dataset health check
st.markdown("---")
st.markdown("## 🏥 Dataset Health Check")

health_checks = []

# Check 1: Minimum images
min_images = 100
if stats["total_images"] >= min_images:
    health_checks.append(("✅", f"Sufficient images ({stats['total_images']} >= {min_images})"))
else:
    health_checks.append(("❌", f"Too few images ({stats['total_images']} < {min_images})"))

# Check 2: Class balance
total_ride = sum(stats["class_distribution"][split]["ride"] for split in stats["splits"])
total_cowtail = sum(stats["class_distribution"][split]["cowtail"] for split in stats["splits"])
ratio = total_ride / total_cowtail if total_cowtail > 0 else 0

if 0.5 <= ratio <= 2.0:
    health_checks.append(("✅", f"Classes are balanced (ratio: {ratio:.2f})"))
else:
    health_checks.append(("⚠️", f"Classes are imbalanced (ratio: {ratio:.2f})"))

# Check 3: Train set size
if "train" in stats["splits"]:
    train_pct = stats["splits"]["train"]["images"] / stats["total_images"] * 100
    if train_pct >= 60:
        health_checks.append(("✅", f"Training set is adequate ({train_pct:.1f}%)"))
    else:
        health_checks.append(("⚠️", f"Training set is small ({train_pct:.1f}%)"))

# Check 4: Annotations per image
avg_annotations = stats["total_annotations"] / stats["total_images"]
if avg_annotations >= 1.0:
    health_checks.append(("✅", f"Good annotation density ({avg_annotations:.2f} per image)"))
else:
    health_checks.append(("⚠️", f"Low annotation density ({avg_annotations:.2f} per image)"))

# Display health checks
for icon, message in health_checks:
    st.markdown(f"{icon} {message}")

# Recommendations
st.markdown("---")
st.markdown("## 💡 Recommendations")

recommendations = []

if stats["total_images"] < 500:
    recommendations.append("Consider collecting more images for better model performance")

if ratio < 0.5 or ratio > 2.0:
    recommendations.append("Address class imbalance by collecting more samples of the minority class or using data augmentation")

if avg_annotations < 1.5:
    recommendations.append("Some images have few annotations. Verify labeling quality")

if "test" in stats["splits"] and stats["splits"]["test"]["images"] < 50:
    recommendations.append("Test set is small. Consider increasing test set size for more reliable evaluation")

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
else:
    st.success("✅ Dataset looks good! No major issues detected.")

# Footer
st.markdown("---")
st.caption(f"Dataset path: {processed_dir.absolute()}")
