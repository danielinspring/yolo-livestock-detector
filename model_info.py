"""
Model Info Utility
Provides functions to load and display model training metadata
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st


def get_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training info JSON for a given model file.
    
    Looks for a corresponding *_info.json file in the same directory.
    For example:
        - models/2026-01-11/combined_model_143052.pt
        - models/2026-01-11/combined_model_143052_info.json
    
    Args:
        model_path: Path to the .pt model file
        
    Returns:
        Dictionary with training info or None if not found
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return None
    
    # Try to find matching info file
    # Pattern 1: model_name.pt -> model_name_info.json
    info_path = model_path.parent / f"{model_path.stem}_info.json"
    
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    # Pattern 2: Check for latest_training.json in same directory
    latest_path = model_path.parent / "latest_training.json"
    if latest_path.exists():
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                # Verify this info matches the model
                if info.get("model", {}).get("model_file") == model_path.name:
                    return info
        except (json.JSONDecodeError, IOError):
            pass
    
    return None


def display_model_info_card(model_path: str, expanded: bool = False):
    """
    Display model training info as a Streamlit expander.
    
    Args:
        model_path: Path to the .pt model file
        expanded: Whether to expand by default
    """
    info = get_model_info(model_path)
    
    if info is None:
        return
    
    model_info = info.get("model", {})
    training_config = info.get("training_config", {})
    dataset = info.get("dataset", {})
    hardware = info.get("hardware", {})
    metrics = info.get("metrics", {})
    timestamps = info.get("timestamps", {})
    
    with st.expander("📋 Model Training Info", expanded=expanded):
        # Model Details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔧 Base Model**")
            base_model = model_info.get("base_model", "Unknown")
            st.markdown(f"`{base_model}`")
            
            size_mb = model_info.get("model_size_mb")
            if size_mb:
                st.caption(f"Size: {size_mb} MB")
        
        with col2:
            st.markdown("**⚙️ Training Config**")
            epochs = training_config.get("epochs", "?")
            batch = training_config.get("batch_size", "?")
            img_size = training_config.get("image_size", [])
            st.markdown(f"Epochs: `{epochs}`")
            st.markdown(f"Batch: `{batch}`")
            if img_size:
                st.markdown(f"Image: `{img_size[0]}x{img_size[1]}`")
        
        with col3:
            st.markdown("**📊 Results**")
            if metrics:
                mAP50 = metrics.get("mAP50")
                precision = metrics.get("precision")
                recall = metrics.get("recall")
                if mAP50:
                    st.markdown(f"mAP50: `{mAP50:.3f}`" if isinstance(mAP50, float) else f"mAP50: `{mAP50}`")
                if precision:
                    st.markdown(f"Precision: `{precision:.3f}`" if isinstance(precision, float) else f"Precision: `{precision}`")
                if recall:
                    st.markdown(f"Recall: `{recall:.3f}`" if isinstance(recall, float) else f"Recall: `{recall}`")
            else:
                st.caption("No metrics available")
        
        # Additional Details
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Dataset**")
            dataset_name = dataset.get("dataset_name", "Unknown")
            num_classes = dataset.get("num_classes", "?")
            class_names = dataset.get("class_names", [])
            st.markdown(f"Dataset: `{dataset_name}`")
            st.markdown(f"Classes: `{num_classes}` - {class_names}")
        
        with col2:
            st.markdown("**🖥️ Hardware**")
            gpu_name = hardware.get("gpu_name")
            if gpu_name:
                gpu_mem = hardware.get("gpu_memory_gb", "?")
                st.markdown(f"GPU: `{gpu_name}`")
                st.markdown(f"Memory: `{gpu_mem} GB`")
            else:
                st.markdown("GPU: `CPU only`")
        
        # Timestamp
        trained_at = timestamps.get("trained_at", "Unknown")
        if trained_at != "Unknown":
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(trained_at)
                st.caption(f"🕐 Trained: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                st.caption(f"🕐 Trained: {trained_at}")


def display_model_info_compact(model_path: str):
    """
    Display compact model info (single line) for sidebar.
    
    Args:
        model_path: Path to the .pt model file
    """
    info = get_model_info(model_path)
    
    if info is None:
        return
    
    model_info = info.get("model", {})
    training_config = info.get("training_config", {})
    
    base_model = model_info.get("base_model", "?")
    epochs = training_config.get("epochs", "?")
    
    st.caption(f"Base: `{base_model}` | Epochs: `{epochs}`")
