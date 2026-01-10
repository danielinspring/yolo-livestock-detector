"""
YOLO Combined Model - Streamlit Web UI
Main application entry point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="YOLO Combined Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Password authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.get("password", "admin123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.markdown("## 🔐 Login Required")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter password to access the app"
        )
        st.caption("Contact administrator if you forgot the password.")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.markdown("## 🔐 Login Required")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter password to access the app"
        )
        st.error("❌ Incorrect password. Please try again.")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Do not continue if password is wrong

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<div class="main-header">🎯 YOLO Combined Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ride & Cowtail Detection System</div>', unsafe_allow_html=True)

# Welcome section
st.markdown("---")
st.markdown("### Welcome to the YOLO Detection System")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 📸 Inference")
    st.markdown("""
    - Upload images or videos
    - Real-time detection
    - Adjustable confidence
    - Download results
    """)
    if st.button("Go to Inference →", key="btn_inference"):
        st.switch_page("pages/1_📸_Inference.py")

with col2:
    st.markdown("#### 🏷️ Auto-Label")
    st.markdown("""
    - Batch labeling
    - YOLO format output
    - Visualization
    - Export labels
    """)
    if st.button("Go to Auto-Label →", key="btn_autolabel"):
        st.switch_page("pages/2_🏷️_Auto_Label.py")

with col3:
    st.markdown("#### 📊 Dataset Info")
    st.markdown("""
    - Dataset statistics
    - Class distribution
    - Data visualization
    - Quality checks
    """)
    if st.button("Go to Dataset Info →", key="btn_dataset"):
        st.switch_page("pages/3_📊_Dataset_Info.py")

with col4:
    st.markdown("#### 🚀 Training")
    st.markdown("""
    - Train YOLO models
    - Real-time logs
    - Background training
    - ClearML tracking
    """)
    if st.button("Go to Training →", key="btn_training"):
        st.switch_page("pages/4_🚀_Training.py")

st.markdown("---")

# Second row for Evaluation
col_eval, col_empty1, col_empty2, col_empty3 = st.columns(4)

with col_eval:
    st.markdown("#### 📈 Evaluation")
    st.markdown("""
    - Sample predictions
    - Confusion matrix
    - Performance metrics
    - Supervision viz
    """)
    if st.button("Go to Evaluation →", key="btn_evaluation"):
        st.switch_page("pages/5_📈_Evaluation.py")

st.markdown("---")

# Project info
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Model Information")

    # Check if model exists
    model_path = Path("models/best.pt")
    if model_path.exists():
        st.success("✓ Trained model available")
        st.info(f"Model: {model_path}")
        size_mb = model_path.stat().st_size / (1024 * 1024)
        st.metric("Model Size", f"{size_mb:.1f} MB")
    else:
        st.warning("⚠ No trained model found")
        st.info("Train a model using the 🚀 Training page or CLI: `python scripts/train.py`")

with col2:
    st.markdown("### 📦 Dataset Status")

    # Check dataset
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        train_dir = processed_dir / "images" / "train"
        val_dir = processed_dir / "images" / "val"
        test_dir = processed_dir / "images" / "test"

        if train_dir.exists():
            train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
            val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0
            test_count = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png"))) if test_dir.exists() else 0

            st.success("✓ Dataset ready")

            cols = st.columns(3)
            with cols[0]:
                st.metric("Train", train_count)
            with cols[1]:
                st.metric("Val", val_count)
            with cols[2]:
                st.metric("Test", test_count)
        else:
            st.warning("⚠ Dataset not split")
            st.info("Run: `python scripts/split_data.py`")
    else:
        st.warning("⚠ No processed dataset")
        st.info("Run: `python scripts/preprocess_data.py`")

st.markdown("---")

# Quick start guide
with st.expander("📖 Quick Start Guide"):
    st.markdown("""
    ### Getting Started

    **1. Prepare Data**
    ```bash
    # Preprocess Label Studio export
    python scripts/preprocess_data.py --images /path/to/images

    # Split dataset
    python scripts/split_data.py
    ```

    **2. Train Model**
    ```bash
    # Train YOLOv8s model
    python scripts/train.py --model yolov8s --epochs 100
    ```

    **3. Use the Web UI**
    - **Inference**: Upload images/videos for detection
    - **Auto-Label**: Batch label new images
    - **Dataset Info**: View dataset statistics

    **4. Advanced Features**
    ```bash
    # Evaluate model
    python scripts/evaluate.py --weights models/best.pt

    # Compare models
    python scripts/compare_models.py --combined models/best.pt

    # Command-line inference
    python scripts/inference.py --source video.mp4 --weights models/best.pt
    ```

    For detailed documentation, see `QUICKSTART.md` and `README.md`
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    st.markdown("Use the pages above to:")
    st.markdown("- 📸 Run inference")
    st.markdown("- 🏷️ Auto-label images")
    st.markdown("- 📊 View dataset info")
    st.markdown("- 🚀 Train models")
    st.markdown("- 📈 Evaluate models")

    st.markdown("---")

    st.markdown("## ⚙️ Configuration")

    st.markdown("### Model Classes")
    st.code("""
0: cowtail
1: ride
    """)

    st.markdown("### Input Size")
    st.code("640 x 384")

    st.markdown("---")

    st.markdown("## 📚 Resources")
    st.markdown("[YOLO Documentation](https://docs.ultralytics.com)")
    st.markdown("[GitHub Repository](https://github.com/ultralytics/ultralytics)")

    st.markdown("---")
    st.caption("YOLO Combined Model v1.0")
