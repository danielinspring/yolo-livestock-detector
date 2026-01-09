"""
YOLO Training Page
Train models through the GUI with real-time logs and persistent status
"""

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_manager import TrainingManager

# Page configuration
st.set_page_config(
    page_title="Training - YOLO Combined Model",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        padding: 1rem;
        border-radius: 0.5rem;
        height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-completed {
        color: #2196F3;
        font-weight: bold;
    }
    .status-failed {
        color: #f44336;
        font-weight: bold;
    }
    .status-stopped {
        color: #FF9800;
        font-weight: bold;
    }
    .status-idle {
        color: #9E9E9E;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize training manager
manager = TrainingManager()

# Page header
st.markdown("# 🚀 Model Training")
st.markdown("Train YOLO models with real-time monitoring. Training continues even if you close the browser.")
st.markdown("---")

# Get current status
status = manager.get_status()
is_running = manager.is_running()

# Status display
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_text = status.get("status", "idle").upper()
    status_class = f"status-{status.get('status', 'idle')}"
    st.markdown(f"**Status:** <span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)

with col2:
    if status.get("start_time"):
        start_time = datetime.fromisoformat(status["start_time"])
        st.markdown(f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.markdown("**Started:** -")

with col3:
    if is_running and status.get("start_time"):
        start_time = datetime.fromisoformat(status["start_time"])
        elapsed = datetime.now() - start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        st.markdown(f"**Elapsed:** {hours:02d}:{minutes:02d}:{seconds:02d}")
    elif status.get("end_time") and status.get("start_time"):
        start_time = datetime.fromisoformat(status["start_time"])
        end_time = datetime.fromisoformat(status["end_time"])
        elapsed = end_time - start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        st.markdown(f"**Duration:** {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        st.markdown("**Elapsed:** -")

with col4:
    config = status.get("config", {})
    if config:
        st.markdown(f"**Config:** {config.get('model', '-')} | {config.get('epochs', '-')} epochs")
    else:
        st.markdown("**Config:** -")

st.markdown("---")

# Main content layout
if is_running:
    # Show running status and logs
    st.markdown("### 📊 Training in Progress")
    st.info(f"🔄 {status.get('message', 'Training is running...')}")
    
    # Stop button
    if st.button("⏹️ Stop Training", type="primary", use_container_width=True):
        result = manager.stop_training()
        if result["success"]:
            st.success(result["message"])
            time.sleep(1)
            st.rerun()
        else:
            st.error(result["message"])
    
    st.markdown("---")
    
    # Real-time logs
    st.markdown("### 📋 Training Logs (Real-time)")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh logs", value=True)
    
    # Log display
    logs = manager.get_logs(last_n_lines=100)
    
    # Display logs in a scrollable container
    st.markdown(f'<div class="log-container">{logs}</div>', unsafe_allow_html=True)
    
    # Download full log
    log_path = manager.get_log_file_path()
    if log_path.exists():
        with open(log_path, 'r') as f:
            full_log = f.read()
        st.download_button(
            label="📥 Download Full Log",
            data=full_log,
            file_name="training.log",
            mime="text/plain"
        )
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(2)
        st.rerun()

else:
    # Show configuration form
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### ⚙️ Training Configuration")
        
        # Model selection
        model = st.selectbox(
            "Model Version",
            options=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                     'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
            index=1,  # yolov8s default
            help="Select YOLO model version. Smaller models (n) are faster but less accurate."
        )
        
        # Epochs
        epochs = st.slider(
            "Epochs",
            min_value=10,
            max_value=500,
            value=150,
            step=10,
            help="Number of training epochs. More epochs = longer training but potentially better results."
        )
        
        # Batch size
        batch_size = st.select_slider(
            "Batch Size",
            options=[4, 8, 16, 32, 64],
            value=16,
            help="Batch size for training. Larger batches need more GPU memory."
        )
        
        # Image size
        col_w, col_h = st.columns(2)
        with col_w:
            img_width = st.number_input("Image Width", value=640, min_value=320, max_value=1280, step=32)
        with col_h:
            img_height = st.number_input("Image Height", value=384, min_value=192, max_value=960, step=32)
        
        # Resume training
        resume = st.checkbox(
            "Resume from checkpoint",
            value=False,
            help="Continue training from last checkpoint if available"
        )
        
        # Device selection
        device = st.selectbox(
            "Device",
            options=["", "cpu", "0", "0,1"],
            format_func=lambda x: "Auto (GPU if available)" if x == "" else x,
            help="Select training device"
        )
        
        st.markdown("---")
        
        # ClearML Integration
        st.markdown("### 📊 ClearML Tracking")
        clearml_enabled = st.checkbox(
            "Enable ClearML",
            value=False,
            help="Enable experiment tracking with ClearML. Requires clearml-init setup."
        )
        
        if clearml_enabled:
            clearml_project = st.text_input(
                "Project Name",
                value="YOLO-Training",
                help="ClearML project name for organizing experiments"
            )
            clearml_task = st.text_input(
                "Task Name (optional)",
                value="",
                placeholder="Auto-generated if empty",
                help="Custom task name. Leave empty for auto-generated name."
            )
            st.info("💡 Make sure you've run `clearml-init` to configure credentials.")
        else:
            clearml_project = "YOLO-Training"
            clearml_task = ""
        
        st.markdown("---")
        
        # Start button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Starting training..."):
                result = manager.start_training(
                    model=model,
                    epochs=epochs,
                    batch_size=batch_size,
                    resume=resume,
                    device=device,
                    img_width=img_width,
                    img_height=img_height,
                    clearml_enabled=clearml_enabled,
                    clearml_project=clearml_project,
                    clearml_task=clearml_task,
                )
                
                if result["success"]:
                    st.success(result["message"])
                    if clearml_enabled:
                        st.info("📊 ClearML tracking enabled. View experiments at: https://app.clear.ml")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(result["message"])
    
    with col_right:
        st.markdown("### 📊 Previous Training")
        
        if status.get("status") in ["completed", "failed", "stopped"]:
            # Show previous training info
            if status.get("status") == "completed":
                st.success(f"✅ {status.get('message', 'Training completed')}")
            elif status.get("status") == "failed":
                st.error(f"❌ {status.get('message', 'Training failed')}")
            else:
                st.warning(f"⏹️ {status.get('message', 'Training stopped')}")
            
            prev_config = status.get("config", {})
            if prev_config:
                st.markdown("**Previous Configuration:**")
                st.json(prev_config)
            
            # Show logs from previous training
            st.markdown("**Last Training Logs:**")
            logs = manager.get_logs(last_n_lines=50)
            if logs:
                st.code(logs, language="text")
                
                # Download full log
                log_path = manager.get_log_file_path()
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        full_log = f.read()
                    st.download_button(
                        label="📥 Download Full Log",
                        data=full_log,
                        file_name="training.log",
                        mime="text/plain"
                    )
        else:
            st.info("No previous training history. Configure and start training to begin.")
        
        # Dataset status
        st.markdown("---")
        st.markdown("### 📦 Dataset Status")
        
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            train_dir = processed_dir / "images" / "train"
            val_dir = processed_dir / "images" / "val"
            
            if train_dir.exists():
                train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
                val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png"))) if val_dir.exists() else 0
                
                st.success("✓ Dataset ready for training")
                col_t, col_v = st.columns(2)
                with col_t:
                    st.metric("Training Images", train_count)
                with col_v:
                    st.metric("Validation Images", val_count)
            else:
                st.warning("⚠️ Dataset not split")
                st.info("Run: `python scripts/split_data.py`")
        else:
            st.warning("⚠️ No processed dataset")
            st.info("Run: `python scripts/preprocess_data.py`")
        
        # Model status
        st.markdown("---")
        st.markdown("### 🎯 Trained Model")
        
        model_path = Path("models/best.pt")
        if model_path.exists():
            st.success("✓ Trained model available")
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.metric("Model Size", f"{size_mb:.1f} MB")
        else:
            st.info("No trained model yet. Start training to create one.")

# Sidebar info
with st.sidebar:
    st.markdown("## 🚀 Training")
    st.markdown("""
    Train YOLO models directly from the web interface.
    
    **Features:**
    - Background training
    - Real-time log viewing
    - Persistent status
    - Resume from checkpoint
    
    **Tips:**
    - Start with smaller epochs (10-20) to test
    - Monitor logs for errors
    - Training persists even if browser closes
    """)
    
    st.markdown("---")
    
    # Manual refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Clear status (for debugging)
    with st.expander("🔧 Advanced"):
        if st.button("Clear Training Status"):
            if not manager.is_running():
                manager.clear_status()
                st.success("Status cleared")
                st.rerun()
            else:
                st.error("Cannot clear while training is running")
