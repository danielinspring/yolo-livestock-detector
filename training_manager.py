"""
Training Manager Module
Handles background training process with persistent status and logs
"""

import subprocess
import json
import os
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class TrainingManager:
    """Manages YOLO training as a background process with persistent state"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.status_file = self.results_dir / "training_status.json"
        self.log_file = self.results_dir / "training.log"
        self.pid_file = self.results_dir / "training.pid"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status from file"""
        default_status = {
            "status": "idle",  # idle, running, completed, failed, stopped
            "start_time": None,
            "end_time": None,
            "config": {},
            "pid": None,
            "message": "No training in progress"
        }
        
        if not self.status_file.exists():
            return default_status
        
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            
            # Verify if running process is still alive
            if status.get("status") == "running":
                if not self._is_process_alive(status.get("pid")):
                    # Process died unexpectedly
                    status["status"] = "failed"
                    status["end_time"] = datetime.now().isoformat()
                    status["message"] = "Training process terminated unexpectedly"
                    self._save_status(status)
            
            return status
        except (json.JSONDecodeError, IOError):
            return default_status
    
    def _save_status(self, status: Dict[str, Any]):
        """Save status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def _is_process_alive(self, pid: Optional[int]) -> bool:
        """Check if a process with given PID is running"""
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    
    def is_running(self) -> bool:
        """Check if training is currently running"""
        status = self.get_status()
        return status.get("status") == "running" and self._is_process_alive(status.get("pid"))
    
    def _create_dataset_yaml(self, data_path: str) -> Path:
        """Create dataset.yaml for the selected dataset"""
        import yaml

        data_path = Path(data_path)
        yaml_content = {
            'path': str(data_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,
            'names': ['cowtail', 'ride']
        }

        yaml_path = Path("configs/dataset.yaml")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        return yaml_path

    def start_training(
        self,
        model: str = "yolov8s",
        epochs: int = 100,
        batch_size: int = 16,
        resume: bool = False,
        device: str = "",
        img_width: int = 640,
        img_height: int = 384,
        clearml_enabled: bool = False,
        clearml_project: str = "YOLO-Training",
        clearml_task: str = "",
        data_path: str = "",
    ) -> Dict[str, Any]:
        """
        Start training in background process

        Args:
            data_path: Path to processed dataset directory (e.g., data/dataset/processed_20260110_123456)

        Returns:
            Status dict with 'success' key indicating if training started
        """
        # Check if already running
        if self.is_running():
            return {
                "success": False,
                "message": "Training is already in progress"
            }

        # Create/update dataset.yaml for selected dataset
        if data_path:
            data_yaml = self._create_dataset_yaml(data_path)
        else:
            data_yaml = Path("configs/dataset.yaml")

        # Verify dataset yaml exists
        if not data_yaml.exists():
            return {
                "success": False,
                "message": f"Dataset config not found: {data_yaml}"
            }

        # Build command
        import sys
        cmd = [
            sys.executable, "scripts/train.py",
            "--model", model,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
            "--img-width", str(img_width),
            "--img-height", str(img_height),
            "--data", str(data_yaml),
        ]

        if device:
            cmd.extend(["--device", device])

        if resume:
            cmd.append("--resume")
        
        # Clear previous log
        if self.log_file.exists():
            self.log_file.unlink()
        
        # Set up environment variables (including ClearML)
        env = os.environ.copy()
        if clearml_enabled:
            env["CLEARML_ENABLED"] = "1"
            if clearml_project:
                env["CLEARML_PROJECT"] = clearml_project
            if clearml_task:
                env["CLEARML_TASK"] = clearml_task
        else:
            env["CLEARML_ENABLED"] = "0"
        
        # Start subprocess with output redirected to log file
        try:
            with open(self.log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent),
                    start_new_session=True,  # Detach from parent
                    env=env,
                )
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
            
            # Save status
            config = {
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "resume": resume,
                "device": device,
                "img_width": img_width,
                "img_height": img_height,
                "data_path": data_path,
                "clearml_enabled": clearml_enabled,
                "clearml_project": clearml_project if clearml_enabled else None,
                "clearml_task": clearml_task if clearml_enabled else None,
            }

            # Get dataset name for display
            dataset_name = Path(data_path).name if data_path else "default"

            status = {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "config": config,
                "pid": process.pid,
                "message": f"Training started with {model}, {epochs} epochs on {dataset_name}"
            }
            self._save_status(status)
            
            return {
                "success": True,
                "message": f"Training started (PID: {process.pid})",
                "pid": process.pid
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to start training: {str(e)}"
            }
    
    def stop_training(self) -> Dict[str, Any]:
        """Stop the running training process"""
        status = self.get_status()
        
        if status.get("status") != "running":
            return {
                "success": False,
                "message": "No training is currently running"
            }
        
        pid = status.get("pid")
        if pid is None:
            return {
                "success": False,
                "message": "No PID found for running training"
            }
        
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            
            # Wait a moment and check if process terminated
            time.sleep(1)
            
            if self._is_process_alive(pid):
                # Force kill if still running
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            
            # Update status
            status["status"] = "stopped"
            status["end_time"] = datetime.now().isoformat()
            status["message"] = "Training stopped by user"
            self._save_status(status)
            
            return {
                "success": True,
                "message": "Training stopped successfully"
            }
            
        except ProcessLookupError:
            # Process already terminated
            status["status"] = "stopped"
            status["end_time"] = datetime.now().isoformat()
            status["message"] = "Training process already terminated"
            self._save_status(status)
            return {
                "success": True,
                "message": "Training process already terminated"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to stop training: {str(e)}"
            }
    
    def get_logs(self, last_n_lines: Optional[int] = None) -> str:
        """
        Get training logs
        
        Args:
            last_n_lines: If specified, return only last N lines
        
        Returns:
            Log content as string
        """
        if not self.log_file.exists():
            return "No logs available yet."
        
        try:
            with open(self.log_file, 'r') as f:
                if last_n_lines is None:
                    return f.read()
                else:
                    lines = f.readlines()
                    return "".join(lines[-last_n_lines:])
        except IOError:
            return "Error reading log file."
    
    def get_log_file_path(self) -> Path:
        """Get path to log file for download"""
        return self.log_file
    
    def clear_status(self):
        """Clear training status (for testing/reset)"""
        if self.status_file.exists():
            self.status_file.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()
    
    def mark_completed(self, success: bool = True, message: str = ""):
        """Mark training as completed (called by monitoring or when process ends)"""
        status = self.get_status()
        status["status"] = "completed" if success else "failed"
        status["end_time"] = datetime.now().isoformat()
        status["message"] = message or ("Training completed successfully" if success else "Training failed")
        self._save_status(status)
