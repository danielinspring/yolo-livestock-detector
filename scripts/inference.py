"""
Inference Script for YOLO Combined Model
Run detection on images, videos, or real-time streams
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import time
from datetime import datetime


class YOLOInference:
    def __init__(
        self,
        weights_path,
        img_size=(640, 384),
        conf_threshold=0.25,
        iou_threshold=0.45,
        device='',
    ):
        """
        Initialize YOLO inference

        Args:
            weights_path: Path to model weights
            img_size: Input image size (width, height)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to use ('' for auto, 'cpu', '0', etc.)
        """
        self.weights_path = Path(weights_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        # Load model
        print(f"Loading model from: {self.weights_path}")
        self.model = YOLO(str(self.weights_path))
        print("Model loaded successfully!")

        # Class names
        self.class_names = ['ride', 'cowtail']
        self.class_colors = {
            0: (0, 255, 0),    # Green for ride
            1: (255, 0, 0),    # Blue for cowtail
        }

    def predict_image(self, source, save_dir='results/inference', save=True, show=False):
        """
        Run inference on image(s)

        Args:
            source: Image path or directory
            save_dir: Directory to save results
            save: Whether to save results
            show: Whether to show results
        """
        print(f"\nRunning inference on: {source}")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=source,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save,
            show=show,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
            save_txt=True,
            save_conf=True,
            line_width=2,
        )

        print(f"Results saved to: {save_dir}")

        # Print detection summary
        total_detections = 0
        class_counts = {0: 0, 1: 0}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_detections += 1

        print(f"\nDetection Summary:")
        print(f"  Total detections: {total_detections}")
        print(f"  Ride: {class_counts.get(0, 0)}")
        print(f"  Cowtail: {class_counts.get(1, 0)}")

        return results

    def predict_video(self, source, save_dir='results/inference', save=True, show=False):
        """
        Run inference on video

        Args:
            source: Video path or camera index (0 for webcam)
            save_dir: Directory to save results
            save: Whether to save results
            show: Whether to show results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Check if source is camera
        is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

        if is_camera:
            source = int(source) if isinstance(source, str) else source
            print(f"\nRunning real-time inference on camera {source}")
        else:
            print(f"\nRunning inference on video: {source}")

        # Open video
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else -1

        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames > 0:
            print(f"  Total frames: {total_frames}")

        # Setup video writer
        output_path = None
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.mp4"
            output_path = save_dir / output_filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")

        # Inference loop
        frame_count = 0
        fps_counter = []
        class_counts = {0: 0, 1: 0}

        print("\nStarting inference...")
        print("Press 'q' to quit")

        try:
            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run inference
                results = self.model.predict(
                    frame,
                    imgsz=self.img_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )

                # Draw detections
                annotated_frame = frame.copy()
                detections_this_frame = {0: 0, 1: 0}

                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Update counts
                        detections_this_frame[class_id] += 1

                        # Draw box
                        color = self.class_colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2
                        )

                        # Draw label
                        label = f"{self.class_names[class_id]}: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1) + label_size[0], int(y1)),
                            color,
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

                # Calculate FPS
                inference_time = time.time() - start_time
                current_fps = 1 / inference_time if inference_time > 0 else 0
                fps_counter.append(current_fps)

                # Keep only last 30 FPS measurements
                if len(fps_counter) > 30:
                    fps_counter.pop(0)

                avg_fps = sum(fps_counter) / len(fps_counter)

                # Draw info
                info_text = [
                    f"Frame: {frame_count}",
                    f"FPS: {avg_fps:.1f}",
                    f"Ride: {detections_this_frame[0]}",
                    f"Cowtail: {detections_this_frame[1]}",
                ]

                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # Save frame
                if save and output_path:
                    out.write(annotated_frame)

                # Show frame
                if show:
                    cv2.imshow('YOLO Inference', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Print progress
                if not is_camera and total_frames > 0:
                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {avg_fps:.1f}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            cap.release()
            if save and output_path:
                out.release()
            if show:
                cv2.destroyAllWindows()

            print(f"\nProcessed {frame_count} frames")
            print(f"Average FPS: {sum(fps_counter) / len(fps_counter):.1f}")

            if save:
                print(f"Output saved to: {output_path}")

    def predict_stream(self, source, save_dir='results/inference', save=True, show=True):
        """
        Run inference on live stream (wrapper for predict_video with camera source)

        Args:
            source: Camera index (default: 0 for default webcam)
            save_dir: Directory to save results
            save: Whether to save results
            show: Whether to show results
        """
        self.predict_video(source, save_dir, save, show)


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on images/videos/streams')

    parser.add_argument(
        '--weights',
        type=str,
        default='models/best.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source: image path, video path, directory, or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--img-width',
        type=int,
        default=640,
        help='Image width (default: 640)'
    )
    parser.add_argument(
        '--img-height',
        type=int,
        default=384,
        help='Image height (default: 384)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (default: auto, options: cpu, 0, 0,1, etc.)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/inference',
        help='Directory to save results'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show results in real-time'
    )

    args = parser.parse_args()

    # Initialize inference
    inference = YOLOInference(
        weights_path=args.weights,
        img_size=(args.img_width, args.img_height),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )

    # Determine source type
    source = args.source

    # Check if source is a number (camera)
    if source.isdigit():
        source = int(source)
        inference.predict_stream(
            source=source,
            save_dir=args.save_dir,
            save=not args.no_save,
            show=args.show,
        )
    # Check if source is video
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')):
        inference.predict_video(
            source=source,
            save_dir=args.save_dir,
            save=not args.no_save,
            show=args.show,
        )
    # Otherwise treat as image or directory
    else:
        inference.predict_image(
            source=source,
            save_dir=args.save_dir,
            save=not args.no_save,
            show=args.show,
        )

    print("\nInference complete!")


if __name__ == "__main__":
    main()
