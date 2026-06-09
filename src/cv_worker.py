# src/cv_worker.py
import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional
from src.event_engine import EventEngine

# Robust imports to allow compile-time checks and fallback simulation
# on development hardware lacking physical Hailo-8L PCIe accelerators
HAILO_SDK_AVAILABLE = False
try:
    from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, InferModel
    HAILO_SDK_AVAILABLE = True
except ImportError:
    print("[NPU Warning] Hailo SDK ('hailo_platform') not found on system path.")
    print("[NPU Warning] Defaulting CVWorker execution to CPU Simulation/Mock Mode.")

class Track:
    """Stateful tracking item managing track lifetime and trajectory history."""
    def __init__(self, track_id: int, tlbr: List[float], score: float, class_id: int):
        self.track_id = track_id
        self.tlbr = tlbr  # [top, left, bottom, right] -> [ymin, xmin, ymax, xmax]
        self.score = score
        self.class_id = class_id
        self.history: List[Tuple[float, float]] = []
        self.time_since_update = 0
        self.age = 0

class SimpleByteTracker:
    """
    Lightweight, robust CPU-bound tracker executing IoU greedy-association.
    Provides identical interfaces to ByteTrack, eliminating complex external DLL dependencies.
    """
    def __init__(self, track_buffer: int = 45):
        self.track_buffer = track_buffer
        self.next_id = 1
        self.tracked_stracks: List[Track] = []

    def update(self, detections: List[List[float]], img_shape: Tuple[int, ...]) -> List[Track]:
        # Clean expired tracks that haven't been matched within the buffer limit
        self.tracked_stracks = [t for t in self.tracked_stracks if t.time_since_update < self.track_buffer]
        
        for t in self.tracked_stracks:
            t.time_since_update += 1
            t.age += 1

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracked_stracks)))

        if self.tracked_stracks and detections:
            # Construct IoU distance matrix between tracks and new detections
            ious = np.zeros((len(self.tracked_stracks), len(detections)))
            for t_idx, track in enumerate(self.tracked_stracks):
                for d_idx, det in enumerate(detections):
                    ious[t_idx, d_idx] = self.iou(track.tlbr, det[:4])
            
            # Greedy association based on maximum IoU overlap
            for t_idx in range(len(self.tracked_stracks)):
                best_d_idx = -1
                best_iou = 0.3  # IoU matching threshold
                for d_idx in unmatched_detections:
                    if ious[t_idx, d_idx] > best_iou:
                        best_iou = ious[t_idx, d_idx]
                        best_d_idx = d_idx
                
                if best_d_idx != -1:
                    matched_indices.append((t_idx, best_d_idx))
                    unmatched_detections.remove(best_d_idx)
                    unmatched_tracks.remove(t_idx)

        # Update matched tracks with new coordinates and class details
        for t_idx, d_idx in matched_indices:
            track = self.tracked_stracks[t_idx]
            det = detections[d_idx]
            track.tlbr = det[:4]
            track.score = det[4]
            track.class_id = int(det[5])
            track.time_since_update = 0

        # Create new track instances for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_id,
                tlbr=det[:4],
                score=det[4],
                class_id=int(det[5])
            )
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        # Return only active confirmed tracks that were updated this frame
        return [t for t in self.tracked_stracks if t.time_since_update == 0]

    @staticmethod
    def iou(box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        b1_y1, b1_x1, b1_y2, b1_x2 = box1
        b2_y1, b2_x1, b2_y2, b2_x2 = box2
        
        y1 = max(b1_y1, b2_y1)
        x1 = max(b1_x1, b2_x1)
        y2 = min(b1_y2, b2_y2)
        x2 = min(b1_x2, b2_x2)
        
        inter_area = max(0.0, y2 - y1) * max(0.0, x2 - x1)
        box1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        box2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

class CVWorker:
    """
    Downstream processing worker that coordinates Hailo NPU hardware inference,
    object tracking metrics, and polygon boundaries event checks.
    """
    def __init__(self, config: Any):
        self.config = config
        self.event_engine = EventEngine(config.ZONES_CONFIG)
        self.tracker = SimpleByteTracker(config.TRACK_BUFFER_SIZE)
        self.mock_targets = self._initialize_mock_targets()
        
        if HAILO_SDK_AVAILABLE:
            try:
                self._initialize_hardware_npu()
                self.hardware_active = True
                print("[NPU] Hailo-8L NPU PCIe hardware successfully initialized.")
            except Exception as err:
                print(f"[NPU Error] Physical accelerator failure: {err}.")
                print("[NPU] Reverting to CPU Simulation Mode.")
                self.hardware_active = False
        else:
            self.hardware_active = False

    def _initialize_hardware_npu(self):
        # Scan PCIe bus for available Hailo-8L cores
        self.vdevice = VDevice()
        self.hef = HEF(self.config.HEF_PATH)
        
        # Configure model inputs and pipeline stream descriptors
        configure_params = ConfigureParams.create_from_hef(self.hef)
        self.configured_network = self.vdevice.configure(self.hef, configure_params)[0]
        
        self.input_vstream_params = InputVStreamParams.make(self.configured_network)
        self.output_vstream_params = OutputVStreamParams.make(self.configured_network)
        
        # Load compiled HEF inference descriptors
        self.runner = self.vdevice.create_infer_model(self.hef)
        self.runner.set_batch_size(1)

    def _initialize_mock_targets(self) -> List[Dict[str, Any]]:
        # Set up coordinates to simulate realistic tracking trajectories
        return [
            {"id": 104, "type": "Person", "x": 100.0, "y": 80.0, "vx": 3.2, "vy": 0.5, "conf": 0.94, "class_id": 0},
            {"id": 112, "type": "Person", "x": 480.0, "y": 70.0, "vx": -2.5, "vy": 1.2, "conf": 0.88, "class_id": 0},
            {"id": 201, "type": "Forklift", "x": 200.0, "y": 250.0, "vx": 2.0, "vy": -0.8, "conf": 0.92, "class_id": 1}
        ]

    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Executes NPU inference (or OpenCV contours simulation) on the frame,
        associates identities via tracking filters, and updates zone rules.
        """
        h, w, c = frame.shape
        
        if self.hardware_active:
            # Resize image to match model input dimensions
            resized_frame = cv2.resize(frame, (640, 640))
            # Perform hardware inference block (INT8 PCIe Gen 2 mapping)
            npu_outputs = self.runner.infer(resized_frame)
            detections = self._post_process_detections(npu_outputs, (h, w))
            tracks = self._update_tracker(detections, frame.shape)
        else:
            # CPU Fallback Simulation Mode
            tracks = self._simulate_cv_tracking(w, h)
            # Small artificial sleep to simulate ~38ms hardware latency budget
            time.sleep(0.038)
            
        # Evaluate spatial polygon containment and crossings
        triggered_events = self.event_engine.update(tracks, timestamp)
        
        return tracks, triggered_events

    def _post_process_detections(self, npu_outputs: List[np.ndarray], orig_shape: Tuple[int, int]) -> List[List[float]]:
        """Parses raw Hailo NPU output tensors into bounding boxes."""
        detections = []
        if not npu_outputs:
            return detections
            
        # Standard Hailo Object Detection output format parser
        # Usually demuxed as: [Boxes (100, 4), Classes (100,), Scores (100,)]
        try:
            boxes = npu_outputs[0][0]    # [ymin, xmin, ymax, xmax] normalized
            scores = npu_outputs[1][0]
            classes = npu_outputs[2][0]
            
            h, w = orig_shape
            for i in range(len(scores)):
                score = float(scores[i])
                if score < 0.45:
                    continue
                    
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Convert back to absolute pixels
                y1 = float(ymin * h)
                x1 = float(xmin * w)
                y2 = float(ymax * h)
                x2 = float(xmax * w)
                class_id = int(classes[i])
                
                # Format: [ymin, xmin, ymax, xmax, score, class_id]
                detections.append([y1, x1, y2, x2, score, class_id])
        except IndexError:
            pass
            
        return detections

    def _update_tracker(self, detections: List[List[float]], img_shape: Tuple[int, ...]) -> List[Any]:
        """Runs the coordinate matching update loop on the active tracker."""
        return self.tracker.update(detections, img_shape)

    def _simulate_cv_tracking(self, w: int, h: int) -> List[Track]:
        """
        Generates simulated coordinate tracking updates that mirror physical objects
        moving across the warehouse floor. Used for developer validation.
        """
        simulated_tracks = []
        
        for tgt in self.mock_targets:
            # Simple bounce physics inside screen canvas boundaries
            tgt["x"] += tgt["vx"]
            tgt["y"] += tgt["vy"]
            
            if tgt["x"] < 50 or tgt["x"] > w - 50:
                tgt["vx"] *= -1
            if tgt["y"] < 50 or tgt["y"] > h - 50:
                tgt["vy"] *= -1
                
            # Construct a relative bounding box centered around coordinates
            bw, bh = (64, 48) if tgt["type"] == "Forklift" else (36, 58)
            top = tgt["y"] - bh / 2
            left = tgt["x"] - bw / 2
            bottom = tgt["y"] + bh / 2
            right = tgt["x"] + bw / 2
            
            trackObj = Track(
                track_id=tgt["id"],
                tlbr=[top, left, bottom, right],
                score=tgt["conf"],
                class_id=tgt["class_id"]
            )
            simulated_tracks.append(trackObj)
            
        return simulated_tracks
