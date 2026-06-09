# src/main.py
import os
import sys
import time
import signal
import numpy as np
from multiprocessing import shared_memory
from typing import Optional

# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EdgeConfig
from src.capture import RTSPCapture
from src.cv_worker import CVWorker
from src.mqtt_publisher import EventPublisher

# Optional systemd integration for system daemon watchdog heartbeats
SYSTEMD_DAEMON_AVAILABLE = False
try:
    from systemd import daemon
    SYSTEMD_DAEMON_AVAILABLE = True
except ImportError:
    pass

class EdgeVisionGateway:
    """
    Core orchestrator class that starts the ingestion pipeline, feeds NPU inference,
    processes spatial rules via the event engine, and manages MQTT publishing.
    """
    def __init__(self, config_path: Optional[str] = None):
        print("====================================================")
        print("  Initializing EdgeVision Platform Smart Gateway   ")
        print("====================================================")
        
        # 1. Load System Configuration
        self.config = EdgeConfig.load_from_json(config_path or "config.json")
        self.device_id = self.config.DEVICE_ID
        
        # Create persistent recording directories if needed
        os.makedirs(self.config.LOCAL_RECORD_DIR, exist_ok=True)
        
        # 2. Setup shared memory buffer allocation for zero-copy frames handoff
        self.frame_size = int(self.config.FRAME_WIDTH * self.config.FRAME_HEIGHT * self.config.FRAME_CHANNELS)
        self.shm_allocated = False
        self.shm = None
        self._allocate_shared_memory()
        
        # 3. Setup Components
        self.capture_pipeline = None
        self.cv_worker = None
        self.publisher = None
        self.running = False
        
        # Bind system termination signals
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)

    def _allocate_shared_memory(self):
        try:
            # Check if segment already exists from a previous dirty crash run
            self.shm = shared_memory.SharedMemory(name=self.config.SHM_NAME)
            print(f"[Main] Attached to existing shared-memory block: '{self.config.SHM_NAME}'")
        except FileNotFoundError:
            # Allocate a new memory block
            self.shm = shared_memory.SharedMemory(name=self.config.SHM_NAME, create=True, size=self.frame_size)
            print(f"[Main] Allocated shared-memory block: '{self.config.SHM_NAME}' ({self.frame_size} bytes)")
        
        self.shared_frame = np.ndarray(
            (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, self.config.FRAME_CHANNELS),
            dtype=np.uint8,
            buffer=self.shm.buf
        )
        self.shm_allocated = True

    def start(self):
        self.running = True
        
        # Initialize Event Publisher with SQLite fallback queueing
        self.publisher = EventPublisher(self.device_id, self.config)
        
        # Initialize CVWorker (detectors, tracking configs, npu compilations)
        self.cv_worker = CVWorker(self.config)
        
        # Initialize isolated FFmpeg decoder pipeline
        self.capture_pipeline = RTSPCapture(
            rtsp_url=self.config.RTSP_URL,
            shm_name=self.config.SHM_NAME,
            frame_shape=(self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH, self.config.FRAME_CHANNELS),
            target_fps=self.config.FPS
        )
        self.capture_pipeline.start()
        
        print("[Main] EdgeVision Smart Gateway components successfully started.")
        self._run_loop()

    def _run_loop(self):
        frame_interval = 1.0 / self.config.FPS
        last_loop_time = time.time()
        loop_counter = 0
        
        # Notify systemd that service initialization is complete
        if SYSTEMD_DAEMON_AVAILABLE:
            daemon.notify("READY=1")
            print("[Main] systemd notifier: READY sent.")
            
        print("[Main] Processing loop active. Monitoring camera feed...")
        
        while self.running:
            start_time = time.time()
            
            # Fetch frame copy from shared memory buffer
            # (Extremely fast, zero CPU-copy deserialization overhead)
            current_frame = self.shared_frame.copy()
            
            # Run model inference, track filters, and calculate boundaries crossing alerts
            tracks, triggered_events = self.cv_worker.process_frame(current_frame, start_time)
            
            # Publish triggered event packets to the cloud gateway
            for event in triggered_events:
                print(f"[ALERT] Triggered {event['type']} (Track ID: {event['track_id']})")
                self.publisher.publish_event(
                    event_type=event["type"],
                    track_id=event["track_id"],
                    bbox=event["bbox"],
                    extra_meta={"latency_ms": (time.time() - start_time) * 1000}
                )
                
            # Perform periodic telemetry logs
            loop_counter += 1
            if loop_counter % (self.config.FPS * 10) == 0:
                # Log telemetry metrics every 10 seconds
                processing_time = (time.time() - start_time) * 1000
                print(f"[Telemetry] Processed 10s frame chunk. Avg Step Latency: {processing_time:.2f}ms")
                
                # Signal systemd watchdog daemon that the loop is alive and healthy
                if SYSTEMD_DAEMON_AVAILABLE:
                    daemon.notify("WATCHDOG=1")
            
            # Enforce pipeline target frame rate ticks pacing
            elapsed = time.time() - start_time
            sleep_needed = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_needed)

    def _handle_termination(self, signum, frame):
        print(f"\n[Termination] Signal {signum} received. Cleaning up resource allocations...")
        self.running = False
        
        if self.capture_pipeline:
            self.capture_pipeline.stop()
            
        if self.publisher:
            self.publisher.close()
            
        if self.shm_allocated:
            try:
                self.shm.close()
                self.shm.unlink()
                print("[Main] Cleaned up shared-memory segment allocations.")
            except Exception as err:
                print(f"[Main Error] Shared-memory unlink failed: {err}")
                
        print("[Termination] EdgeVision Smart Gateway safely exited.")
        sys.exit(0)

if __name__ == "__main__":
    # Allow custom config overrides via CLI parameters
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    gateway = EdgeVisionGateway(config_file)
    gateway.start()
