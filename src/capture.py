# src/capture.py
import subprocess
import threading
import time
import os
import signal
import numpy as np
from multiprocessing import shared_memory
from typing import Tuple

class RTSPCapture:
    """
    Subprocess Manager that spins up an isolated FFmpeg instance to decode
    real-time RTSP H.264 camera streams into raw BGR frames.
    Frames are written directly into a pre-allocated shared-memory block
    to implement a zero-copy handoff to the downstream processing threads.
    """
    def __init__(self, rtsp_url: str, shm_name: str, 
                 frame_shape: Tuple[int, int, int] = (1080, 1920, 3), 
                 target_fps: int = 15):
        self.rtsp_url = rtsp_url
        self.shm_name = shm_name
        self.shape = frame_shape
        self.target_fps = target_fps
        self.frame_size = int(np.prod(frame_shape))
        
        # Connect to pre-allocated shared memory segment
        try:
            self.shm = shared_memory.SharedMemory(name=shm_name)
            self.shared_array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            print(f"[Capture] Linked to existing shared-memory segment: '{shm_name}'")
        except FileNotFoundError:
            # Create a new segment if it does not exist
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.frame_size)
            self.shared_array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            print(f"[Capture] Created new shared-memory segment: '{shm_name}' ({self.frame_size} bytes)")
            
        self.process = None
        self.running = False
        self.worker_thread = None

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()
        print("[Capture] FFmpeg ingestion worker thread launched.")

    def _cmd(self) -> list:
        # Optimized FFmpeg parameters for ultra-low latency real-time streaming
        return [
            "ffmpeg",
            "-rtsp_transport", "tcp",           # Enforce TCP transport to prevent UDP packet loss artifacts
            "-hwaccel", "auto",                  # Enable hardware accelerated decoding if available on target host
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",                 # Decode to raw BGR24 for direct consumption by OpenCV
            "-vf", f"fps={self.target_fps},scale={self.shape[1]}:{self.shape[0]}",
            "-an",                               # Disable audio stream decoding to preserve bandwidth and CPU cycle budget
            "-sn",                               # Disable subtitle decoding
            "-tune", "zerolatency",              # Instruct encoder/decoder to eliminate buffer latency
            "pipe:1"                             # Output directly to stdout pipe
        ]

    def _capture_loop(self):
        consecutive_failures = 0
        
        while self.running:
            print(f"[Capture] Launching isolated FFmpeg decoder instance...")
            
            # Start FFmpeg subprocess
            self.process = subprocess.Popen(
                self._cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size
            )
            
            try:
                consecutive_failures = 0
                self._read_stdout()
            except Exception as err:
                consecutive_failures += 1
                backoff_time = min(2 ** consecutive_failures, 30) # Exponential backoff capped at 30 seconds
                print(f"[Capture Error] Ingestion stream broken: {err}.")
                print(f"[Capture] Restarting capture pipeline in {backoff_time}s...")
                
                # Cleanup subprocess safely
                self._cleanup_process()
                
                time.sleep(backoff_time)

    def _read_stdout(self):
        while self.running:
            # Block and read exactly one raw frame chunk from stdout pipe
            chunk = self.process.stdout.read(self.frame_size)
            if not chunk or len(chunk) != self.frame_size:
                # Retrieve errors from stderr to enrich debug logs
                stderr_output = ""
                try:
                    # Non-blocking check of stderr
                    if self.process.stderr:
                        stderr_output = self.process.stderr.read(256).decode('utf-8', errors='ignore')
                except Exception:
                    pass
                raise IOError(f"Incomplete raw video chunk read ({len(chunk)}/{self.frame_size} bytes). FFmpeg stderr: {stderr_output}")

            # Map raw bytes into the Shared Memory array (zero-copy write)
            frame_np = np.frombuffer(chunk, dtype=np.uint8).reshape(self.shape)
            np.copyto(self.shared_array, frame_np)

    def _cleanup_process(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            self.process = None

    def stop(self):
        print("[Capture] Stopping RTSP Ingestion...")
        self.running = False
        self._cleanup_process()
        if self.worker_thread:
            self.worker_thread.join(timeout=3)
        try:
            self.shm.close()
            # Only unlink if we created it (usually unlinked at system shutdown)
            self.shm.unlink()
        except Exception:
            pass
        print("[Capture] Ingestion pipeline successfully stopped.")
