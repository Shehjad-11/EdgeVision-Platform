# src/config.py
import json
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field

class SpatialZones(BaseModel):
    # Definition of detection polygons and line crossing boundaries
    polygons: Dict[str, List[List[float]]] = Field(
        default_factory=lambda: {
            "loading": [[120.0, 60.0], [280.0, 60.0], [240.0, 200.0], [80.0, 200.0]],
            "restricted": [[360.0, 100.0], [520.0, 100.0], [500.0, 260.0], [320.0, 260.0]]
        }
    )
    lines: Dict[str, List[List[float]]] = Field(
        default_factory=lambda: {
            "line": [[280.0, 40.0], [300.0, 280.0]]
        }
    )
    dwell_limits: Dict[str, float] = Field(
        default_factory=lambda: {
            "loading": 4.0,
            "restricted": 0.0
        }
    )

class EdgeConfig(BaseModel):
    """
    Validation schema defining the system runtime configuration variables.
    Pydantic parses environment overrides and configuration JSON schemas.
    """
    DEVICE_ID: str = "cam-wh-east-04"
    HEF_PATH: str = "/opt/edgevision/models/yolov8n_hailo.hef"
    RTSP_URL: str = "rtsp://admin:securepass123@10.8.0.100:554/stream1"
    
    # Camera capture dimensions and framerate presets
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080
    FRAME_CHANNELS: int = 3
    FPS: int = 15
    SHM_NAME: str = "edgevision_frame_shm"
    
    # Tracking Kalman Buffer constraints
    TRACK_BUFFER_SIZE: int = 45
    
    # Resilience buffer parameters
    SQLITE_DB_PATH: str = "/var/lib/edgevision/queue_buffer.db"
    LOCAL_RECORD_DIR: str = "/opt/edgevision/recordings"
    
    # AWS IoT Endpoint settings
    ENDPOINT: str = "a3q12abcd9e0qp-ats.iot.us-east-1.amazonaws.com"
    PORT: int = 8883
    CA_FILE: str = "/etc/edgevision/certs/AmazonRootCA1.pem"
    CERT_FILE: str = "/etc/edgevision/certs/device.pem.crt"
    KEY_FILE: str = "/etc/edgevision/certs/private.pem.key"
    
    ZONES_CONFIG: SpatialZones = Field(default_factory=SpatialZones)

    @classmethod
    def load_from_json(cls, file_path: str) -> 'EdgeConfig':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Config file {file_path} not found or malformed. Deploying default specs.")
            return cls()
