# cloud/backend_api.py
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends

# Robust fallback imports to ensure script compiles and runs offline
BOTO3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import ClientError
    from boto3.dynamodb.conditions import Key
    BOTO3_AVAILABLE = True
except ImportError:
    print("[Cloud Warning] AWS SDK ('boto3') not installed.")
    print("[Cloud Warning] Defaulting FastAPI Backend to Mock Database Mode.")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="EdgeVision Cloud API Backend",
    version="1.0.0",
    description="REST API Gateway managing edge cameras, event queries, and S3 pre-signed media links."
)

# Enable CORS for public portfolio showcase calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# DATA VALIDATION SCHEMAS (PYDANTIC)
# =========================================================================
class ZoneConfig(BaseModel):
    loading: List[List[float]]
    restricted: List[List[float]]

class DeviceConfigUpdate(BaseModel):
    version: int
    zones: ZoneConfig

class DeviceSchema(BaseModel):
    device_id: str
    status: str
    ip_address: str
    fw_version: str
    npu_temp_c: float
    uptime_sec: int
    active_zones: List[str]

class EventSchema(BaseModel):
    event_id: str
    device_id: str
    timestamp: str
    event_type: str
    track_id: int
    confidence: float
    clip_s3_url: str
    thumbnail_s3_url: str

# Mock Database Store for Offline execution
MOCK_DEVICES = [
    {
        "device_id": "cam-wh-east-04",
        "status": "ONLINE",
        "ip_address": "10.8.0.14",
        "fw_version": "v1.4.2",
        "npu_temp_c": 49.3,
        "uptime_sec": 2643800,
        "active_zones": ["loading", "restricted"]
    },
    {
        "device_id": "cam-wh-dock-01",
        "status": "ONLINE",
        "ip_address": "10.8.0.22",
        "fw_version": "v1.4.2",
        "npu_temp_c": 47.1,
        "uptime_sec": 1948500,
        "active_zones": ["dock"]
    }
]

MOCK_EVENTS = [
    {
        "event_id": "evt_9f27a4d1b82",
        "device_id": "cam-wh-east-04",
        "timestamp": "2026-06-09T17:42:01.082Z",
        "event_type": "RESTRICTED_ZONE_VIOLATION",
        "track_id": 104,
        "confidence": 0.94,
        "clip_s3_url": "s3://edgevision-prod-clips/clips/cam-wh-east-04/20260609_174201.mp4",
        "thumbnail_s3_url": "s3://edgevision-prod-clips/thumbnails/evt_9f27a4d1b82.jpg"
    },
    {
        "event_id": "evt_8d11c0f4f91",
        "device_id": "cam-wh-east-04",
        "timestamp": "2026-06-09T17:41:24.412Z",
        "event_type": "LOADING_ZONE_DWELL_ALERT",
        "track_id": 98,
        "confidence": 0.88,
        "clip_s3_url": "s3://edgevision-prod-clips/clips/cam-wh-east-04/20260609_174124.mp4",
        "thumbnail_s3_url": "s3://edgevision-prod-clips/thumbnails/evt_8d11c0f4f91.jpg"
      }
]

# =========================================================================
# API ROUTER ENDPOINTS
# =========================================================================

@app.get("/api/v1/devices", response_model=List[DeviceSchema])
async def get_registered_devices():
    """Fetches status metrics for all registered Smart Camera Gateways."""
    if BOTO3_AVAILABLE and not os.environ.get("MOCK_MODE"):
        try:
            # Fetch statuses from DynamoDB device registry tables
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            table = dynamodb.Table('EdgeVisionDevices')
            response = table.scan()
            return response.get('Items', [])
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"DynamoDB Query Failed: {err}")
    return MOCK_DEVICES

@app.get("/api/v1/events", response_model=List[EventSchema])
async def get_event_logs(device_id: Optional[str] = None, limit: int = 20):
    """Queries spatial alert event items from the central database."""
    if BOTO3_AVAILABLE and not os.environ.get("MOCK_MODE"):
        try:
            dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
            table = dynamodb.Table('EdgeVisionEvents')
            
            # Simple query optimization using partition index
            if device_id:
                response = table.query(
                    KeyConditionExpression=Key('device_id').eq(device_id),
                    Limit=limit
                )
            else:
                response = table.scan(Limit=limit)
            return response.get('Items', [])
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"DynamoDB Scan Failed: {err}")
    return MOCK_EVENTS

@app.get("/api/v1/events/{event_id}/clip")
async def get_clip_signed_url(event_id: str):
    """
    Generates a secure S3 Pre-Signed URL with a 1-hour expiration.
    Decouples raw video security access from public API networks.
    """
    bucket_name = "edgevision-prod-clips"
    object_key = f"clips/cam-wh-east-04/{event_id}_transcoded.mp4"
    
    if BOTO3_AVAILABLE and not os.environ.get("MOCK_MODE"):
        try:
            s3_client = boto3.client('s3', region_name='us-east-1')
            # Generate pre-signed URL valid for 3600 seconds
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_key},
                ExpiresIn=3600
            )
            return {"event_id": event_id, "pre_signed_url": url, "expires_in_sec": 3600}
        except ClientError as err:
            raise HTTPException(status_code=500, detail=f"S3 Pre-signing generation failed: {err}")
            
    # Mock pre-signed URL for testing
    mock_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE&Expires=3600&Signature=MockSigValue="
    return {"event_id": event_id, "pre_signed_url": mock_url, "expires_in_sec": 3600}

@app.post("/api/v1/devices/{device_id}/config")
async def update_device_config(device_id: str, config: DeviceConfigUpdate):
    """
    Pushes an OTA configuration coordinate zone update to the edge camera.
    Publishes parameters to the device AWS IoT MQTT config channel.
    """
    payload = {
        "timestamp": int(time.time()),
        "version": config.version,
        "zones": config.zones.dict()
    }
    
    if BOTO3_AVAILABLE and not os.environ.get("MOCK_MODE"):
        try:
            # Publish configuration payload directly to AWS IoT Core MQTT broker topic
            iot_client = boto3.client('iot-data', region_name='us-east-1')
            iot_client.publish(
                topic=f"devices/{device_id}/config",
                qos=1,
                payload=config.json()
            )
            return {"status": "SUCCESS", "device_id": device_id, "msg": "OTA update published via IoT Core topic."}
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"AWS IoT Core Publish Failed: {err}")
            
    # Mock response
    return {
        "status": "SUCCESS", 
        "device_id": device_id, 
        "published_topic": f"devices/{device_id}/config",
        "qos": 1,
        "payload": payload
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
