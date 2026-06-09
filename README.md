# EdgeVision Platform - Real-time Edge AI Camera Gateway

EdgeVision is an end-to-end, high-performance computer vision platform designed for real-time object tracking and boundary crossing detection. Engineered to run efficiently on low-power edge hardware (Raspberry Pi 5 + Hailo-8L NPU), it combines high-performance edge inference with mutual TLS cloud ingestion and serverless media processing.

---

## 🛠️ System Architecture

The platform architecture is designed to minimize inter-process communication overhead and host CPU load:

```
[ Reolink PoE Camera (RTSP) ]
            │
            ▼
┌─────────────────────────────────────── Edge Gateway (Raspberry Pi 5) ────────────────────────────────────────┐
│                                                                                                              │
│  ┌────────────────────────┐  Raw BGR  ┌────────────────────────┐  Inference   ┌───────────────────────────┐  │
│  │   FFmpeg Subprocess    │ ────────> │   Shared Memory (SHM)  │ ───────────> │     Hailo-8L NPU Core     │  │
│  │    Ingestion Engine    │           │    Circular Buffer     │              │    (INT8 Quantized HEF)   │  │
│  └────────────────────────┘           └────────────────────────┘              └───────────────────────────┘  │
│                                                                                             │                │
│                                                                                             ▼                │
│  ┌────────────────────────┐           ┌────────────────────────┐  Centroids   ┌───────────────────────────┐  │
│  │    Local NVMe Storage  │ <──────── │    Event Engine        │ <─────────── │   ByteTrack Track Filters │  │
│  │ (Rolling 5-min clips)  │  (Slice)  │  (Ray Casting Bounds)  │              │     (Kalman Estimator)    │  │
│  └────────────────────────┘           └────────────────────────┘              └───────────────────────────┘  │
│              │                                     │                                                         │
└──────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┘
               │ (Secure Uploads)                    │ (mTLS 1.3 Events)
               ▼                                     ▼
┌────────────────────────────────────────────── AWS Cloud Services ────────────────────────────────────────────┐
│              │                                     │                                                         │
│              ▼                                     ▼                                                         │
│   ┌──────────────────────┐              ┌──────────────────────┐              ┌──────────────────────────┐   │
│   │    Amazon S3 Bucket  │              │     AWS IoT Core     │ ───────────> │    FastAPI API Server    │   │
│   └──────────────────────┘              └──────────────────────┘              │      (ECS Container)     │   │
│              │                                     │                          └──────────────────────────┘   │
│              ▼ (Object Trigger)                    ▼ (IoT Rules)                                             │
│   ┌──────────────────────┐              ┌──────────────────────┐                                             │
│   │   AWS Lambda ffmpeg  │              │   Amazon DynamoDB    │                                             │
│   │ (Serverless HLS)     │              │     (Alert Logs)     │                                             │
│   └──────────────────────┘              └──────────────────────┘                                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 💾 Repository Directory Structure

```
├── README.md                          # Comprehensive documentation
├── index.html                         # Premium interactive case-study dashboard
├── src/
│   ├── main.py                        # System orchestrator runtime
│   ├── config.py                      # Pydantic schema validation configs
│   ├── capture.py                     # Resilient FFmpeg stdout capture
│   ├── cv_worker.py                   # Hailo NPU / OpenCV contour processor
│   ├── event_engine.py                # Polygon Ray Casting and Line intersections math
│   └── mqtt_publisher.py              # Mutual TLS AWS IoT client with SQLite backing
├── deployment/
│   ├── edgevision.service             # Hardened sandboxed systemd service descriptor
│   ├── ansible_deploy.yml             # Automatic provisioning playbook
│   └── wireguard_peer.conf            # Secure split-tunnel VPN mesh interface
└── cloud/
    ├── backend_api.py                 # FastAPI Cloud Backend router endpoints
    └── transcoder_lambda.py           # AWS Lambda trigger for serverless HLS transcoding
```

---

## 🔌 Hardware Configuration

- **Host Controller:** Raspberry Pi 5 (4-core Broadcom BCM2712 ARM64 @ 2.4 GHz, 8 GB RAM).
- **Accelerator:** Hailo-8L NPU (13 TOPS, PCIe Gen 2.0 interface, M.2 2230 form factor via M.2 HAT+).
- **Camera:** Reolink RLC-811A 8MP PoE IP Camera (H.264 video feed stream at 1080p, 15 FPS).
- **Storage:** 256GB NVMe SSD (continuous circular recordings log writing buffer).

---

## 🚀 Installation & Automated Provisioning

Deploying the Edge AI system to a fleet of edge cameras is fully automated using the provided Ansible playbook:

### Prerequisites

On your deployment machine, install Ansible:
```bash
pip install ansible
```

Configure your inventory hosts file `/etc/ansible/hosts`:
```ini
[edge_cameras]
camera-east-04 ansible_host=192.168.1.14 ansible_user=ubuntu
camera-dock-01 ansible_host=192.168.1.22 ansible_user=ubuntu
```

### Running the Deployment

To provision, download driver packages, install python environments, compile kernel modules, and launch the service:
```bash
ansible-playbook -i hosts deployment/ansible_deploy.yml --ask-become-pass
```

---

## 🔒 Edge Hardening (systemd Sandbox)

The system daemon configuration `/etc/systemd/system/edgevision.service` applies kernel-level namespaces separation:

- `ProtectSystem=strict` mounts the host root directories as read-only.
- `ProtectHome=true` isolates user folders.
- `PrivateTmp=true` creates an isolated virtual `/tmp` directory.
- `DeviceAllow=/dev/hailo0 rw` limits access strictly to the Hailo NPU PCIe interface.
- `WatchdogSec=30` automatically force-terminates and restarts the daemon if the internal processing loop hangs for over 30 seconds.

---

## 📡 Cloud API Gateway Endpoint Specification

The FastAPI backend exposes the following REST endpoints for client dashboard operations:

| Method | Path | Description | Authentication |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/devices` | Returns a list of active edge cameras and telemetry | Cognito JWT |
| `GET` | `/api/v1/events` | Queries spatial alerts history from DynamoDB | Cognito JWT |
| `GET` | `/api/v1/events/{id}/clip` | Generates a secure pre-signed S3 download URL (Expires: 1h) | Cognito JWT |
| `POST` | `/api/v1/devices/{id}/config` | Publishes updated zone configurations OTA via AWS IoT | Cognito JWT |

---

## 📊 Performance Benchmarks & Field Results

Continuous telemetry recording over a 30-day testing period returned the following parameters:

- **Host CPU Savings:** Offloading inference to the Hailo NPU reduces average Pi 5 CPU utilization from **98% (CPU-bound bottleneck)** down to **45.2%**.
- **Inference Latency Profile:** **38.2 ms** average inference time per frame (INT8 Quantized YOLOv8n network).
- **Resilience:** Auto-recovered **100% of RTSP stream disconnects** (average recovery time of **4.8 seconds**).
- **End-to-End latency:** Events are recorded, trimmed, uploaded, transcoded on AWS Lambda, and referenced in the database within **22.4 seconds** of the physical incident trigger.
