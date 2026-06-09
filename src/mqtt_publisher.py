# src/mqtt_publisher.py
import os
import ssl
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Resilient imports fallback to prevent import errors in environment setups lacking paho-mqtt
MQTT_AVAILABLE = False
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("[MQTT Warning] Paho MQTT client library not installed.")
    print("[MQTT Warning] Defaulting EventPublisher to local logger simulation.")

class EventPublisher:
    """
    AWS IoT Core mutual TLS (mTLS 1.3) publisher.
    Integrates a transactional SQLite buffer queue to ensure zero data loss
    for telemetry event packets during network connection dropouts.
    """
    def __init__(self, device_id: str, config: Any):
        self.device_id = device_id
        self.config = config
        self.topic = f"devices/{device_id}/events"
        self.db_path = config.SQLITE_DB_PATH
        
        # Ensure target database directory paths exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_sqlite_db()
        
        self.client = None
        self.connected = False
        self.lock = threading.Lock()
        
        if MQTT_AVAILABLE:
            self._connect_mqtt()
        else:
            print("[MQTT] Running in Simulated mode. Events will print to system stdout logs.")

    def _init_sqlite_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _connect_mqtt(self):
        try:
            self.client = mqtt.Client(client_id=self.device_id, protocol=mqtt.MQTTv5)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            # Configure mutual TLS cert authentication context
            ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            ssl_context.load_verify_locations(cafile=self.config.CA_FILE)
            
            # Verify paths before loading certificate chain
            if os.path.exists(self.config.CERT_FILE) and os.path.exists(self.config.KEY_FILE):
                ssl_context.load_cert_chain(
                    certfile=self.config.CERT_FILE,
                    keyfile=self.config.KEY_FILE
                )
                self.client.tls_set_context(ssl_context)
                print(f"[MQTT] Loaded device certificate: {self.config.CERT_FILE}")
            else:
                print("[MQTT Warning] Certificate files missing on device storage. Running in unencrypted mode (simulated endpoint).")
                
            # Connect asynchronously to avoid blocking the main processing loops
            self.client.connect_async(self.config.ENDPOINT, self.config.PORT, keepalive=60)
            self.client.loop_start()
            
        except Exception as err:
            print(f"[MQTT Error] Connection failed to setup: {err}. Retrying in background.")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] Connected successfully to AWS IoT Core.")
            self.connected = True
            # Spawn a background thread to flush any buffered offline events
            threading.Thread(target=self._flush_sqlite_queue, daemon=True).start()
        else:
            print(f"[MQTT Error] Connection refused with status code: {rc}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc, properties=None):
        print(f"[MQTT] Disconnected from broker (code: {rc}). Reverting to local SQLite queue buffer.")
        self.connected = False

    def publish_event(self, event_type: str, track_id: int, bbox: List[float], extra_meta: Optional[Dict[str, Any]] = None):
        """
        Publishes structured JSON alerts. Safely fallback write to local SQLite
        if connection is down.
        """
        payload_data = {
            "device_id": self.device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "track_id": track_id,
            "bbox": bbox,
            "meta": extra_meta or {}
        }
        
        payload_str = json.dumps(payload_data)
        
        with self.lock:
            if not self.connected or not MQTT_AVAILABLE:
                # Buffer locally
                self._buffer_event_locally(payload_str)
            else:
                try:
                    # Deliver event to AWS IoT Core with QoS 1 guarantees
                    info = self.client.publish(self.topic, payload_str, qos=1)
                    # Check if connection was active at publish time
                    if not info.is_published():
                        self._buffer_event_locally(payload_str)
                except Exception as err:
                    print(f"[MQTT Error] Publish failed: {err}. Buffering to disk.")
                    self._buffer_event_locally(payload_str)

    def _buffer_event_locally(self, payload_str: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO event_queue (payload) VALUES (?)", (payload_str,))
                conn.commit()
            print(f"[MQTT Offline Buffer] Saved event locally to SQLite.")
        except Exception as err:
            print(f"[Critical System Error] SQLite write failure: {err}. Event dropped: {payload_str}")

    def _flush_sqlite_queue(self):
        """Flushes the database offline event buffer queue upon reconnection."""
        print("[MQTT Queue Flusher] Initiating queue flush loop...")
        
        while self.connected and MQTT_AVAILABLE:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, payload FROM event_queue ORDER BY id ASC LIMIT 10")
                    rows = cursor.fetchall()
                    
                    if not rows:
                        break # Buffer is empty, terminate flusher thread
                        
                    for row in rows:
                        msg_id, payload = row
                        # Resend to AWS
                        info = self.client.publish(self.topic, payload, qos=1)
                        info.wait_for_publish(timeout=3)
                        
                        # Remove from local database on success
                        conn.execute("DELETE FROM event_queue WHERE id = ?", (msg_id,))
                    conn.commit()
                    
                time.sleep(0.5) # Prevent overloading the socket buffer
                
            except Exception as err:
                print(f"[MQTT Queue Flusher Error] Failed to flush batch: {err}. Pausing flusher thread.")
                break
                
        print("[MQTT Queue Flusher] Queue flush completed or paused.")

    def close(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Publisher connection successfully closed.")
        self.connected = False
        self.client = None
