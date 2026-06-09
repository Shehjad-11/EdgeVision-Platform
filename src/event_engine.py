# src/event_engine.py
import time
from typing import Dict, List, Tuple, Any

class EventEngine:
    """
    Stateful geometry processor that runs rule-based checks on target track coordinates.
    Computes polygon containment (ray-casting) and vector boundary crossings.
    Runs on the CPU at extremely low latency (sub-millisecond).
    """
    def __init__(self, zones_config: Any):
        # Extract zones and line coordinates from configuration schemas
        self.polygons = zones_config.polygons
        self.lines = zones_config.lines
        self.dwell_limits = zones_config.dwell_limits
        
        # Track loitering durations: {track_id: enter_timestamp}
        self.dwell_states: Dict[int, float] = {}
        # Track last coordinates to calculate directional intersection: {track_id: last_centroid}
        self.history_states: Dict[int, List[Tuple[float, float]]] = {}

    def update(self, confirmed_tracks: List[Any], current_time: float) -> List[Dict[str, Any]]:
        triggered_events = []
        active_ids = set()

        for track in confirmed_tracks:
            track_id = track.track_id
            active_ids.add(track_id)
            
            # Fetch target bounding box coordinates and estimate bottom-center centroid (contact point)
            # tlbr: [top, left, bottom, right]
            tlbr = track.tlbr
            centroid = (float((tlbr[1] + tlbr[3]) / 2.0), float(tlbr[2]))
            
            # Update trajectory segment history
            if track_id not in self.history_states:
                self.history_states[track_id] = [centroid]
            else:
                self.history_states[track_id].append(centroid)
                if len(self.history_states[track_id]) > 5:
                    self.history_states[track_id].pop(0)

            # 1. Evaluate Polygon Dwell Violations
            for zone_name, polygon in self.polygons.items():
                is_inside = self.point_in_polygon(centroid, polygon)
                
                if is_inside:
                    if track_id not in self.dwell_states:
                        self.dwell_states[track_id] = current_time
                        triggered_events.append({
                            "type": f"ZONE_ENTRY_{zone_name.upper()}",
                            "track_id": track_id,
                            "timestamp": current_time,
                            "bbox": [float(x) for x in tlbr]
                        })
                    else:
                        dwell_time = current_time - self.dwell_states[track_id]
                        limit = self.dwell_limits.get(zone_name, 0.0)
                        if limit > 0 and dwell_time >= limit:
                            triggered_events.append({
                                "type": f"DWELL_LIMIT_EXCEEDED_{zone_name.upper()}",
                                "track_id": track_id,
                                "dwell_time": float(dwell_time),
                                "timestamp": current_time,
                                "bbox": [float(x) for x in tlbr]
                            })
                else:
                    # Target is outside. If it was inside previously, log an exit alert
                    if track_id in self.dwell_states:
                        entry_time = self.dwell_states.pop(track_id)
                        triggered_events.append({
                            "type": f"ZONE_EXIT_{zone_name.upper()}",
                            "track_id": track_id,
                            "dwell_time": float(current_time - entry_time),
                            "timestamp": current_time,
                            "bbox": [float(x) for x in tlbr]
                        })

            # 2. Evaluate Line Crossing Intersections
            if len(self.history_states[track_id]) >= 2:
                last_pos = self.history_states[track_id][-2]
                curr_pos = self.history_states[track_id][-1]
                
                for line_name, line_seg in self.lines.items():
                    if len(line_seg) >= 2:
                        p1, p2 = line_seg[0], line_seg[1]
                        if self.check_line_intersection(last_pos, curr_pos, p1, p2):
                            triggered_events.append({
                                "type": f"LINE_CROSSING_{line_name.upper()}",
                                "track_id": track_id,
                                "timestamp": current_time,
                                "bbox": [float(x) for x in tlbr]
                            })

        # Purge deleted/expired tracking IDs from states
        expired_ids = set(self.dwell_states.keys()) - active_ids
        for exp_id in expired_ids:
            self.dwell_states.pop(exp_id, None)
            
        expired_histories = set(self.history_states.keys()) - active_ids
        for exp_id in expired_histories:
            self.history_states.pop(exp_id, None)

        return triggered_events

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
        """
        Ray-Casting containment check. Tests intersections of a horizontal line
        extending from point to infinity with the polygon edges.
        Runs in O(N) where N is the number of vertices.
        """
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False
            
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    @staticmethod
    def check_line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], 
                                q1: List[float], q2: List[float]) -> bool:
        """
        Vector intersection calculations. Checks if line segment p1->p2
        intersects with line segment q1->q2.
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        # Segment p1-p2 intersects segment q1-q2 if and only if:
        # ccw(p1, q1, q2) != ccw(p2, q1, q2) AND ccw(p1, p2, q1) != ccw(p1, p2, q2)
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
