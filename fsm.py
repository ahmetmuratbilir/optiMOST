# pip install opencv-python mediapipe ultralytics openpyxl
import cv2
import json
import os
import time
import numpy as np
from collections import deque

SEC_TO_TMU = 27.8

def point_in_polygon(point, polygon):
    """Noktanın poligon içinde olup olmadığını kontrol eder (OpenCV pointPolygonTest)."""
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return dist >= 0

class EMAFilter:
    """Jitter'ı önlemek için Tek Üstel Düzleştirme (EMA) filtresi."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.value = None

    def filter(self, new_val):
        if self.value is None:
            self.value = np.array(new_val, dtype=np.float32)
        else:
            self.value = self.alpha * np.array(new_val, dtype=np.float32) + (1.0 - self.alpha) * self.value
        return self.value

class MOSTTracker:
    def __init__(self, config_source="workspace_config.json"):
        self.config_source = config_source
        self.load_config()
        self.reset_system()

    def load_config(self):
        if isinstance(self.config_source, dict):
            self.config = self.config_source
        elif os.path.exists(self.config_source):
            with open(self.config_source, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Stations listesini oku
        self.stations = {}
        for st in self.config.get("stations", []):
            self.stations[st["name"]] = st["polygon"]

        self.recipe = self.config.get("recipe", ["Box 1", "Box 2", "Assembly Area"])
        
        # Grasp Esik Degerleri
        grasp_cfg = self.config.get("grasp", {})
        self.pinch_threshold = grasp_cfg.get("pinch_threshold", 0.28)
        self.release_threshold = grasp_cfg.get("release_threshold", 0.40)
        self.velocity_threshold = grasp_cfg.get("velocity_threshold", 8.0)
        self.grasp_confirm_frames = grasp_cfg.get("grasp_confirm_frames", 4)
        self.release_confirm_frames = grasp_cfg.get("release_confirm_frames", 3)

    def reset_system(self):
        self.state = "IDLE"
        self.state_start_time = time.time()
        
        # Recete kontrolu
        self.current_recipe_idx = 0
        self.cycle_number = 1
        self.sequence_error = False
        self.isg_violation_in_cycle = False
        
        # Eklem filtreleri
        self.filters = {
            "wrist": EMAFilter(0.6),
            "middle_mcp": EMAFilter(0.6),
            "index_tip": EMAFilter(0.6),
            "thumb_tip": EMAFilter(0.6)
        }
        
        # Hiz filtresi
        self.prev_index_tip = None
        self.velocity_history = deque(maxlen=5)
        
        # Debounce sayaclari
        self.grasp_confirm_counter = 0
        self.place_confirm_counter = 0
        
        # Metrik takibi
        self.last_pinch_ratio = 0.0
        self.last_velocity = 0.0
        self.active_station = "None"
        
        # Zaman kayitlari
        self.current_cycle_steps = {
            "Reach": 0.0,
            "Grasp": 0.0,
            "Move": 0.0,
            "Place": 0.0,
            "Return": 0.0
        }
        self.reported_cycles = []

    def reset_cycle(self, next_cycle=True):
        self.state = "IDLE"
        self.state_start_time = time.time()
        self.current_recipe_idx = 0
        self.grasp_confirm_counter = 0
        self.place_confirm_counter = 0
        self.sequence_error = False
        self.active_station = "None"
        self.current_cycle_steps = {"Reach": 0.0, "Grasp": 0.0, "Move": 0.0, "Place": 0.0, "Return": 0.0}
        if next_cycle:
            self.isg_violation_in_cycle = False

    def update(self, hand_landmarks, frame_shape):
        """
        hand_landmarks: 21 adet landmark içeren liste
        frame_shape: (height, width)
        """
        h, w = frame_shape
        current_time = time.time()
        duration = current_time - self.state_start_time
        
        # 1. Koordinatları piksele donustur ve filtrele
        def get_pixel_coord(idx):
            lm = hand_landmarks[idx]
            # Eger koordinatlar 0-1 arasindaysa pixel'e cevir, yoksa direk kullan
            if 0.0 <= lm['x'] <= 1.01 and 0.0 <= lm['y'] <= 1.01:
                x = lm['x'] * w
                y = lm['y'] * h
            else:
                x = lm['x']
                y = lm['y']
            return np.array([x, y], dtype=np.float32)

        wrist = get_pixel_coord(0)
        thumb_tip = get_pixel_coord(4)
        index_tip = get_pixel_coord(8)
        middle_mcp = get_pixel_coord(9)
        
        # EMA Filtresi uygula
        f_wrist = self.filters["wrist"].filter(wrist)
        f_thumb_tip = self.filters["thumb_tip"].filter(thumb_tip)
        f_index_tip = self.filters["index_tip"].filter(index_tip)
        f_middle_mcp = self.filters["middle_mcp"].filter(middle_mcp)
        
        # 2. Normalize Edilmis Pinch Orani Hesabi
        # Wrist ile Middle Finger MCP arasi mesafe
        hand_size = np.linalg.norm(f_wrist - f_middle_mcp)
        pinch_dist = np.linalg.norm(f_thumb_tip - f_index_tip)
        pinch_ratio = pinch_dist / max(hand_size, 15.0)
        self.last_pinch_ratio = pinch_ratio
        
        # 3. Hiz Hesaplama (INDEX_FINGER_TIP yer degisimi)
        if self.prev_index_tip is not None:
            displacement = np.linalg.norm(f_index_tip - self.prev_index_tip)
            self.velocity_history.append(displacement)
        else:
            self.velocity_history.append(0.0)
        self.prev_index_tip = f_index_tip.copy()
        
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        self.last_velocity = avg_velocity
        
        # FSM Kosullari
        is_pinching = pinch_ratio < self.pinch_threshold
        is_releasing = pinch_ratio > self.release_threshold
        is_still = avg_velocity < self.velocity_threshold
        
        hand_pos = f_index_tip
        
        # Reçete Hedef Tanımlamaları
        target_station_name = self.recipe[self.current_recipe_idx] if self.current_recipe_idx < len(self.recipe) else "Assembly Area"
        target_polygon = self.stations.get(target_station_name, [])
        assembly_polygon = self.stations.get("Assembly Area", [])
        home_polygon = self.stations.get("Home Area", [])
        
        # Durum Makinesi
        if self.state == "IDLE":
            # Home Area varsa el oradan cikinca basla, yoksa Assembly Area disina cikinca basla
            if home_polygon:
                if not point_in_polygon(hand_pos, home_polygon):
                    self.state = "REACH"
                    self.state_start_time = current_time
            else:
                if not point_in_polygon(hand_pos, assembly_polygon):
                    self.state = "REACH"
                    self.state_start_time = current_time
                    
        elif self.state == "REACH":
            # Kutuya girildi mi?
            # Eger sira hatali bir kutuya girilmisse tespit et
            for name, poly in self.stations.items():
                if name != target_station_name and name != "Assembly Area" and name != "Home Area":
                    if point_in_polygon(hand_pos, poly):
                        self.sequence_error = True
            
            if point_in_polygon(hand_pos, target_polygon):
                # Hız filtresi: el yavaslayana kadar REACH durumunda kalır
                if is_still:
                    self.current_cycle_steps["Reach"] += duration
                    self.state = "GRASP"
                    self.state_start_time = current_time
                    self.grasp_confirm_counter = 0
                    self.active_station = target_station_name
                    
        elif self.state == "GRASP":
            # Kutu icinde pinch (kavrama) bekleniyor
            if point_in_polygon(hand_pos, target_polygon):
                if is_pinching:
                    self.grasp_confirm_counter += 1
                    if self.grasp_confirm_counter >= self.grasp_confirm_frames:
                        self.current_cycle_steps["Grasp"] += duration
                        self.state = "MOVE"
                        self.state_start_time = current_time
                        self.place_confirm_counter = 0
                else:
                    self.grasp_confirm_counter = max(0, self.grasp_confirm_counter - 1)
            else:
                # Eger kavramadan kutuyu terk ederse REACH fazina geri don
                self.state = "REACH"
                self.state_start_time = current_time
                self.grasp_confirm_counter = 0
                
        elif self.state == "MOVE":
            # Montaj alanina ulasilinca PLACE baslar
            if point_in_polygon(hand_pos, assembly_polygon):
                self.current_cycle_steps["Move"] += duration
                self.state = "PLACE"
                self.state_start_time = current_time
                self.place_confirm_counter = 0
                self.active_station = "Assembly Area"
                
        elif self.state == "PLACE":
            # Montaj alaninda parmaklar acilirsa veya el montaj alanini terk ederse
            if point_in_polygon(hand_pos, assembly_polygon):
                if is_releasing:
                    self.place_confirm_counter += 1
                else:
                    self.place_confirm_counter = max(0, self.place_confirm_counter - 1)
            else:
                # El montaj alanini terk ederse direk birakildi kabul et
                self.place_confirm_counter = self.release_confirm_frames
                
            if self.place_confirm_counter >= self.release_confirm_frames:
                self.current_cycle_steps["Place"] += duration
                self.current_recipe_idx += 1
                
                # Recete tamamlandi mi?
                if self.current_recipe_idx >= len(self.recipe) - 1:
                    if home_polygon:
                        self.state = "RETURNING_HOME"
                        self.state_start_time = current_time
                        self.active_station = "Home Area"
                    else:
                        self.save_completed_cycle()
                        self.reset_cycle(next_cycle=True)
                else:
                    # Siradaki kutuya gec
                    self.state = "REACH"
                    self.state_start_time = current_time
                    self.grasp_confirm_counter = 0
                    self.place_confirm_counter = 0
                    
        elif self.state == "RETURNING_HOME":
            if point_in_polygon(hand_pos, home_polygon):
                self.current_cycle_steps["Return"] += duration
                self.save_completed_cycle()
                self.reset_cycle(next_cycle=True)
                
        return self.state

    def save_completed_cycle(self):
        duration_sec = round(sum(self.current_cycle_steps.values()), 2)
        total_tmu = round(duration_sec * SEC_TO_TMU, 1)
        
        cycle_data = {
            "cycle_no": self.cycle_number,
            "station": ", ".join(self.recipe[:-1]),
            "state_sequence": "IDLE-REACH-GRASP-MOVE-PLACE" + ("-RETURN-IDLE" if self.stations.get("Home Area") else "-IDLE"),
            "duration_sec": duration_sec,
            "tmu": total_tmu,
            "sequence_ok": "False" if self.sequence_error else "True"
        }
        self.reported_cycles.append(cycle_data)
        print(f"\n[DONGU TAMAMLANDI] Dongu {self.cycle_number} -> Sure: {duration_sec} sn | TMU: {total_tmu} | Sira Hatasi: {self.sequence_error}")
        self.cycle_number += 1

    def get_metrics(self):
        # Debounce progress string
        if self.state == "GRASP":
            debounce_str = f"{self.grasp_confirm_counter}/{self.grasp_confirm_frames}"
        elif self.state == "PLACE":
            debounce_str = f"{self.place_confirm_counter}/{self.release_confirm_frames}"
        else:
            debounce_str = "0"
            
        cycle_time = sum(self.current_cycle_steps.values())
        tmu = cycle_time * SEC_TO_TMU
        
        return {
            "state": self.state,
            "pinch_ratio": self.last_pinch_ratio,
            "velocity": self.last_velocity,
            "debounce_count": debounce_str,
            "cycle_time_sec": cycle_time,
            "tmu": tmu,
            "sequence_error": self.sequence_error,
            "active_station": self.active_station
        }

    def save_report(self, path):
        import csv
        # Klasoru olustur
        os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["cycle_no", "station", "state_sequence", "duration_sec", "tmu", "sequence_ok"])
            for c in self.reported_cycles:
                writer.writerow([
                    c["cycle_no"],
                    c["station"],
                    c["state_sequence"],
                    c["duration_sec"],
                    c["tmu"],
                    c["sequence_ok"]
                ])
        print(f"[RAPOR] CSV kaydedildi: {path}")
