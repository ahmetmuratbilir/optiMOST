import numpy as np
import cv2
import time

# TMU Sabiti (1 Saniye = 27.8 TMU)
SEC_TO_TMU = 27.8

def point_in_polygon(point, polygon):
    """Noktanın poligon içinde olup olmadığını kontrol eder (OpenCV pointPolygonTest)."""
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    # point: (x, y)
    dist = cv2.pointPolygonTest(pts, (int(point[0]), int(point[1])), False)
    return dist >= 0

class EMAFilter:
    """Jitter'ı önlemek için Tek Üstel Düzleştirme (EMA) filtresi."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def filter(self, new_val):
        if self.value is None:
            self.value = np.array(new_val)
        else:
            self.value = self.alpha * np.array(new_val) + (1 - self.alpha) * self.value
        return self.value

class MOSTStateMachine:
    def __init__(self, config_data):
        self.config = config_data
        self.rois = config_data.get("rois", {})
        self.recipe = config_data.get("assembly_recipe", [])
        
        # Filtreler (Sol ve Sağ el için ayrı koordinat düzleştiriciler)
        self.filters = {
            "left": {
                "index": EMAFilter(), 
                "thumb": EMAFilter(), 
                "wrist": EMAFilter(),
                "mcp": EMAFilter()
            },
            "right": {
                "index": EMAFilter(), 
                "thumb": EMAFilter(), 
                "wrist": EMAFilter(),
                "mcp": EMAFilter()
            }
        }
        
        # FSM Durumları
        self.state = "IDLE" # IDLE, REACH, GRASP, MOVE, PLACE
        self.state_start_time = time.time()
        
        # Çevrim Kontrolü
        self.current_recipe_idx = 0
        self.cycle_number = 1
        self.isg_violation_in_cycle = False
        
        # Anlık Süre / TMU Kayıtları
        self.current_cycle_steps = {
            "Reach": 0.0,
            "Grasp": 0.0,
            "Move": 0.0,
            "Place": 0.0
        }
        
        # Bitmiş Çevrim Raporları
        self.reported_cycles = []
        
        # Kararlılık Sayaçları
        self.grasp_confirm_counter = 0
        self.place_confirm_counter = 0
        
        # Ekran Uyarı Mesajları
        self.sequence_error = False
        self.active_warning = ""

    def reset_cycle(self, next_cycle=True):
        self.state = "IDLE"
        self.state_start_time = time.time()
        self.current_recipe_idx = 0
        self.grasp_confirm_counter = 0
        self.place_confirm_counter = 0
        self.sequence_error = False
        self.active_warning = ""
        self.current_cycle_steps = {"Reach": 0.0, "Grasp": 0.0, "Move": 0.0, "Place": 0.0}
        if next_cycle:
            self.isg_violation_in_cycle = False

    def check_isg_violation(self, safe_status):
        """Çevrim boyunca herhangi bir İSG ihlali oldu mu?"""
        if not safe_status:
            self.isg_violation_in_cycle = True

    def process_frame(self, hand_landmarks, hand_label, current_time):
        """
        hand_landmarks: 21 adet landmark içeren liste (MediaPipe formatında: {'x', 'y'})
        hand_label: "Left" veya "Right"
        """
        if not self.recipe:
            self.active_warning = "Reçete tanımlanmamış!"
            return
            
        # 1. Koordinatları Çıkar ve Filtrele
        wrist = np.array([hand_landmarks[0]['x'], hand_landmarks[0]['y']])
        index_mcp = np.array([hand_landmarks[5]['x'], hand_landmarks[5]['y']])
        index_tip = np.array([hand_landmarks[8]['x'], hand_landmarks[8]['y']])
        thumb_tip = np.array([hand_landmarks[4]['x'], hand_landmarks[4]['y']])
        
        # EMA Filtrelemesi uygula
        side = "left" if hand_label == "Left" else "right"
        f_wrist = self.filters[side]["wrist"].filter(wrist)
        f_index_mcp = self.filters[side]["mcp"].filter(index_mcp)
        f_index_tip = self.filters[side]["index"].filter(index_tip)
        f_thumb_tip = self.filters[side]["thumb"].filter(thumb_tip)
        
        # 2. Dinamik Parmak Açıklık Oranı (Hand Size Normalization)
        hand_size = np.linalg.norm(f_wrist - f_index_mcp)
        pinch_dist = np.linalg.norm(f_index_tip - f_thumb_tip) / max(hand_size, 1e-6)
        
        # Pinch (Kavrama) Koşulları (Ölçeklenmiş oranlar)
        is_pinching = pinch_dist < 0.28
        is_releasing = pinch_dist > 0.40
        
        # El konumu olarak işaret parmağı ucunu baz alalım
        hand_pos = f_index_tip
        
        # Hedef Kutuları Belirle
        target_roi_name = self.recipe[self.current_recipe_idx]
        target_roi = self.rois.get(target_roi_name, [])
        assembly_roi_name = self.recipe[-1] # Genellikle reçetenin son adımı montaj alanıdır
        assembly_roi = self.rois.get(assembly_roi_name, [])
        
        # Diğer kutulardan birine girdi mi? (Sıra kontrolü)
        in_wrong_box = False
        for roi_name, roi_poly in self.rois.items():
            if roi_name != target_roi_name and roi_name != assembly_roi_name:
                if point_in_polygon(hand_pos, roi_poly):
                    in_wrong_box = True
                    self.sequence_error = True
                    self.active_warning = f"Sıra Hatası! Hedef: {target_roi_name}, Girilen: {roi_name}"
                    
        # 3. Durum Makinesi (FSM) Geçişleri
        duration = current_time - self.state_start_time
        
        if self.state == "IDLE":
            self.active_warning = "Başlamak için Kutuya uzanın."
            # El montaj alanından çıkıp hedef kutuya doğru hareket ettiğinde REACH başlar
            if not point_in_polygon(hand_pos, assembly_roi):
                self.state = "REACH"
                self.state_start_time = current_time
                self.sequence_error = False
                self.active_warning = f"{target_roi_name} kutusuna uzanılıyor."
                
        elif self.state == "REACH":
            # Hedef kutuya ulaşıldığında GRASP başlar
            if point_in_polygon(hand_pos, target_roi):
                # Reach süresini kaydet
                self.current_cycle_steps["Reach"] += duration
                self.state = "GRASP"
                self.state_start_time = current_time
                self.grasp_confirm_counter = 0
                self.active_warning = f"{target_roi_name} kutusunda kavrama bekleniyor."
                
        elif self.state == "GRASP":
            # El hedef kutuda ve parmaklar birleştiğinde (pinch) kavrama başlar
            if point_in_polygon(hand_pos, target_roi):
                if is_pinching:
                    self.grasp_confirm_counter += 1
                    if self.grasp_confirm_counter >= 3: # 3 ardışık kare kararlılık
                        self.current_cycle_steps["Grasp"] += duration
                        self.state = "MOVE"
                        self.state_start_time = current_time
                        self.active_warning = f"Parça alındı. {assembly_roi_name} alanına taşınıyor."
                else:
                    self.grasp_confirm_counter = max(0, self.grasp_confirm_counter - 1)
            else:
                # Kutu dışına erken çıkarsa REACH'e geri dön
                self.state = "REACH"
                self.state_start_time = current_time
                self.grasp_confirm_counter = 0
                
        elif self.state == "MOVE":
            # Montaj alanına ulaşıldığında PLACE başlar
            if point_in_polygon(hand_pos, assembly_roi):
                self.current_cycle_steps["Move"] += duration
                self.state = "PLACE"
                self.state_start_time = current_time
                self.place_confirm_counter = 0
                self.active_warning = "Montaj alanına yerleştiriliyor."
                
        elif self.state == "PLACE":
            # Parmaklar açıldığında VEYA el montaj alanını terk ettiğinde bırakma tamamlanır
            if point_in_polygon(hand_pos, assembly_roi):
                if is_releasing:
                    self.place_confirm_counter += 1
            else:
                # El montaj alanını terk ettiyse direkt bırakıldı kabul et
                self.place_confirm_counter = 5 
                
            if self.place_confirm_counter >= 3:
                self.current_cycle_steps["Place"] += duration
                
                # Reçete sırasını ilerlet
                self.current_recipe_idx += 1
                
                # Reçete tamamlandı mı?
                if self.current_recipe_idx >= len(self.recipe) - 1:
                    # Bir tam çevrim (cycle) bitti!
                    self.save_completed_cycle()
                    self.reset_cycle(next_cycle=True)
                else:
                    # Sıradaki kutuya geç
                    self.state = "REACH"
                    self.state_start_time = current_time
                    self.grasp_confirm_counter = 0
                    self.place_confirm_counter = 0

    def save_completed_cycle(self):
        """Tamamlanan montaj çevrimini TMU birimine dönüştürerek rapor listesine kaydeder."""
        tmu_steps = {k: round(v * SEC_TO_TMU, 1) for k, v in self.current_cycle_steps.items()}
        total_tmu = round(sum(tmu_steps.values()), 1)
        
        cycle_data = {
            "Döngü No": self.cycle_number,
            "Uzanma (TMU)": tmu_steps["Reach"],
            "Kavrama (TMU)": tmu_steps["Grasp"],
            "Taşıma (TMU)": tmu_steps["Move"],
            "Yerleştirme (TMU)": tmu_steps["Place"],
            "Toplam (TMU)": total_tmu,
            "İSG İhlali": "Var" if self.isg_violation_in_cycle else "Yok"
        }
        
        self.reported_cycles.append(cycle_data)
        print(f"\n[DÖNGÜ TAMAMLANDI] Döngü {self.cycle_number} -> Toplam: {total_tmu} TMU | İSG: {cycle_data['İSG İhlali']}")
        
        self.cycle_number += 1
