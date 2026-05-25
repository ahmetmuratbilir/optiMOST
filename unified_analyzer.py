import cv2
import json
import os
import sys
import time
import threading
import queue
import numpy as np
import pandas as pd
from fsm import MOSTStateMachine

CONFIG_FILE = "workspace_config.json"
EXPORT_EXCEL = "MOST_Entegre_Analiz_Raporu.xlsx"
EXPORT_CSV = "MOST_Entegre_Analiz_Raporu.csv"

# YOLO Entegrasyonu için Hazırlık
yolo_available = False
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    print("[UYARI] 'ultralytics' paketi kurulu değil. YOLO İSG denetimi simülasyon modunda çalışacak.")

class VideoStream:
    """Kamera veya videodan kareleri gecikmesiz okumak için Thread yapısı."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=3)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            # Kuyruk doluysa eski kareyi çıkar, yenisini ekle (gecikmeyi önler)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
        self.cap.release()

    def read(self):
        if self.stopped or self.q.empty():
            return None
        return self.q.get()

    def stop(self):
        self.stopped = True

def draw_roi_polygons(frame, rois, active_target_name):
    """Tanımlı tüm poligonları ve isimlerini ekrana çizer."""
    for name, pts in rois.items():
        pts_arr = np.array(pts, dtype=np.int32)
        # Aktif hedef bölgeyi mavi, diğerlerini sarı çiz
        color = (255, 0, 0) if name == active_target_name else (0, 255, 255)
        cv2.polylines(frame, [pts_arr], True, color, 2)
        
        # Etiket için merkez noktayı bul
        m = cv2.moments(pts_arr)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = pts[0][0], pts[0][1]
            
        cv2.putText(frame, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

def draw_hud(frame, fsm, isg_status, fps, elbow_angle=None):
    """MOST, İSG ve Ergonomi verilerini ekranın solunda şeffaf dikey bir panelde birleştirir."""
    h, w = frame.shape[:2]
    
    # 1. Sol Üst Konum ve Boyutlar (Genişlik: 190, Yükseklik: 245)
    x1, y1 = 10, 10
    x2, y2 = 200, 255
    
    # Şeffaf Arka Plan Paneli
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
    alpha = 0.30  # %30 şeffaflık
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Panel İnce Çerçevesi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    
    # FSM Durumuna Göre Dinamik Sol Kenar Akış Şeridi (Accent Bar)
    state_colors = {
        "IDLE": (120, 120, 120),       # Gri
        "REACH": (255, 255, 0),       # Camgöbeği (Cyan)
        "GRASP": (0, 165, 255),       # Turuncu
        "MOVE": (255, 0, 0),          # Mavi
        "PLACE": (0, 255, 0),         # Yeşil
        "RETURNING_HOME": (0, 255, 255) # Sarı
    }
    accent_color = state_colors.get(fsm.state, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x1 + 3, y2), accent_color, -1)
    
    # --- 1. MOST İŞ ETÜDÜ BÖLÜMÜ ---
    cv2.putText(frame, "MOST IS ETUDU", (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dongu:{fsm.cycle_number} | {fsm.state}", (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    recipe_step = fsm.recipe[fsm.current_recipe_idx] if fsm.current_recipe_idx < len(fsm.recipe) else "Bitti"
    cv2.putText(frame, f"Hedef: {recipe_step}", (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1, cv2.LINE_AA)
    
    tmu = fsm.current_cycle_steps
    tmu_reach = round(tmu.get("Reach", 0.0) * 27.8, 1)
    tmu_grasp = round(tmu.get("Grasp", 0.0) * 27.8, 1)
    tmu_move = round(tmu.get("Move", 0.0) * 27.8, 1)
    tmu_place = round(tmu.get("Place", 0.0) * 27.8, 1)
    tmu_return = round(tmu.get("Return", 0.0) * 27.8, 1)
    total_tmu = round(tmu_reach + tmu_grasp + tmu_move + tmu_place + tmu_return, 1)
    
    cv2.putText(frame, f"U:{tmu_reach} K:{tmu_grasp} T:{tmu_move}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y:{tmu_place} D:{tmu_return} TMU", (18, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Toplam: {total_tmu} TMU", (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Bölücü Çizgi 1
    cv2.line(frame, (15, 104), (195, 104), (80, 80, 80), 1)
    
    # --- 2. İSG KONTROLÜ BÖLÜMÜ ---
    cv2.putText(frame, "ISG KKD DURUMU", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)
    
    k_color = (0, 255, 0) if isg_status.get("Kask", True) else (0, 0, 255)
    v_color = (0, 255, 0) if isg_status.get("Yelek", True) else (0, 0, 255)
    e_color = (0, 255, 0) if isg_status.get("Eldiven", True) else (0, 0, 255)
    
    cv2.putText(frame, "Kask", (18, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.33, k_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Yelek", (70, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.33, v_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Eldiven", (125, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.33, e_color, 1, cv2.LINE_AA)
    
    # Bölücü Çizgi 2
    cv2.line(frame, (15, 142), (195, 142), (80, 80, 80), 1)
    
    # --- 3. ERGONOMİ BÖLÜMÜ ---
    cv2.putText(frame, "ERGONOMI", (18, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 150, 0), 1, cv2.LINE_AA)
    
    if elbow_angle is not None:
        if elbow_angle > 150:
            angle_color = (0, 0, 255) # Kırmızı
            strain_txt = "YUKSEK"
        elif elbow_angle > 120:
            angle_color = (0, 255, 255) # Sarı
            strain_txt = "ORTA"
        else:
            angle_color = (0, 255, 0) # Yeşil
            strain_txt = "DUSUK"
        cv2.putText(frame, f"Aci: {int(elbow_angle)} deg", (18, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.35, angle_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gerginlik: {strain_txt}", (18, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, angle_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Aci: Tespit yok", (18, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "Gerginlik: --", (18, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        
    # Bölücü Çizgi 3
    cv2.line(frame, (15, 194), (195, 194), (80, 80, 80), 1)
    
    # --- 4. KAVRAMA / PINCH KONTROL DETAYI ---
    cv2.putText(frame, "KAVRAMA KONTROL", (18, 207), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Oran: {fsm.last_pinch_dist:.2f} / {fsm.pinch_threshold:.2f}", (18, 221), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Bölücü Çizgi 4
    cv2.line(frame, (15, 230), (195, 230), (80, 80, 80), 1)
    
    # --- 5. TAZELENME HIZI (FPS) ---
    cv2.putText(frame, f"Sistem: {round(fps, 1)} FPS", (18, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Alt Ortadaki Uyarı Paneli (Aynı kalacak)
    if fsm.active_warning or fsm.sequence_error:
        warn_text = fsm.active_warning
        color = (0, 0, 255) if (fsm.sequence_error or "Hata" in warn_text) else (0, 255, 255)
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), color, 1)
        cv2.putText(frame, warn_text, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def calculate_angle(a, b, c):
    """Üç nokta arasındaki açıyı derece olarak hesaplar."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"[HATA] Yapılandırma dosyası bulunamadı: {CONFIG_FILE}")
        print("Lütfen önce 'roi_selector.py' ile alanları tanımlayın.")
        sys.exit(1)
        
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        
    rois = config_data.get("rois", {})
    recipe = config_data.get("assembly_recipe", [])
    video_path = config_data.get("video_path", 0)
    
    if not rois or not recipe:
        print("[HATA] Yapılandırma dosyasındaki ROIs veya reçete eksik.")
        sys.exit(1)
        
    # FSM ve MediaPipe Başlatma
    fsm = MOSTStateMachine(config_data)
    
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # YOLO Model Yükleme
    yolo_model = None
    if yolo_available:
        model_path = config_data.get("model_path", "yolov8n.pt")
        print(f"YOLO Modeli yükleniyor: {model_path}")
        try:
            yolo_model = YOLO(model_path)
        except Exception as e:
            print(f"YOLO yüklenemedi, fallback modele geçiliyor: {e}")
            try:
                yolo_model = YOLO("yolov8n.pt")
            except Exception:
                yolo_model = None
                
    # Kamera / Video Akışı
    stream = VideoStream(video_path).start()
    time.sleep(1.0) # Akışın başlaması için bekle
    
    # İSG Durum Değişkenleri
    isg_status = {"Kask": True, "Yelek": True, "Eldiven": True}
    
    # FPS Hesaplama Değişkenleri
    prev_time = time.time()
    fps = 0.0
    frame_counter = 0
    elbow_angle = None
    cached_pose_results = None
    
    cv2.namedWindow("MOST & ISG Entegre Analiz Paneli", cv2.WINDOW_NORMAL)
    
    print("\n--- ANALİZ BAŞLATILDI ---")
    print("'q' tuşuna basarak analizi bitirebilir ve Excel raporunu alabilirsiniz.\n")
    
    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_counter += 1
        
        # 1. MediaPipe El ve İskelet (Pose) Takibi
        hand_results = hands.process(frame_rgb)
        
        # Pose modelini 2 karede bir çalıştırarak CPU yükünü hafifletiyoruz (Frame Skip)
        if frame_counter % 2 == 0:
            cached_pose_results = pose.process(frame_rgb)
        
        hand_detected = False
        active_hand_label = "Right"
        
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # El etiketini al (Sol veya Sağ)
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                active_hand_label = hand_label
                
                # Eklem koordinatlarını piksel değerine dönüştür
                pixel_landmarks = []
                for lm in hand_landmarks.landmark:
                    pixel_landmarks.append({
                        'x': int(lm.x * w),
                        'y': int(lm.y * h)
                    })
                    
                # Durum makinesine besle
                fsm.process_frame(pixel_landmarks, hand_label, time.time())
                hand_detected = True
                
                # Arayüze el eklemlerini çiz
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Aktif elin kol eklem açılarını hesapla ve çiz
        elbow_angle = None
        if hand_detected and cached_pose_results and cached_pose_results.pose_landmarks:
            lm_pose = cached_pose_results.pose_landmarks.landmark
            is_left_hand = (active_hand_label == "Left")
            try:
                if is_left_hand:
                    # Sol Kol: Omuz(11), Dirsek(13), Bilek(15)
                    sh = [lm_pose[11].x, lm_pose[11].y]
                    el = [lm_pose[13].x, lm_pose[13].y]
                    wr = [lm_pose[15].x, lm_pose[15].y]
                else:
                    # Sağ Kol: Omuz(12), Dirsek(14), Bilek(16)
                    sh = [lm_pose[12].x, lm_pose[12].y]
                    el = [lm_pose[14].x, lm_pose[14].y]
                    wr = [lm_pose[16].x, lm_pose[16].y]
                
                # Açıyı derece olarak hesapla
                elbow_angle = calculate_angle(sh, el, wr)
                
                # İskelet çizgilerini çiz (Kalın turkuaz çizgi + kırmızı eklem yuvarlakları)
                pts = [
                    (int(sh[0] * w), int(sh[1] * h)),
                    (int(el[0] * w), int(el[1] * h)),
                    (int(wr[0] * w), int(wr[1] * h))
                ]
                cv2.line(frame, pts[0], pts[1], (255, 255, 0), 4, cv2.LINE_AA)
                cv2.line(frame, pts[1], pts[2], (255, 255, 0), 4, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            except Exception:
                pass
                
        # 2. YOLO İSG Denetimi (Kare Atlama - 15 Karede bir çalışır)
        if frame_counter % 15 == 0:
            if yolo_model is not None:
                # YOLO Çıkarımı
                yolo_results = yolo_model(frame, verbose=False)
                
                # İSG Kuralları (COCO 'person' fallback veya custom model)
                helmet_found = False
                vest_found = False
                gloves_found = False
                
                names = yolo_model.names
                for r in yolo_results:
                    for box in r.boxes:
                        cls_name = names[int(box.cls[0])].lower()
                        if "helmet" in cls_name or "kask" in cls_name:
                            helmet_found = True
                        if "vest" in cls_name or "yelek" in cls_name:
                            vest_found = True
                        if "glove" in cls_name or "eldiven" in cls_name:
                            gloves_found = True
                            
                if "person" in names.values() and not any(k in names.values() for k in ["helmet", "kask", "vest", "yelek"]):
                    person_found = any(names[int(box.cls[0])].lower() == "person" for r in yolo_results for box in r.boxes)
                    if person_found:
                        isg_status["Kask"] = True
                        isg_status["Yelek"] = True
                        isg_status["Eldiven"] = hand_detected
                    else:
                        isg_status["Kask"] = False
                        isg_status["Yelek"] = False
                        isg_status["Eldiven"] = False
                else:
                    isg_status["Kask"] = helmet_found
                    isg_status["Yelek"] = vest_found
                    isg_status["Eldiven"] = gloves_found
            else:
                isg_status["Kask"] = True
                isg_status["Yelek"] = True
                isg_status["Eldiven"] = hand_detected
                
        # 3. İSG Durumunu FSM'ye Bildir (Çevrim ihlal takibi için)
        fsm.check_isg_violation(all(isg_status.values()))
        
        # 4. FPS ve HUD Çizimi
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        
        # Poligonları çiz
        active_target_name = recipe[fsm.current_recipe_idx] if fsm.current_recipe_idx < len(recipe) else ""
        draw_roi_polygons(frame, rois, active_target_name)
        
        # HUD ve Uyarıları çiz (Dirsek açısını buraya besliyoruz)
        draw_hud(frame, fsm, isg_status, fps, elbow_angle)
        
        # Görüntüyü göster
        cv2.imshow("MOST & ISG Entegre Analiz Paneli", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    # Kaynakları serbest bırak ve raporu yazdır
    stream.stop()
    cv2.destroyAllWindows()
    
    # 5. Raporlama (Excel & CSV)
    if fsm.reported_cycles:
        df = pd.DataFrame(fsm.reported_cycles)
        
        # Excel kaydetme (openpyxl yoksa csv fallback)
        try:
            df.to_excel(EXPORT_EXCEL, index=False)
            print(f"\n[RAPOR] Analiz tamamlandı. Excel raporu kaydedildi: '{EXPORT_EXCEL}'")
        except Exception as e:
            print(f"\n[UYARI] Excel kaydedilemedi (openpyxl eksik olabilir): {e}")
            
        try:
            df.to_csv(EXPORT_CSV, index=False, encoding='utf-8-sig')
            print(f"[RAPOR] CSV raporu kaydedildi: '{EXPORT_CSV}'")
        except Exception as e:
            print(f"[HATA] CSV kaydedilemedi: {e}")
            
        # Rapor özetini ekrana bas
        print("\n=== ANALİZ ÖZET TABLOSU ===")
        print(df.to_string(index=False))
    else:
        print("\n[UYARI] Analiz bitirildi fakat kaydedilmiş iş çevrimi bulunamadı.")

if __name__ == "__main__":
    main()
