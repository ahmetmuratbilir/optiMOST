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

def draw_hud(frame, fsm, isg_status, fps):
    """Sağ üst ve sol üst HUD panellerini daha kompakt ve şık çizer."""
    h, w = frame.shape[:2]
    
    # 1. Sol Üst MOST İş Etüdü Paneli (Kompakt boyut)
    cv2.rectangle(frame, (10, 10), (250, 145), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 145), (255, 255, 255), 1)
    
    cv2.putText(frame, "MOST IS ETUDU PANELI", (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dongu: {fsm.cycle_number} | Durum: {fsm.state}", (18, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    recipe_step = fsm.recipe[fsm.current_recipe_idx] if fsm.current_recipe_idx < len(fsm.recipe) else "Bitti"
    cv2.putText(frame, f"Hedef: {recipe_step}", (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1, cv2.LINE_AA)
    
    tmu = fsm.current_cycle_steps
    tmu_reach = round(tmu.get("Reach", 0.0) * 27.8, 1)
    tmu_grasp = round(tmu.get("Grasp", 0.0) * 27.8, 1)
    tmu_move = round(tmu.get("Move", 0.0) * 27.8, 1)
    tmu_place = round(tmu.get("Place", 0.0) * 27.8, 1)
    tmu_return = round(tmu.get("Return", 0.0) * 27.8, 1)
    total_tmu = round(tmu_reach + tmu_grasp + tmu_move + tmu_place + tmu_return, 1)
    
    cv2.putText(frame, f"Uzan:{tmu_reach} | Kav:{tmu_grasp} | Tas:{tmu_move}", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Yerl:{tmu_place} | Donus:{tmu_return}", (18, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Toplam: {total_tmu} TMU | FPS: {round(fps, 1)}", (18, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Sağ Üst İSG (KKD) Paneli (Kompakt boyut)
    cv2.rectangle(frame, (w - 190, 10), (w - 10, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 190, 10), (w - 10, 110), (255, 255, 255), 1)
    cv2.putText(frame, "ISG KKD KONTROLU", (w - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    
    y_offset = 48
    for item, status in isg_status.items():
        text = f"{item}: {'TAMAM' if status else 'EKSIK'}"
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(frame, text, (w - 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        y_offset += 20

    # 3. Alt Ortadaki Uyarı Paneli
    if fsm.active_warning or fsm.sequence_error:
        warn_text = fsm.active_warning
        color = (0, 0, 255) if (fsm.sequence_error or "Hata" in warn_text) else (0, 255, 255)
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h - 50), (w - 10, h - 10), color, 1)
        cv2.putText(frame, warn_text, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

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
    
    cv2.namedWindow("MOST & ISG Entegre Analiz Paneli", cv2.WINDOW_NORMAL)
    
    print("\n--- ANALİZ BAŞLATILDI ---")
    print("'q' tuşuna basarak analizi bitirebilir ve Excel raporunu alabilirsiniz.\n")
    
    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        
        # 1. MediaPipe El Takibi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        hand_detected = False
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # El etiketini al (Sol veya Sağ)
                hand_label = results.multi_handedness[idx].classification[0].label
                
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
                
        # 2. YOLO İSG Denetimi (Kare Atlama - 15 Karede bir çalışır)
        frame_counter += 1
        if frame_counter % 15 == 0:
            if yolo_model is not None:
                # YOLO Çıkarımı
                yolo_results = yolo_model(frame, verbose=False)
                
                # İSG Kuralları (Burada basit bir COCO veya custom model eşleştirmesi simüle edilir)
                # Eğer custom modelimiz varsa kask/yelek sınıflarını ararız.
                # Standart yolov8n.pt ise, sadece 'person' arar, KKD'leri 'True' varsayar.
                helmet_found = False
                vest_found = False
                gloves_found = False
                
                # Çıktı sınıflarını kontrol et
                names = yolo_model.names
                for r in yolo_results:
                    for box in r.boxes:
                        cls_name = names[int(box.cls[0])].lower()
                        # Eğer kask, yelek, eldiven gibi sınıflar modelde tanımlıysa
                        if "helmet" in cls_name or "kask" in cls_name:
                            helmet_found = True
                        if "vest" in cls_name or "yelek" in cls_name:
                            vest_found = True
                        if "glove" in cls_name or "eldiven" in cls_name:
                            gloves_found = True
                            
                # Eğer standart yolov8n ise ve kişi varsa, ekipmanları simüle et (veya eldiveni hand_detected'dan al)
                if "person" in names.values() and not any(k in names.values() for k in ["helmet", "kask", "vest", "yelek"]):
                    # Simülasyon Modu: Personel algılandığında ekipmanları simüle et
                    person_found = any(names[int(box.cls[0])].lower() == "person" for r in yolo_results for box in r.boxes)
                    if person_found:
                        isg_status["Kask"] = True
                        isg_status["Yelek"] = True
                        # Eldiveni MediaPipe el tespitiyle ilişkilendir
                        isg_status["Eldiven"] = hand_detected
                    else:
                        isg_status["Kask"] = False
                        isg_status["Yelek"] = False
                        isg_status["Eldiven"] = False
                else:
                    # Custom Model Modu:
                    isg_status["Kask"] = helmet_found
                    isg_status["Yelek"] = vest_found
                    isg_status["Eldiven"] = gloves_found
            else:
                # Simülasyon Modu (YOLO Modeli yüklü değilse)
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
        
        # HUD ve Uyarıları çiz
        draw_hud(frame, fsm, isg_status, fps)
        
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
