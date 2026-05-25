# pip install opencv-python mediapipe ultralytics openpyxl
import cv2
import json
import os
import sys
import time
import argparse
import threading
import queue
import numpy as np
from fsm import MOSTTracker, point_in_polygon

yolo_available = False
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    print("[UYARI] 'ultralytics' paketi yuklu degil. YOLO ISG denetimi simule edilecek.")

def calculate_angle(a, b, c):
    """Uc nokta arasindaki aciyi derece olarak hesaplar."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def draw_hud(frame, metrics, isg_status, fps, recipe_step, tracker, left_elbow_angle=None, right_elbow_angle=None):
    h, w = frame.shape[:2]
    
    # 1. Sol Ust Konum ve Boyutlar (Genislik: 190, Yukseklik: 195)
    x1, y1 = 10, 10
    x2, y2 = 200, 195
    
    # Seffaf Arka Plan Paneli
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 10, 10), -1)
    alpha = 0.30
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    
    # Panel Ince Cercevesi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    
    # FSM Durumuna Gore Dinamik Accent Bar
    state = metrics["state"]
    state_colors = {
        "IDLE": (120, 120, 120),       # Gri
        "REACH": (255, 255, 0),       # Camgobegi (Cyan)
        "GRASP": (0, 165, 255),       # Turuncu
        "MOVE": (255, 0, 0),          # Mavi
        "PLACE": (0, 255, 0),         # Yesil
        "RETURNING_HOME": (0, 255, 255) # Sari
    }
    accent_color = state_colors.get(state, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x1 + 3, y2), accent_color, -1)
    
    # --- YAZILAR ---
    # 1. FSM Durumu + FPS
    cv2.putText(frame, f"{state} | {round(fps, 1)} FPS", (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    
    # 2. Pinch Orani ve Progress Bar
    pinch_ratio = metrics["pinch_ratio"]
    cv2.putText(frame, f"Pinch: {pinch_ratio:.2f}", (18, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Progress Bar Border
    cv2.rectangle(frame, (18, 49), (185, 55), (150, 150, 150), 1)
    # Fill (oran 0.0 - 1.0 arasinda sinirlandirilir)
    fill_width = int(min(max(pinch_ratio, 0.0), 1.0) * 165)
    # Pinch esigi altindaysa yesil, yoksa turuncu ciz
    fill_color = (0, 255, 0) if pinch_ratio < tracker.pinch_threshold else (0, 165, 255)
    cv2.rectangle(frame, (19, 50), (19 + fill_width, 54), fill_color, -1)
    
    # 3. Anlik Hiz (Velocity)
    cv2.putText(frame, f"Hiz: {metrics['velocity']:.1f} px/fr", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 4. Debounce Sayaci
    cv2.putText(frame, f"Debounce: {metrics['debounce_count']}", (18, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 5. Sure + TMU
    cv2.putText(frame, f"Sure: {metrics['cycle_time_sec']:.1f}s ({metrics['tmu']:.1f} TMU)", (18, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Bolucu cizgi 1
    cv2.line(frame, (15, 114), (195, 114), (80, 80, 80), 1)
    
    # 6. KKD Badgeleri
    cv2.putText(frame, "ISG KKD DURUMU", (18, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    
    k_color = (0, 255, 0) if isg_status.get("helmet", True) else (0, 0, 255)
    v_color = (0, 255, 0) if isg_status.get("vest", True) else (0, 0, 255)
    e_color = (0, 255, 0) if isg_status.get("glove", True) else (0, 0, 255)
    
    cv2.putText(frame, "Kask", (18, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.33, k_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Yelek", (70, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.33, v_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Eldiven", (125, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.33, e_color, 1, cv2.LINE_AA)
    
    # Bolucu cizgi 2
    cv2.line(frame, (15, 153), (195, 153), (80, 80, 80), 1)
    
    # 7. Çift Kol Ergonomi Değerleri (Sol ve Sag Aci)
    cv2.putText(frame, "ERGONOMI", (18, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "Sol: ", (18, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    x_offset = 18 + cv2.getTextSize("Sol: ", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    
    if left_elbow_angle is not None:
        l_ang_str = f"{int(left_elbow_angle)}"
        l_color = (0, 0, 255) if left_elbow_angle > 150 else ((0, 255, 255) if left_elbow_angle > 120 else (0, 255, 0))
        cv2.putText(frame, l_ang_str, (x_offset, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, l_color, 1, cv2.LINE_AA)
        x_offset += cv2.getTextSize(l_ang_str, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    else:
        cv2.putText(frame, "--", (x_offset, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1, cv2.LINE_AA)
        x_offset += cv2.getTextSize("--", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
        
    cv2.putText(frame, " | Sag: ", (x_offset, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    x_offset += cv2.getTextSize(" | Sag: ", cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    
    if right_elbow_angle is not None:
        r_ang_str = f"{int(right_elbow_angle)}"
        r_color = (0, 0, 255) if right_elbow_angle > 150 else ((0, 255, 255) if right_elbow_angle > 120 else (0, 255, 0))
        cv2.putText(frame, r_ang_str, (x_offset, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, r_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "--", (x_offset, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 150), 1, cv2.LINE_AA)

def draw_station_polygons(frame, stations, active_target_name):
    """Sistemdeki tum bolgeleri farkli renklerde cizer."""
    colors = [
        (0, 255, 0),    # Yesil
        (255, 0, 0),    # Mavi
        (0, 0, 255),    # Kirmizi
        (0, 255, 255),  # Sari
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Camgobegi
    ]
    for idx, (name, pts) in enumerate(stations.items()):
        pts_arr = np.array(pts, dtype=np.int32)
        # Aktif olan hedef bolgeyi parlak ciz, digerlerini standart ciz
        color = (255, 0, 255) if name == active_target_name else colors[idx % len(colors)]
        cv2.polylines(frame, [pts_arr], True, color, 2)
        
        m = cv2.moments(pts_arr)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = pts[0][0], pts[0][1]
            
        cv2.putText(frame, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def export_to_excel(reported_cycles, filename="rapor.xlsx"):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "MOST Analiz Raporu"
    
    headers = ["cycle_no", "station", "state_sequence", "duration_sec", "tmu", "sequence_ok"]
    ws.append(headers)
    
    for c in reported_cycles:
        ws.append([
            c["cycle_no"],
            c["station"],
            c["state_sequence"],
            c["duration_sec"],
            c["tmu"],
            c["sequence_ok"]
        ])
        
    wb.save(filename)
    print(f"[RAPOR] Excel kaydedildi: {filename}")

# --- THREAD FONKSIYONLARI ---

def video_reader(source, frame_queue, stop_event):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Video kaynagi acilamadi: {source}")
        stop_event.set()
        return
        
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video/Kamera akisi sona erdi.")
            break
            
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
        time.sleep(0.01) # CPU'yu rahatlatmak icin
        
    cap.release()

def ai_inference(frame_queue, results_queue, stop_event, tracker, config_data):
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
    
    # YOLO Yukle
    yolo_model = None
    yolo_skip = config_data.get("yolo_skip_frames", 15)
    yolo_path = config_data.get("yolo_model_path", "yolov8n.pt")
    if yolo_available and os.path.exists(yolo_path):
        try:
            yolo_model = YOLO(yolo_path)
            print(f"[YOLO] KKD modeli yuklendi: {yolo_path}")
        except Exception as e:
            print(f"[UYARI] YOLO yuklenirken hata olustu: {e}")
            
    frame_count = 0
    ppe_required = config_data.get("ppe", {"helmet": True, "vest": True, "glove": True})
    isg_status = {k: True for k in ppe_required.keys()}
    
    yolo_boxes = []
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1
        
        # 1. MediaPipe El Tesbiti
        hand_results = hands.process(frame_rgb)
        
        # 2. MediaPipe Pose (2 frame'de bir)
        pose_results = None
        if frame_count % 2 == 0:
            pose_results = pose.process(frame_rgb)
            
        # Akilli El Secimi ve MOST Tracker Guncellemesi
        hand_detected = False
        active_hand_idx = 0
        
        if hand_results.multi_hand_landmarks:
            # Birden fazla el varsa, aktif kutuya en yakin olani sec
            if len(hand_results.multi_hand_landmarks) > 1:
                min_dist = 999999.0
                target_station_name = tracker.recipe[tracker.current_recipe_idx] if tracker.current_recipe_idx < len(tracker.recipe) else "Assembly Area"
                target_poly = tracker.stations.get(target_station_name, [])
                
                if len(target_poly) >= 3:
                    target_center = np.mean(target_poly, axis=0)
                else:
                    target_center = np.array([w / 2, h / 2])
                    
                for idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
                    # Isaret parmagi ucu (8)
                    itip = np.array([hand_lm.landmark[8].x * w, hand_lm.landmark[8].y * h])
                    if point_in_polygon(itip, target_poly):
                        active_hand_idx = idx
                        break
                    dist = np.linalg.norm(itip - target_center)
                    if dist < min_dist:
                        min_dist = dist
                        active_hand_idx = idx
                        
            active_hand = hand_results.multi_hand_landmarks[active_hand_idx]
            
            # FSM koordinat listesine cevir
            lm_list = []
            for lm in active_hand.landmark:
                lm_list.append({'x': lm.x, 'y': lm.y})
                
            tracker.update(lm_list, (h, w))
            hand_detected = True
            
            # Tespit edilen tum elleri ekrana ciz
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                
        # Iskelet (Pose) ve Cift Dirsek Acisi Cizimleri
        left_elbow_angle = None
        right_elbow_angle = None
        pose_pts_left = []
        pose_pts_right = []
        
        if pose_results and pose_results.pose_landmarks:
            lm_pose = pose_results.pose_landmarks.landmark
            
            # Sol kol eklemleri: Omuz(11), Dirsek(13), Bilek(15)
            try:
                sh_l = [lm_pose[11].x, lm_pose[11].y]
                el_l = [lm_pose[13].x, lm_pose[13].y]
                wr_l = [lm_pose[15].x, lm_pose[15].y]
                left_elbow_angle = calculate_angle(sh_l, el_l, wr_l)
                pose_pts_left = [
                    (int(sh_l[0] * w), int(sh_l[1] * h)),
                    (int(el_l[0] * w), int(el_l[1] * h)),
                    (int(wr_l[0] * w), int(wr_l[1] * h))
                ]
            except Exception:
                pass
                
            # Sag kol eklemleri: Omuz(12), Dirsek(14), Bilek(16)
            try:
                sh_r = [lm_pose[12].x, lm_pose[12].y]
                el_r = [lm_pose[14].x, lm_pose[14].y]
                wr_r = [lm_pose[16].x, lm_pose[16].y]
                right_elbow_angle = calculate_angle(sh_r, el_r, wr_r)
                pose_pts_right = [
                    (int(sh_r[0] * w), int(sh_r[1] * h)),
                    (int(el_r[0] * w), int(el_r[1] * h)),
                    (int(wr_r[0] * w), int(wr_r[1] * h))
                ]
            except Exception:
                pass
                
        # 3. YOLO ISG Denetimi
        if yolo_model is not None and frame_count % yolo_skip == 0:
            yolo_results = yolo_model(frame, verbose=False)
            yolo_boxes.clear()
            
            helmet_found = False
            vest_found = False
            glove_found = False
            
            names = yolo_model.names
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names[cls_id].lower()
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    yolo_boxes.append((xyxy, cls_name, conf))
                    
                    if "helmet" in cls_name or "kask" in cls_name:
                        helmet_found = True
                    if "vest" in cls_name or "yelek" in cls_name:
                        vest_found = True
                    if "glove" in cls_name or "eldiven" in cls_name:
                        glove_found = True
                        
            # ISG durumunu guncelle
            if "helmet" in ppe_required:
                isg_status["helmet"] = helmet_found
            if "vest" in ppe_required:
                isg_status["vest"] = vest_found
            if "glove" in ppe_required:
                isg_status["glove"] = glove_found or hand_detected
        elif yolo_model is None:
            # Simule et
            for k in isg_status.keys():
                isg_status[k] = True
            if "glove" in isg_status:
                isg_status["glove"] = hand_detected
                
        # Sonuclari GUI thread'e gonder
        res = {
            "frame": frame,
            "pose_pts_left": pose_pts_left,
            "pose_pts_right": pose_pts_right,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "isg_status": isg_status,
            "yolo_boxes": yolo_boxes,
            "metrics": tracker.get_metrics(),
            "recipe_step": tracker.recipe[tracker.current_recipe_idx] if tracker.current_recipe_idx < len(tracker.recipe) else "Assembly Area"
        }
        
        if results_queue.full():
            try:
                results_queue.get_nowait()
            except queue.Empty:
                pass
        results_queue.put(res)

def main():
    parser = argparse.ArgumentParser(description="MOST & ISG Entegre Analiz Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Video source (dosya yolu veya kamera indexi)")
    args = parser.parse_args()
    
    # Yapilandirmayi oku
    config_path = "workspace_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    else:
        print("[HATA] Yapilandirma dosyasi bulunamadi: workspace_config.json")
        sys.exit(1)
        
    video_source = args.source
    if video_source is None:
        video_source = config_data.get("video_path", 0)
        
    # Sayi ise int'e cevir
    try:
        video_source = int(video_source)
    except ValueError:
        pass
        
    # MOSTTracker baslat
    tracker = MOSTTracker(config_path)
    
    frame_queue = queue.Queue(maxsize=2)
    results_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Thread'leri olustur
    t1 = threading.Thread(target=video_reader, args=(video_source, frame_queue, stop_event), daemon=True)
    t2 = threading.Thread(target=ai_inference, args=(frame_queue, results_queue, stop_event, tracker, config_data), daemon=True)
    
    t1.start()
    t2.start()
    
    prev_time = time.time()
    fps = 0.0
    
    cv2.namedWindow("MOST & ISG Entegre Analiz Paneli", cv2.WINDOW_NORMAL)
    
    print("\n--- CANLI ANALIZ HATLARI BASLATILDI ---")
    print("'q' veya ESC tusuna basarak cikis yapabilir ve raporlarinizi alabilirsiniz.\n")
    
    while not stop_event.is_set():
        try:
            res = results_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        frame = res["frame"]
        h, w = frame.shape[:2]
        
        # FPS Hesabi
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        
        # 1. Pose cizimleri
        if res["pose_pts_left"]:
            pts = res["pose_pts_left"]
            cv2.line(frame, pts[0], pts[1], (0, 255, 255), 3, cv2.LINE_AA) # Sol kol sari
            cv2.line(frame, pts[1], pts[2], (0, 255, 255), 3, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                
        if res["pose_pts_right"]:
            pts = res["pose_pts_right"]
            cv2.line(frame, pts[0], pts[1], (255, 255, 0), 3, cv2.LINE_AA) # Sag kol camgobegi
            cv2.line(frame, pts[1], pts[2], (255, 255, 0), 3, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                
        # 2. YOLO Bounding Box cizimleri
        for box, cls_name, conf in res["yolo_boxes"]:
            x1, y1, x2, y2 = [int(v) for v in box]
            if cls_name in ["helmet", "kask", "vest", "yelek", "glove", "eldiven"]:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
        # 3. Istasyon poligonlarini ciz
        draw_station_polygons(frame, tracker.stations, res["recipe_step"])
        
        # 4. HUD Cizimi
        draw_hud(frame, res["metrics"], res["isg_status"], fps, res["recipe_step"], tracker, res["left_elbow_angle"], res["right_elbow_angle"])
        
        # 5. Alt Ortadaki Rehberlik / Uyarı Paneli
        guidance_text = res["metrics"]["guidance"]
        if guidance_text:
            is_err = res["metrics"]["sequence_error"]
            bg_color = (0, 0, 180) if is_err else (15, 15, 15)
            border_color = (0, 0, 255) if is_err else (0, 255, 255)
            text_color = (255, 255, 255) if is_err else (255, 255, 200)
            
            # Yari seffaf arka plan bandi
            by1 = h - 50
            by2 = h - 15
            bx1 = 10
            bx2 = w - 10
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
            alpha = 0.50
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, 1)
            
            # Ortalanmis yazi
            font_scale = 0.42
            thickness = 1
            text_size = cv2.getTextSize(guidance_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            tx = bx1 + (bx2 - bx1 - text_size[0]) // 2
            ty = by1 + (by2 - by1 + text_size[1]) // 2
            
            cv2.putText(frame, guidance_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            
        cv2.imshow("MOST & ISG Entegre Analiz Paneli", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Kapatma istegi alindi.")
            stop_event.set()
            break
            
    # Kaynaklari serbest birak ve bitir
    stop_event.set()
    cv2.destroyAllWindows()
    
    # Raporlama
    tracker.save_report("rapor.csv")
    export_to_excel(tracker.reported_cycles, "rapor.xlsx")
    print("\n--- ANALIZ SONLANDIRILDI ---")

if __name__ == "__main__":
    main()
