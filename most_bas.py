import cv2
import mediapipe as mp
import pandas as pd
import time

# --- YAPILANDIRMA ---
VIDEO_PATH = "C:/Users/asus/Downloads/is_etudu.mov" 
FRAME_STABILITY = 4   # Elin nesne üzerinde durma kararlılığı
START_DELAY = 30      # Videonun oturması için ilk 1 saniyeyi atla

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Değişkenler
current_phase_data = {"A": 0, "B": 0, "G": 0}
analiz_raporu = []
adim_sayisi = 0
sol_ayak_onceki_x = None
# Başlangıçta True: İlk boru bırakılana kadar kayıt almaz
kavrama_kilidi = True 
duraklama_sayaci = 0
baslangic_saniyesi = 0 

def faz_paketle_ve_kaydet(saniye):
    """A, B ve G değerlerini BasicMOST formatında paketler."""
    global adim_sayisi, current_phase_data, kavrama_kilidi
    if not kavrama_kilidi:
        # MOST TMU Hesaplama: (A+B+G) * 10
        faz_tmu = (current_phase_data["A"] + current_phase_data["B"] + current_phase_data["G"]) * 10
        analiz_raporu.append({
            "Video Saniyesi": round(saniye, 2),
            "İş Adımı": "Alma Fazı (Get)",
            "Dizin": f"A{current_phase_data['A']} B{current_phase_data['B']} G{current_phase_data['G']}",
            "Toplam TMU": faz_tmu
        })
        print(f"KAYDEDİLDİ: {round(saniye, 2)}. sn | A{current_phase_data['A']} B{current_phase_data['B']} G{current_phase_data['G']}")
        kavrama_kilidi = True

cap = cv2.VideoCapture(VIDEO_PATH)
# Orijinal boyutu korumak için pencere ayarı
cv2.namedWindow("MOST Analiz ve Dogrulama", cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    mevcut_saniye = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if cap.get(cv2.CAP_PROP_POS_FRAMES) < START_DELAY: continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # --- A PARAMETRESİ (Adım Takibi) ---
        sol_ayak_x = lm[mp_pose.PoseLandmark.LEFT_ANKLE].x
        if sol_ayak_onceki_x is not None:
            if abs(sol_ayak_x - sol_ayak_onceki_x) > 0.045:
                if current_phase_data["A"] == 0:
                    baslangic_saniyesi = mevcut_saniye
                adim_sayisi += 0.2 
        sol_ayak_onceki_x = sol_ayak_x

        # A İndeksleri
        if adim_sayisi < 1: current_phase_data["A"] = max(current_phase_data["A"], 1)
        elif 1 <= adim_sayisi <= 2: current_phase_data["A"] = max(current_phase_data["A"], 3)
        elif 3 <= adim_sayisi <= 4: current_phase_data["A"] = max(current_phase_data["A"], 6)

        # --- B PARAMETRESİ (Eğilme) ---
        shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y
        wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
        knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y

        if wrist_y > knee_y: current_phase_data["B"] = max(current_phase_data["B"], 6)
        elif shoulder_y > hip_y - 0.16: current_phase_data["B"] = max(current_phase_data["B"], 3)

        # --- G PARAMETRESİ (Kavrama ve Kilit Kontrolü) ---
        el_hizi = abs(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x - lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
        
        # El stand (kelepçe/ip) hizasında ve yavaşsa
        if el_hizi < 0.012 and wrist_y > 0.45:
            duraklama_sayaci += 1
            if duraklama_sayaci > FRAME_STABILITY:
                current_phase_data["G"] = 1
                faz_paketle_ve_kaydet(baslangic_saniyesi if baslangic_saniyesi != 0 else mevcut_saniye)
        else:
            duraklama_sayaci = 0
            # El standdan uzaklaştığında kilidi aç ve hafızayı sıfırla
            if el_hizi > 0.04 or wrist_y < 0.4:
                if kavrama_kilidi:
                    current_phase_data = {"A": 0, "B": 0, "G": 0}
                    adim_sayisi = 0
                    baslangic_saniyesi = 0
                    kavrama_kilidi = False
                    print(f"--- {round(mevcut_saniye, 2)}. sn: Yeni Alma İçin Hazır ---")

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Ekrana Bilgi Yaz
    cv2.putText(frame, f"Sn: {round(mevcut_saniye, 1)} | A:{current_phase_data['A']} B:{current_phase_data['B']}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("MOST Analiz ve Dogrulama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Excel çıktısını al
if analiz_raporu:
    pd.DataFrame(analiz_raporu).to_excel("MOST_Saniye_Kontrollu_Rapor13.xlsx", index=False)
    print("\nAnaliz Bitti! 'MOST_Saniye_Kontrollu_Rapor13.xlsx' dosyası oluşturuldu.")