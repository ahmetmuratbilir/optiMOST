import cv2
import mediapipe as mp
import pandas as pd
import time

# --- YAPILANDIRMA ---
VIDEO_PATH = "C:/Users/asus/Downloads/is_etudu.mov" 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Değişkenler
current_phase_data = {"A": 0, "B": 0, "G": 0}
analiz_raporu = []
adim_sayisi = 0
sol_ayak_onceki_x = None
kavrama_kilidi = True 
duraklama_sayaci = 0
baslangic_saniyesi = 0 

# Kontrol Mekanizması Değişkenleri
video_akis_aktif = False 
olcum_baslatildi = False 

def faz_paketle_ve_kaydet(saniye):
    global adim_sayisi, current_phase_data, kavrama_kilidi, olcum_baslatildi
    if not kavrama_kilidi:
        # --- A PARAMETRE FİLTRESİ (Gönderdiğin Hassas Ayar) ---
        if adim_sayisi < 1.1: 
            current_phase_data["A"] = 1
        elif 1.1 <= adim_sayisi <= 2.1:
            current_phase_data["A"] = 3
        else:
            current_phase_data["A"] = 6
        
        faz_tmu = (current_phase_data["A"] + current_phase_data["B"] + current_phase_data["G"]) * 10
        analiz_raporu.append({
            "Video Saniyesi": round(saniye, 2),
            "Dizin": f"A{current_phase_data['A']} B{current_phase_data['B']} G{current_phase_data['G']}",
            "Toplam TMU": faz_tmu,
            "Kayıt Saati": time.strftime('%H:%M:%S')
        })
        print(f"KAYDEDİLDİ: {round(saniye, 2)}. sn | A{current_phase_data['A']} B{current_phase_data['B']} G{current_phase_data['G']}")
        
        # SIFIRLAMA VE BEKLEME MODU
        adim_sayisi = 0
        kavrama_kilidi = True
        olcum_baslatildi = False # Kayıt sonrası yeni 'a' tuşuna kadar durur

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read() # İlk kareyi dondurarak başlat

if not ret:
    print("Video bulunamadı!")
    exit()

cv2.namedWindow("MOST Etut Paneli", cv2.WINDOW_AUTOSIZE)

print("\n--- KONTROL PANELİ ---")
print("'s': VİDEO OYNAT | 'p': DURAKLAT | 'a': ÖLÇÜMÜ BAŞLAT | 'q': KAYDET VE ÇIK\n")

while cap.isOpened():
    if video_akis_aktif:
        ret, frame = cap.read()
        if not ret: break
        mevcut_saniye = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if olcum_baslatildi:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # --- A PARAMETRESİ (0.05 Eşik ve 0.2 Artış) ---
                sol_ayak_x = lm[mp_pose.PoseLandmark.LEFT_ANKLE].x
                if sol_ayak_onceki_x is not None:
                    if abs(sol_ayak_x - sol_ayak_onceki_x) > 0.05:
                        if current_phase_data["A"] == 0:
                            baslangic_saniyesi = mevcut_saniye
                        adim_sayisi += 0.2
                sol_ayak_onceki_x = sol_ayak_x

                # Görsel anlık A güncelleme
                if adim_sayisi < 1: current_phase_data["A"] = 1
                elif 1 <= adim_sayisi <= 2: current_phase_data["A"] = 3
                else: current_phase_data["A"] = 6

                # --- B PARAMETRESİ (0.16 Eşik) ---
                shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y
                wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
                knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y

                if wrist_y > knee_y: current_phase_data["B"] = max(current_phase_data["B"], 6)
                elif shoulder_y > hip_y - 0.16: # Gönderdiğin hassas değer
                    current_phase_data["B"] = max(current_phase_data["B"], 3)
                else: current_phase_data["B"] = 0

                # --- G PARAMETRESİ (0.02 Hız ve 0.42 Mesafe) ---
                el_hizi = abs(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x - lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
                
                if el_hizi < 0.02 and wrist_y > 0.42: 
                    duraklama_sayaci += 1
                    if duraklama_sayaci > 4: # Stabilite korundu
                        current_phase_data["G"] = 1
                        faz_paketle_ve_kaydet(baslangic_saniyesi if baslangic_saniyesi != 0 else mevcut_saniye)
                else:
                    duraklama_sayaci = 0
                    if el_hizi > 0.05 or wrist_y < 0.4:
                        if kavrama_kilidi:
                            kavrama_kilidi = False

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        mevcut_saniye = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # --- GÖRSEL PANEL (DEĞİŞMEDİ) ---
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (10, 10), (450, 115), (0, 0, 0), -1)
    
    v_durum = "VIDEO: OYNATILIYOR" if video_akis_aktif else "VIDEO: DURDURULDU"
    a_durum = "ANALIZ: KAYITTA..." if olcum_baslatildi else "ANALIZ: BEKLEMEDE ('a')"
    
    cv2.putText(display_frame, v_durum, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, a_durum, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Sure: {round(mevcut_saniye, 2)} sn", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(display_frame, f"MOST: A{current_phase_data['A']} B{current_phase_data['B']} G{current_phase_data['G']}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("MOST Etut Paneli", display_frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'): video_akis_aktif = True
    elif key == ord('p'): video_akis_aktif = False
    elif key == ord('a'): olcum_baslatildi = True
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- KAYIT VE İSİMLENDİRME ---
if analiz_raporu:
    dosya_adi = input("\nExcel dosyasının ismi ne olsun?: ")
    if not dosya_adi.endswith(".xlsx"): dosya_adi += ".xlsx"
    pd.DataFrame(analiz_raporu).to_excel(dosya_adi, index=False)
    print(f"\nDosya '{dosya_adi}' başarıyla oluşturuldu.")