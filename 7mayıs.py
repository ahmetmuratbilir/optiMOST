import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# --- YAPILANDIRMA ---
VIDEO_PATH = "C:/Users/asus/Downloads/is_etudu.mov" 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Değişkenler
current_data = {"A": 0, "B": 0, "G": 0, "P": 0}
analiz_raporu = []
adim_sayisi = 0
sol_ayak_onceki_x = None
duraklama_sayaci = 0
video_akis_aktif = False 
olcum_baslatildi = False 

def adim_puanla(sayi):
    if sayi < 1.0: return 1
    if 1.0 <= sayi <= 2.2: return 3
    if 2.2 < sayi <= 4.5: return 6
    return 10

def ekran_boyutlandir(img, target_height=800):
    if img is None: return None
    h, w = img.shape[:2]
    ratio = target_height / h
    new_w = int(w * ratio)
    return cv2.resize(img, (new_w, target_height))

def otomatik_kaydet_ve_sifirla(saniye):
    global adim_sayisi, current_data, duraklama_sayaci, sol_ayak_onceki_x
    tmu = (current_data["A"] + current_data["B"] + current_data["G"] + current_data["P"]) * 10
    
    analiz_raporu.append({
        "Saniye": round(saniye, 2),
        "Dizin": f"A{current_data['A']} B{current_data['B']} G{current_data['G']} P{current_data['P']}",
        "TMU": tmu
    })
    print(f"OTOMATİK KAYIT: A{current_data['A']} B{current_data['B']} G{current_data['G']} P{current_data['P']}")

    # SIFIRLAMA
    adim_sayisi = 0
    sol_ayak_onceki_x = None
    duraklama_sayaci = 0
    current_data = {"A": 0, "B": 0, "G": 0, "P": 0}

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

cv2.namedWindow("MOST Etut Paneli", cv2.WINDOW_AUTOSIZE)

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
                sol_ayak_x = lm[mp_pose.PoseLandmark.LEFT_ANKLE].x
                shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y
                wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
                knee_y = lm[mp_pose.PoseLandmark.RIGHT_KNEE].y
                el_hizi = abs(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x - lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x)

                if sol_ayak_onceki_x is not None:
                    if abs(sol_ayak_x - sol_ayak_onceki_x) > 0.045:
                        adim_sayisi += 0.25
                sol_ayak_onceki_x = sol_ayak_x
                current_data["A"] = adim_puanla(adim_sayisi)

                if wrist_y > knee_y: current_data["B"] = max(current_data["B"], 6)
                elif abs(shoulder_y - hip_y) < 0.16: current_data["B"] = max(current_data["B"], 3)

                if el_hizi < 0.025 and wrist_y > 0.40:
                    duraklama_sayaci += 1
                    if duraklama_sayaci > 5:
                        if current_data["G"] == 0: current_data["G"] = 1
                        else: current_data["P"] = 1
                        otomatik_kaydet_ve_sifirla(mevcut_saniye)
                elif el_hizi > 0.05:
                    duraklama_sayaci = 0

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        mevcut_saniye = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    display_frame = ekran_boyutlandir(frame)
    if display_frame is not None:
        cv2.rectangle(display_frame, (10, 10), (420, 160), (0, 0, 0), -1) 
        
        status_txt = "ANALIZ: AKTIF" if olcum_baslatildi else "ANALIZ: BEKLEMEDE ('a')"
        status_col = (0, 255, 0) if olcum_baslatildi else (0, 0, 255)
        
        cv2.putText(display_frame, status_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)
        cv2.putText(display_frame, f"Sure: {round(mevcut_saniye, 2)} sn", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"MOST: A{current_data['A']} B{current_data['B']} G{current_data['G']} P{current_data['P']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Adim Sayaci: {round(adim_sayisi,1)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("MOST Etut Paneli", display_frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'): video_akis_aktif = True
    elif key == ord('p'): video_akis_aktif = False
    elif key == ord('a'): 
        olcum_baslatildi = True
        adim_sayisi = 0
        sol_ayak_onceki_x = None
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- KAYIT SÜRECİ ---
if len(analiz_raporu) > 0:
    print(f"\nİşlem Bitti. Toplam {len(analiz_raporu)} satır kayıt var.")
    op = input("Operatör Adı: ")
    vr = input("Vardiya: ")
    st = input("İstasyon No: ")
    ds = input("Excel Dosya İsmi: ")
    
    df = pd.DataFrame(analiz_raporu)
    df["Operator"] = op
    df["Vardiya"] = vr
    df["Istasyon"] = st
    df["Tarih"] = time.strftime('%d.%m.%Y')
    
    if not ds.endswith(".xlsx"): ds += ".xlsx"
    df.to_excel(ds, index=False)
    
    db = "tum_analizler_veritabani.csv"
    if not os.path.isfile(db):
        df.to_csv(db, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(db, mode='a', index=False, header=False, encoding='utf-8-sig')
    print(f"\nVeriler '{ds}' ve '{db}' dosyalarına kaydedildi.")
else:
    print("Kaydedilecek hareket bulunamadı.")