# pip install opencv-python
import cv2
import json
import os
import argparse
import sys
import numpy as np

CONFIG_FILE = "workspace_config.json"

def ask_station_name():
    """Terminalden veya Tkinter popup penceresinden isim sorar."""
    try:
        # Oncelikle terminalden okumaya calis
        print("\n--> Istasyon adini terminale girin ve Enter'a basin.")
        name = input("Istasyon ismi (ornek: Box 1): ").strip()
        if name:
            return name
    except (EOFError, Exception):
        pass

    # Terminal okuma basarisiz/engellenmis ise Tkinter popup kullan
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring("Istasyon Tanimi", "Istasyon ismini girin:", parent=root)
        root.destroy()
        if name:
            return name.strip()
    except Exception as e:
        print(f"[UYARI] Tkinter baslatilamadi: {e}")
        
    return f"Station_{np.random.randint(100, 999)}"

def main():
    parser = argparse.ArgumentParser(description="MOST ROI Selector - Istasyon Sınır Tanımlama Arayuzu")
    parser.add_argument("--source", type=str, default=None, help="Video dosyasi yolu veya kamera indexi (örn: 0)")
    args = parser.parse_args()

    # Config dosyasini oku
    config_data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"[UYARI] Config okunamadi: {e}")

    # Video kaynagini belirle
    video_source = args.source
    if video_source is None:
        video_source = config_data.get("video_path", 0)

    # Kamera index'ini integer'a cevir
    try:
        video_source = int(video_source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[HATA] Video kaynagi acilamadi: {video_source}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[HATA] Ilk frame okunamadi.")
        sys.exit(1)

    clone = frame.copy()
    current_pts = []
    
    # Mevcut stations listesini yukle
    stations = config_data.get("stations", [])
    
    # Renk listesi (Rastgele veya belirgin renkler)
    colors = [
        (0, 255, 0),    # Yesil
        (255, 0, 0),    # Mavi
        (0, 0, 255),    # Kirmizi
        (0, 255, 255),  # Sari
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Camgobegi
    ]

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_pts, clone, frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([x, y])
            # Noktayi ciz
            cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
            if len(current_pts) > 1:
                cv2.line(clone, tuple(current_pts[-2]), tuple(current_pts[-1]), (0, 0, 255), 1)
            cv2.imshow("MOST ROI Selector", clone)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_pts) >= 3:
                # Cokgeni kapat ve ciz
                cv2.line(clone, tuple(current_pts[-1]), tuple(current_pts[0]), (0, 0, 255), 1)
                cv2.imshow("MOST ROI Selector", clone)
                
                # Istasyon ismi sor
                name = ask_station_name()
                if name:
                    stations.append({
                        "name": name,
                        "polygon": current_pts.copy()
                    })
                    print(f"[YENI ALAN] Eklendi: {name} | Noktalar: {current_pts}")
                
                # Temizle ve yeniden ciz
                current_pts.clear()
                redraw()
            else:
                print("[UYARI] Cokgen icin en az 3 nokta gereklidir.")

    def redraw():
        nonlocal clone, frame
        clone = frame.copy()
        for idx, st in enumerate(stations):
            pts = np.array(st["polygon"], dtype=np.int32)
            color = colors[idx % len(colors)]
            
            # Poligonu ciz
            cv2.polylines(clone, [pts], True, color, 2)
            
            # Ismini ciz
            m = cv2.moments(pts)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                cx, cy = st["polygon"][0][0], st["polygon"][0][1]
            cv2.putText(clone, st["name"], (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
        cv2.imshow("MOST ROI Selector", clone)

    cv2.namedWindow("MOST ROI Selector")
    cv2.setMouseCallback("MOST ROI Selector", mouse_callback)
    
    redraw()
    
    print("\n=== KULLANIM KILAVUZU ===")
    print("Sol Tik: Nokta Ekle")
    print("Sag Tik: Cokgeni Kapat ve Adlandir")
    print("'s' tusu: Kaydet ve Cik")
    print("'r' tusu: Son cizilen bölgeyi sil")
    print("'q' tusu: Kaydetmeden cik")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            # stations degerini güncelle
            config_data["stations"] = stations
            config_data["video_path"] = video_source
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"\n[BASARILI] Yeni alanlar kaydedildi: {CONFIG_FILE}")
            break
            
        elif key == ord('r'):
            if stations:
                removed = stations.pop()
                print(f"[GERI ALINDI] Silinen alan: {removed['name']}")
                redraw()
            else:
                print("[UYARI] Silinecek kayitli alan yok.")
                
        elif key == ord('q') or key == 27:
            print("\n[UYARI] Degisiklikler kaydedilmeden cikildi.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
