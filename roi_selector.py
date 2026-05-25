import cv2
import json
import os
import sys

CONFIG_FILE = "workspace_config.json"

# Global Değişkenler
points = []
rois = {}
current_frame = None
display_frame = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Konfigürasyon okuma hatası: {e}")
    return {"rois": {}, "assembly_recipe": [], "video_path": 0}

def save_config(config_data):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"\n[BAŞARILI] Konfigürasyon kaydedildi: {CONFIG_FILE}")
    except Exception as e:
        print(f"Konfigürasyon kaydetme hatası: {e}")

def mouse_callback(event, x, y, flags, param):
    global points, display_frame, current_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Sol tıklama ile nokta ekle
        points.append((x, y))
        print(f"Nokta eklendi: ({x}, {y})")
        update_display()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Sağ tıklama ile çokgeni kapat ve isim ver
        if len(points) >= 3:
            update_display()
            cv2.imshow("ROI Editor", display_frame)
            cv2.waitKey(100) # Görüntüyü tazele
            
            roi_name = get_roi_name_popup()
            if roi_name:
                rois[roi_name] = list(points)
                print(f"Bolge kaydedildi: '{roi_name}' -> Koordinatlar: {points}")
            else:
                print("Gecersiz isim. Bolge iptal edildi.")
            points = []
            update_display()
        else:
            print("Cokgen olusturmak icin en az 3 nokta gereklidir.")

def get_roi_name_popup():
    """Tkinter ile native pop-up acarak terminal kilitlemesini ve cokmeleri onler."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring("Bolge Isimlendirme", "Lutfen bu bolgeye bir isim verin:\n(Ornek: Box 1, Box 2, Assembly Area, Home Area)")
        root.destroy()
        return name.strip() if name else None
    except Exception as e:
        print(f"Arayuz pop-up hatasi ({e}), terminalden giris bekleniyor...")
        try:
            return input("\nLutfen bolge ismini girin: ").strip()
        except Exception:
            # Stdin kapaliysa otomatik isimlendir, cokmeyi onle
            return "Box_" + str(len(rois) + 1)

def update_display():
    global display_frame, current_frame, points, rois
    if current_frame is None:
        return
        
    display_frame = current_frame.copy()
    h, w = display_frame.shape[:2]
    
    # Bilgi/Kullanım Paneli Çizimi (OpenCV Türkçe karakter destegi olmadigi icin ASCII uyumlu yazildi)
    cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
    instructions = [
        "Sol Klik: Nokta Ekle  |  Sag Klik: Poligonu Kapat & Isimlendir",
        "'s': Kaydet ve Cik  |  'c': Cizimi Sifirla  |  'q': Iptal Et ve Cik"
    ]
    cv2.putText(display_frame, instructions[0], (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_frame, instructions[1], (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Önceden tanımlanmış bölgeleri çiz
    for name, pts in rois.items():
        # Poligonu çiz
        import numpy as np
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(display_frame, [pts_arr], True, (0, 255, 0), 2)
        
        # Etiket yazdır
        moments = cv2.moments(pts_arr)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = pts[0][0], pts[0][1]
            
        cv2.putText(display_frame, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
    # Anlık çizilen noktaları çiz
    for i, pt in enumerate(points):
        cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(display_frame, points[i-1], pt, (0, 0, 255), 1)

def main():
    global current_frame, display_frame, points, rois
    
    config = load_config()
    video_path = config.get("video_path", 0)
    rois = config.get("rois", {})
    
    # Video/Kamera bağlantısı
    print(f"Kaynak açılıyor: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened() and video_path != 0:
        print("[UYARI] Video dosyası açılamadı, varsayılan web kamerası açılıyor.")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("[HATA] Kamera veya video kaynağı başlatılamadı.")
        sys.exit(1)
        
    # Birkaç kare atlayarak görüntünün oturmasını sağla
    for _ in range(10):
        cap.read()
        
    ret, frame = cap.read()
    if not ret:
        print("[HATA] Görüntü okunamadı.")
        sys.exit(1)
        
    # Ekran boyutlandırma (Gerektiğinde yüksekliği 720'ye ölçekle)
    h, w = frame.shape[:2]
    if h > 720:
        ratio = 720.0 / h
        frame = cv2.resize(frame, (int(w * ratio), 720))
        
    current_frame = frame
    update_display()
    
    cv2.namedWindow("ROI Editor")
    cv2.setMouseCallback("ROI Editor", mouse_callback)
    
    print("\n--- ROI SEÇİCİ BAŞLATILDI ---")
    print("1. Alan tanımlamak için sol klikle köşeleri seçin.")
    print("2. Poligonu kapatmak için son noktayı koyduktan sonra sağ tıklayın ve terminalden ismi girin.")
    print("3. Çizimi kaydetmek için 's' tuşuna basın.")
    
    while True:
        cv2.imshow("ROI Editor", display_frame)
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('s'):
            # Kaydetme ve Reçete Yapılandırma
            if not rois:
                print("[UYARI] Kaydedilecek bölge çizilmedi.")
                continue
                
            config["rois"] = rois
            
            # Reçete girişi
            print("\n--- MONTAJ RECETESI TANIMLAMA ---")
            print("Mevcut Bolgeler:", list(rois.keys()))
            
            recipe_input = get_recipe_popup(list(rois.keys()))
            
            if recipe_input:
                recipe = [r.strip() for r in recipe_input.split(",") if r.strip() in rois]
            else:
                recipe = []
            
            if recipe:
                config["assembly_recipe"] = recipe
                print(f"Recete kaydedildi: {recipe}")
            else:
                # Varsayılan olarak tüm bölgeleri ekle
                config["assembly_recipe"] = list(rois.keys())
                print(f"Gecersiz veya bos recete. Varsayilan sira atandi: {config['assembly_recipe']}")
                
            save_config(config)
            break
            
        elif key == ord('c'):
            # Temizleme
            points = []
            rois = {}
            print("Tum bolgeler temizlendi.")
            update_display()
            
        elif key == ord('q') or key == 27:
            print("Degisiklikler kaydedilmeden cikildi.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

def get_recipe_popup(available_rois):
    """Tkinter ile pop-up acarak montaj recetesi girilmesini saglar."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        prompt = f"Mevcut Bolgeler: {available_rois}\n\nMontaj sirasini virgul ile ayirarak yazin:\n(Ornek: Box 1, Box 2, Assembly Area)"
        recipe_input = simpledialog.askstring("Montaj Recetesi", prompt)
        root.destroy()
        return recipe_input
    except Exception as e:
        print(f"Recete pop-up hatasi ({e}), terminalden giris bekleniyor...")
        try:
            return input("Montaj sirasini girin: ").strip()
        except Exception:
            return None

if __name__ == "__main__":
    main()
