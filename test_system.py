# pip install opencv-python
import numpy as np
import time
from fsm import MOSTTracker

def mock_landmark(wrist_pos, index_tip_pos, thumb_tip_pos):
    """MediaPipe el eklemleri formatında sentetik veri üretir."""
    landmarks = [None] * 21
    # Wrist (0)
    landmarks[0] = {'x': wrist_pos[0], 'y': wrist_pos[1]}
    # Index MCP (5)
    landmarks[5] = {'x': wrist_pos[0], 'y': wrist_pos[1] - 10.0}
    # Thumb Tip (4)
    landmarks[4] = {'x': thumb_tip_pos[0], 'y': thumb_tip_pos[1]}
    # Index Tip (8)
    landmarks[8] = {'x': index_tip_pos[0], 'y': index_tip_pos[1]}
    # Middle MCP (9) - El boyutu normalizasyonu için (50 piksel mesafe)
    landmarks[9] = {'x': wrist_pos[0], 'y': wrist_pos[1] - 50.0}
    return landmarks

def run_tests():
    # 1. Konfigürasyon Kurulumu
    config_data = {
        "stations": [
            {"name": "Box 1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"name": "Box 2", "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
            {"name": "Assembly Area", "polygon": [[10, 20], [20, 20], [20, 30], [10, 30]]}
        ],
        "recipe": ["Box 1", "Box 2", "Assembly Area"],
        "grasp": {
            "pinch_threshold": 0.28,
            "release_threshold": 0.40,
            "velocity_threshold": 8.0,
            "grasp_confirm_frames": 4,
            "release_confirm_frames": 3
        }
    }
    
    fsm = MOSTTracker(config_data)
    
    print("=== ENTEGRE KALİTE KONTROL VE TEST PROGRAMI BAŞLATILDI ===")
    
    # Test 1: IDLE Durumu Doğrulaması
    # El montaj alanında (örneğin 15, 25)
    lm = mock_landmark([15, 28], [15, 25], [15, 55]) # El açık (pinch_dist = 30 / 50 = 0.6)
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"Test 1 - IDLE Kontrolü: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE", "Test 1 başarısız!"
    
    # Test 2: REACH Durumu Geçişi
    # El montaj alanını terk ediyor (örneğin 15, 15)
    lm = mock_landmark([15, 18], [15, 15], [15, 45]) # El açık
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"Test 2 - REACH Kontrolü: Beklenen: REACH, Gerçek: {fsm.state}")
    assert fsm.state == "REACH", "Test 2 başarısız!"
    
    # Test 3: GRASP Durumu Geçişi
    # El Box 1 içine giriyor (örneğin 5, 5)
    lm = mock_landmark([5, 8], [5, 5], [5, 35]) # El açık (kutu içinde kavrama bekleniyor)
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"Test 3 - GRASP Kontrolü: Beklenen: GRASP, Gerçek: {fsm.state}")
    assert fsm.state == "GRASP", "Test 3 başarısız!"
    
    # Test 4: MOVE Durumu Geçişi (Kavrama Doğrulaması)
    # Parmakları kapatıp (pinch_dist = 5 / 50 = 0.1 < 0.28) Box 1'de bekliyoruz
    lm = mock_landmark([5, 8], [5, 5], [5, 10])
    
    # Debounce (4 frame) ve kararlılık kontrolü için 10 kare
    for i in range(10):
        fsm.update(lm, (480, 640))
        
    print(f"Test 4 - MOVE Kontrolü: Beklenen: MOVE, Gerçek: {fsm.state}")
    assert fsm.state == "MOVE", "Test 4 başarısız!"
    
    # Test 5: PLACE Durumu Geçişi
    # El Box 1'den montaj alanına taşıyor (örneğin 15, 25)
    lm = mock_landmark([15, 28], [15, 25], [15, 30]) # Hala parçayı tutuyor
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"Test 5 - PLACE Kontrolü: Beklenen: PLACE, Gerçek: {fsm.state}")
    assert fsm.state == "PLACE", "Test 5 başarısız!"
    
    # Test 6: Reçete Adımı İlerlemesi (PLACE -> REACH (Box 2))
    # Parmakları açıp parçayı bırakıyoruz (pinch_dist = 30 / 50 = 0.6 > 0.40)
    lm = mock_landmark([15, 28], [15, 25], [15, 55])
    for _ in range(10):
        fsm.update(lm, (480, 640))
        
    print(f"Test 6 - Reçete İlerleme Kontrolü: Beklenen: REACH (Box 2 için), Gerçek: {fsm.state}")
    assert fsm.state == "REACH", "Test 6 başarısız!"
    assert fsm.current_recipe_idx == 1, "Reçete dizini ilerlemedi!"
    
    # Test 7: Çevrim Tamamlama Raporlaması
    # İkinci kutuyu (Box 2) da tamamlayalım
    # Box 2'ye ulaştı
    lm = mock_landmark([25, 8], [25, 5], [25, 35]) # Açık el
    for _ in range(5):
        fsm.update(lm, (480, 640))
    assert fsm.state == "GRASP"
    
    # Box 2'de kavradı (pinch kapatıldı)
    lm = mock_landmark([25, 8], [25, 5], [25, 10])
    for _ in range(10):
        fsm.update(lm, (480, 640))
    assert fsm.state == "MOVE"
    
    # Montaj alanına getirdi
    lm = mock_landmark([15, 28], [15, 25], [15, 30])
    for _ in range(5):
        fsm.update(lm, (480, 640))
    assert fsm.state == "PLACE"
    
    # Bıraktı (açık el)
    lm = mock_landmark([15, 28], [15, 25], [15, 55])
    for _ in range(10):
        fsm.update(lm, (480, 640))
    
    # Tüm reçete tamamlandığı için FSM sıfırlanıp IDLE olmalı ve rapor kaydedilmeli
    print(f"Test 7 - Çevrim Tamamlama: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE", "Çevrim sıfırlanmadı!"
    assert len(fsm.reported_cycles) == 1, "Çevrim raporu kaydedilmedi!"
    
    report = fsm.reported_cycles[0]
    print(f"\nKaydedilen Çevrim Verisi (Geriye Dönük Uyumluluk): {report}")
    assert report["cycle_no"] == 1
    assert "duration_sec" in report
    assert "tmu" in report
    print("[BAŞARILI] Geriye dönük uyumluluk testi geçti.\n")

def run_home_area_tests():
    print("=== 'HOME AREA' KONTROL TESTLERİ BAŞLATILDI ===")
    config_data = {
        "stations": [
            {"name": "Box 1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"name": "Assembly Area", "polygon": [[10, 20], [20, 20], [20, 30], [10, 30]]},
            {"name": "Home Area", "polygon": [[0, 30], [10, 30], [10, 40], [0, 40]]}
        ],
        "recipe": ["Box 1", "Assembly Area"],
        "grasp": {
            "pinch_threshold": 0.28,
            "release_threshold": 0.40,
            "velocity_threshold": 8.0,
            "grasp_confirm_frames": 4,
            "release_confirm_frames": 3
        }
    }
    fsm = MOSTTracker(config_data)

    # H1: IDLE başlangıcı (El Home Area içinde, örn: 5, 35)
    lm = mock_landmark([5, 38], [5, 35], [5, 65]) # Açık el
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"H1 - Home IDLE: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE"

    # H2: REACH (El Home Area dışına çıkıyor, örn: 5, 20)
    lm = mock_landmark([5, 23], [5, 20], [5, 50]) # Açık el
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"H2 - Home REACH: Beklenen: REACH, Gerçek: {fsm.state}")
    assert fsm.state == "REACH"

    # H3: GRASP (Box 1'e girdi ve kavradı)
    lm = mock_landmark([5, 8], [5, 5], [5, 10]) # Kapalı el
    for _ in range(10):
        fsm.update(lm, (480, 640))
    print(f"H3 - Home GRASP -> MOVE: Beklenen: MOVE, Gerçek: {fsm.state}")
    assert fsm.state == "MOVE"

    # H4: PLACE (Montaj alanında bıraktı)
    lm = mock_landmark([15, 28], [15, 25], [15, 55]) # Açık el
    for _ in range(10):
        fsm.update(lm, (480, 640))
    # Son adım tamamlandığı için RETURNING_HOME fazına geçmeli
    print(f"H4 - Home PLACE -> RETURNING_HOME: Beklenen: RETURNING_HOME, Gerçek: {fsm.state}")
    assert fsm.state == "RETURNING_HOME"

    # H5: RETURNING_HOME -> IDLE (El Home Area içine döndü, örn: 5, 35)
    lm = mock_landmark([5, 38], [5, 35], [5, 65]) # Açık el
    for _ in range(5):
        fsm.update(lm, (480, 640))
    print(f"H5 - Home Dönüşü Tamamlama: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE"
    assert len(fsm.reported_cycles) == 1, "Rapor kaydedilmedi!"
    
    report = fsm.reported_cycles[0]
    print(f"Kaydedilen Çevrim Verisi (Home Area): {report}")
    print("[BAŞARILI] 'Home Area' entegrasyon testi geçti.\n")

if __name__ == "__main__":
    run_tests()
    run_home_area_tests()
    print("=== TÜM KALİTE KONTROL TESTLERİ BAŞARIYLA GEÇTİ! ===")
