import numpy as np
import time
from fsm import MOSTStateMachine

def mock_landmark(wrist_pos, index_tip_pos, thumb_tip_pos):
    """MediaPipe el eklemleri formatında sentetik veri üretir."""
    landmarks = [None] * 21
    # Wrist (0)
    landmarks[0] = {'x': wrist_pos[0], 'y': wrist_pos[1]}
    # Index MCP (5)
    landmarks[5] = {'x': wrist_pos[0], 'y': wrist_pos[1] - 50.0} # El büyüklüğü ölçeğini 50 yapmak için
    # Thumb Tip (4)
    landmarks[4] = {'x': thumb_tip_pos[0], 'y': thumb_tip_pos[1]}
    # Index Tip (8)
    landmarks[8] = {'x': index_tip_pos[0], 'y': index_tip_pos[1]}
    return landmarks

def run_tests():
    # 1. Konfigürasyon Kurulumu
    config_data = {
        "rois": {
            "Box 1": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "Box 2": [[20, 0], [30, 0], [30, 10], [20, 10]],
            "Assembly Area": [[10, 20], [20, 20], [20, 30], [10, 30]]
        },
        "assembly_recipe": ["Box 1", "Box 2", "Assembly Area"]
    }
    
    fsm = MOSTStateMachine(config_data)
    current_time = time.time()
    
    print("=== ENTEGRE KALİTE KONTROL VE TEST PROGRAMI BAŞLATILDI ===")
    
    # Test 1: IDLE Durumu Doğrulaması
    # El montaj alanında (örneğin 15, 25)
    lm = mock_landmark([15, 28], [15, 25], [15, 55]) # El açık (pinch_dist = 30 / 50 = 0.6)
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"Test 1 - IDLE Kontrolü: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE", "Test 1 başarısız!"
    
    # Test 2: REACH Durumu Geçişi
    # El montaj alanını terk ediyor (örneğin 15, 15)
    current_time += 1.0
    lm = mock_landmark([15, 18], [15, 15], [15, 45]) # El açık
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"Test 2 - REACH Kontrolü: Beklenen: REACH, Gerçek: {fsm.state}")
    assert fsm.state == "REACH", "Test 2 başarısız!"
    
    # Test 3: GRASP Durumu Geçişi
    # El Box 1 içine giriyor (örneğin 5, 5)
    current_time += 1.0
    lm = mock_landmark([5, 8], [5, 5], [5, 35]) # El açık (kutu içinde kavrama bekleniyor)
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"Test 3 - GRASP Kontrolü: Beklenen: GRASP, Gerçek: {fsm.state}")
    assert fsm.state == "GRASP", "Test 3 başarısız!"
    
    # Test 4: MOVE Durumu Geçişi (Kavrama Doğrulaması)
    # Parmakları kapatıp (pinch_dist = 5 / 50 = 0.1 < 0.28) Box 1'de bekliyoruz
    lm = mock_landmark([5, 8], [5, 5], [5, 10])
    
    # EMA yakınsaması ve grasp_confirm_counter (3 kare) tetiklenmesi için 10 kare çalıştırıyoruz
    for i in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
        
    print(f"Test 4 - MOVE Kontrolü: Beklenen: MOVE, Gerçek: {fsm.state}")
    assert fsm.state == "MOVE", "Test 4 başarısız!"
    
    # Test 5: PLACE Durumu Geçişi
    # El Box 1'den montaj alanına taşıyor (örneğin 15, 25)
    current_time += 1.0
    lm = mock_landmark([15, 28], [15, 25], [15, 30]) # Hala parçayı tutuyor
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"Test 5 - PLACE Kontrolü: Beklenen: PLACE, Gerçek: {fsm.state}")
    assert fsm.state == "PLACE", "Test 5 başarısız!"
    
    # Test 6: Reçete Adımı İlerlemesi (PLACE -> REACH (Box 2))
    # Parmakları açıp parçayı bırakıyoruz (pinch_dist = 30 / 50 = 0.6 > 0.40)
    lm = mock_landmark([15, 28], [15, 25], [15, 55])
    for _ in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
        
    print(f"Test 6 - Reçete İlerleme Kontrolü: Beklenen: REACH (Box 2 için), Gerçek: {fsm.state}")
    assert fsm.state == "REACH", "Test 6 başarısız!"
    assert fsm.current_recipe_idx == 1, "Reçete dizini ilerlemedi!"
    
    # Test 7: Çevrim Tamamlama Raporlaması
    # İkinci kutuyu (Box 2) da tamamlayalım
    # Box 2'ye ulaştı
    current_time += 1.0
    lm = mock_landmark([25, 8], [25, 5], [25, 35]) # Açık el
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    assert fsm.state == "GRASP"
    
    # Box 2'de kavradı (pinch kapatıldı)
    lm = mock_landmark([25, 8], [25, 5], [25, 10])
    for _ in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
    assert fsm.state == "MOVE"
    
    # Montaj alanına getirdi
    current_time += 1.0
    lm = mock_landmark([15, 28], [15, 25], [15, 30])
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    assert fsm.state == "PLACE"
    
    # Bıraktı (açık el)
    lm = mock_landmark([15, 28], [15, 25], [15, 55])
    for _ in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
    
    # Tüm reçete tamamlandığı için FSM sıfırlanıp IDLE olmalı ve rapor kaydedilmeli
    print(f"Test 7 - Çevrim Tamamlama: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE", "Çevrim sıfırlanmadı!"
    assert len(fsm.reported_cycles) == 1, "Çevrim raporu kaydedilmedi!"
    
    report = fsm.reported_cycles[0]
    print(f"\nKaydedilen Çevrim Verisi (Geriye Dönük Uyumluluk): {report}")
    assert report["Döngü No"] == 1
    assert "Uzanma (TMU)" in report
    assert "Toplam (TMU)" in report
    print("[BAŞARILI] Geriye dönük uyumluluk testi geçti.\n")

def run_home_area_tests():
    print("=== 'HOME AREA' KONTROL TESTLERİ BAŞLATILDI ===")
    config_data = {
        "rois": {
            "Box 1": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "Assembly Area": [[10, 20], [20, 20], [20, 30], [10, 30]],
            "Home Area": [[0, 30], [10, 30], [10, 40], [0, 40]]
        },
        "assembly_recipe": ["Box 1", "Assembly Area"]
    }
    fsm = MOSTStateMachine(config_data)
    current_time = time.time()

    # H1: IDLE başlangıcı (El Home Area içinde, örn: 5, 35)
    lm = mock_landmark([5, 38], [5, 35], [5, 65]) # Açık el
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"H1 - Home IDLE: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE"

    # H2: REACH (El Home Area dışına çıkıyor, örn: 5, 20)
    current_time += 1.0
    lm = mock_landmark([5, 23], [5, 20], [5, 50]) # Açık el
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"H2 - Home REACH: Beklenen: REACH, Gerçek: {fsm.state}")
    assert fsm.state == "REACH"

    # H3: GRASP (Box 1'e girdi ve kavradı)
    current_time += 1.0
    lm = mock_landmark([5, 8], [5, 5], [5, 10]) # Kapalı el
    for _ in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
    print(f"H3 - Home GRASP -> MOVE: Beklenen: MOVE, Gerçek: {fsm.state}")
    assert fsm.state == "MOVE"

    # H4: PLACE (Montaj alanında bıraktı)
    current_time += 1.0
    lm = mock_landmark([15, 28], [15, 25], [15, 55]) # Açık el
    for _ in range(10):
        current_time += 0.1
        fsm.process_frame(lm, "Right", current_time)
    # Son adım tamamlandığı için RETURNING_HOME fazına geçmeli
    print(f"H4 - Home PLACE -> RETURNING_HOME: Beklenen: RETURNING_HOME, Gerçek: {fsm.state}")
    assert fsm.state == "RETURNING_HOME"

    # H5: RETURNING_HOME -> IDLE (El Home Area içine döndü, örn: 5, 35)
    current_time += 1.0
    lm = mock_landmark([5, 38], [5, 35], [5, 65]) # Açık el
    for _ in range(5):
        fsm.process_frame(lm, "Right", current_time)
    print(f"H5 - Home Dönüşü Tamamlama: Beklenen: IDLE, Gerçek: {fsm.state}")
    assert fsm.state == "IDLE"
    assert len(fsm.reported_cycles) == 1, "Rapor kaydedilmedi!"
    
    report = fsm.reported_cycles[0]
    print(f"Kaydedilen Çevrim Verisi (Home Area): {report}")
    assert report["Dönüş (TMU)"] > 0, "Dönüş süresi hesaplanamadı!"
    print("[BAŞARILI] 'Home Area' entegrasyon testi geçti.\n")

if __name__ == "__main__":
    run_tests()
    run_home_area_tests()
    print("=== TÜM KALİTE KONTROL TESTLERİ BAŞARIYLA GEÇTİ! ===")
