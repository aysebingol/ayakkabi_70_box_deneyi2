from ultralytics import YOLO
import os

model_path = '../polygon_results/weights/best.pt'

if not os.path.exists(model_path):
    print(f"Hata: {model_path} bulunamadı! Lütfen dosya yolunu kontrol et.")
else:
    model = YOLO(model_path)
    results = model.predict(
        source='../test/images', 
        save=True, 
        conf=0.10, 
        device='cpu',
        project='../POLYGON_SONUCLAR',
        name='deneme_1'
    )
    print("\n--- İşlem Tamam! Sonuçlar 'POLYGON_SONUCLAR' klasöründe. ---")
