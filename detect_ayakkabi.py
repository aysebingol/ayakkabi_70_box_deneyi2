from ultralytics import YOLO
import os

model_path = 'box_results/weights/best.pt'
source_path = 'test/images'

if not os.path.exists(model_path):
    print(f"Hata: {model_path} bulunamadı!")
else:
    model = YOLO(model_path)
    
   
    results = model.predict(
    source='test/images', 
    save=True, 
    conf=0.02,        
    iou=0.1,        
    agnostic_nms=True,
    device='cpu'
)
    
    print(f"\n--- İşlem Tamam! ---")
    print(f"Lütfen şu klasöre bakınız: box_result/deneme_1")