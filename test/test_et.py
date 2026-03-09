from ultralytics import YOLO
import os

model = YOLO('box_results/weights/best.pt')

results = model.predict(source='test/images', save=True, conf=0.25)

print("\n Test islemi bitti! Sonuclar 'runs/detect/predict' klasorune kaydedildi.")
