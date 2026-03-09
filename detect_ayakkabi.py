from ultralytics import YOLO

model = YOLO('box_results/weights/last.pt') 

results = model.predict(
    source='test/images', 
    save=True, 
    conf=0.01,         
    iou=0.01,        
    agnostic_nms=True, 
    device='cpu'
)