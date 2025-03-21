from ultralytics import YOLO
# Load model
model = YOLO('trained_model.pt')
model.predict(source = 0,conf = 0.5, 
              imgsz =640,show= True) # Predict image