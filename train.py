from ultralytics import YOLO
import os
Yaml_path = "D:/yolov_11_custom4/data.yaml"
if __name__ == "__main__":
    model = YOLO("yolo11m.pt")
    model.train( data=Yaml_path,imgsz=640,batch=8,epochs=100,workers=1,device=0,fliplr=0.0)
