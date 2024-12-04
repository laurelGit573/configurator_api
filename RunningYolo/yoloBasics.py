from ultralytics import YOLO
import cv2

model = YOLO('../YoloWeights/yolov8l.pt')
results = model("RunningYolo/Images/img.png", show=True)
cv2.waitKey(0)