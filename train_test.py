import cv2
from ultralytics import YOLO

# model = YOLO("runs/detect/train2/weights/best.pt")
model = YOLO("Model/test.pt")

#cap = cv2.VideoCapture(1) #攝影機
cap = cv2.VideoCapture("video/light.mp4") #讀影片

# 建立可調整大小的視窗
cv2.namedWindow("YOLOv8 Live", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Live", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
