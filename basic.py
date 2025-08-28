from ultralytics import YOLO
import cv2

# 載入 YOLOv8 模型（yolov8n.pt 是最輕量的版本）
model = YOLO("Model/yolov8n.pt")

# 開啟攝影機（0 = 預設內建攝影機）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 讀取一張畫面
    if not ret:
        break

    # 使用模型對當前畫面進行預測（stream=True 可節省記憶體）
    results = model.predict(frame, stream=True)

    # 將結果畫到影像上
    for r in results:
        annotated_frame = r.plot()

    # 顯示帶有辨識結果的畫面
    cv2.imshow("YOLOv8 Real-Time", annotated_frame)

    # 按下 q 鍵就結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉攝影機與視窗
cap.release()
cv2.destroyAllWindows()
