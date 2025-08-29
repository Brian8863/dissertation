import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torch

# -------------------------
# 1️⃣ 載入模型
# -------------------------
torch.set_num_threads(6)   # ⚡ 使用 4 核心 (依你 CPU 實際核心數調整)
yolo_model = YOLO("Model/traffic&count.pt")       # YOLO 偵測紅綠燈與倒數區
cnn_model = load_model("cnn_digit_model_new.h5")  # CNN 辨識倒數區數字
img_size = (28, 28)  # CNN 輸入大小

video_path = "video/light2.mp4"  # 影片路徑
video_path2 = 1
cap = cv2.VideoCapture(video_path2)

# -------------------------
# 2️⃣ 動態裁切函數
# -------------------------
def crop_digits_auto(crop_img, max_digits=2, min_digit_width=10):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    col_sum = np.sum(thresh, axis=0)
    if np.all(col_sum == 0):
        return []

    left = np.argmax(col_sum > 0)
    right = len(col_sum) - np.argmax(col_sum[::-1] > 0)
    cropped = crop_img[:, left:right]
    w = right - left
    num_digits = min(max_digits, max(1, w // min_digit_width))
    digit_width = w // num_digits
    digits = [cropped[:, i*digit_width:(i+1)*digit_width] for i in range(num_digits)]
    return digits

# -------------------------
# 3️⃣ 平滑機制 (保存最近 15 幀)
# -------------------------
history = deque(maxlen=15)

# -------------------------
# 4️⃣ 影片辨識
# -------------------------
cv2.namedWindow("Traffic + Countdown", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ⚡ 調低 YOLO 輸入尺寸，加快推論
    results = yolo_model(frame, imgsz=640)
    current_count = None

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)

            # -------------------------
            # 倒數區
            # -------------------------
            if int(cls) == 0:
                crop_img = frame[y1:y2, x1:x2]
                digits = crop_digits_auto(crop_img, max_digits=2)

                batch = []
                for d in digits:
                    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                    if np.mean(gray) < 58:  # 太暗視為「沒數字」
                        continue
                    resized = cv2.resize(gray, img_size)
                    normalized = resized / 255.0
                    batch.append(normalized)

                countdown = ""
                if batch:
                    batch = np.array(batch).reshape(-1, 28, 28, 1)
                    preds = cnn_model.predict(batch, verbose=0)
                    for pred in preds:
                        digit = np.argmax(pred)
                        countdown += str(digit)

                if countdown != "":
                    current_count = countdown
                    history.append(countdown)

                # -------------------------
                # 平滑顯示（取最近 15 幀的眾數）
                # -------------------------
                if len(history) > 0:
                    stable_count = max(set(history), key=history.count)
                    cv2.putText(frame, stable_count, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)

            # -------------------------
            # 綠燈
            # -------------------------
            elif int(cls) == 1:
                cv2.putText(frame, "Green", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

            # -------------------------
            # 紅燈
            # -------------------------
            elif int(cls) == 2:
                cv2.putText(frame, "Red", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)

    cv2.imshow("Traffic + Countdown", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
