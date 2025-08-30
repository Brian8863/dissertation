import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torch
import time

# -------------------------
# 1️⃣ 載入模型
# -------------------------
torch.set_num_threads(6)
yolo_model = YOLO("Model/traffic&count.pt")       # YOLO 偵測紅綠燈與倒數區
cnn_model = load_model("cnn_digit_model_new.h5")  # CNN 辨識倒數區數字
img_size = (28, 28)

video_path = "video/light.mp4"
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
# 3️⃣ 平滑機制
# -------------------------
history = deque(maxlen=15)

# -------------------------
# 4️⃣ 倒數系統變數
# -------------------------
last_count = None
countdown_active = False
stable_count = None
auto_count = None
last_update_time = time.time()
frame_count = 0
last_results = None

# -------------------------
# 5️⃣ 影片辨識
# -------------------------
cv2.namedWindow("Traffic + Countdown", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 每 3 幀才做 YOLO 辨識
    if frame_count % 3 == 0:
        last_results = yolo_model(frame, imgsz=320)  # imgsz 可降速加快辨識

    current_count = None

    if last_results is not None:
        for r in last_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)

                # 倒數區
                if int(cls) == 0:
                    crop_img = frame[y1:y2, x1:x2]
                    digits = crop_digits_auto(crop_img, max_digits=2)

                    batch = []
                    for d in digits:
                        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                        if np.mean(gray) < 58:
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
                        current_count = int(countdown)
                        history.append(current_count)

                        # 平滑顯示
                        if len(history) > 0:
                            stable_count = max(set(history), key=history.count)

                        # 啟動倒數條件
                        if last_count is not None and stable_count is not None:
                            if stable_count == last_count - 1:
                                countdown_active = True
                                auto_count = stable_count
                                last_update_time = time.time()

                        last_count = stable_count

                    # 畫框
                    if stable_count is not None:
                        cv2.putText(frame, str(stable_count), (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)

                # 綠燈
                elif int(cls) == 1:
                    cv2.putText(frame, "Green", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

                # 紅燈
                elif int(cls) == 2:
                    cv2.putText(frame, "Red", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)

    # 自動倒數
    if countdown_active and auto_count is not None:
        now = time.time()
        if now - last_update_time >= 1:
            auto_count -= 1
            last_update_time = now
            if auto_count <= 0:
                countdown_active = False

        cv2.putText(frame, f"Auto: {auto_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)

    cv2.imshow("Traffic + Countdown", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
