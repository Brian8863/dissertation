import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torch
import time

# -------------------------
# 載入模型
# -------------------------
yolo_model = YOLO("Model/traffic&count.pt")
cnn_model = load_model("cnn_digit_model_new.h5")
img_size = (28, 28)

video_path = "video/light.mp4"
cap = cv2.VideoCapture(video_path)

# -------------------------
# 動態裁切函數
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
# 平滑機制
# -------------------------
history = deque(maxlen=15)

# -------------------------
# 倒數系統變數
# -------------------------
last_count = None
countdown_active = False
stable_count = None
auto_count = None
last_update_time = time.time()
frame_count = 0
last_results = None

# -------------------------
# 主迴圈
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 每 3 幀做一次 YOLO 偵測
    if frame_count % 3 == 0:
        last_results = yolo_model(frame, imgsz=640)

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

                        if len(history) > 0:
                            stable_count = max(set(history), key=history.count)

                        # 倒數區編碼
                        code = None
                        if stable_count is not None and stable_count < 10:
                            code = "01000010"  # 倒數區 <10秒
                            code_int = int(code,2)
                        elif stable_count is not None:
                            code = f"Countdown: {stable_count}"  # 可選純數字顯示
                        if code:
                            print("Countdown Code:", code_int)

                # 綠燈
                elif int(cls) == 1:
                    code = "01000001"
                    code_int = int(code,2)
                    print("Green Light Code:", code_int)

                # 紅燈
                elif int(cls) == 2:
                    code = "01000011"
                    code_int = int(code,2)
                    print("Red Light Code:", code_int)

    # 自動倒數
    if countdown_active and auto_count is not None:
        now = time.time()
        if now - last_update_time >= 1:
            auto_count -= 1
            last_update_time = now
            if auto_count <= 0:
                countdown_active = False

        print("Auto Countdown:", auto_count)

cap.release()
