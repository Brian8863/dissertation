import cv2
import os
import numpy as np
from ultralytics import YOLO

# -------------------------
# 設定
# -------------------------
video_path = "video/light.mp4"
output_dir = "digit_dataset"
img_size = (28, 28)  # CNN 輸入尺寸
os.makedirs(output_dir, exist_ok=True)

yolo_model = YOLO("Model/test.pt")
cap = cv2.VideoCapture(video_path)
digit_count = 0

# -------------------------
# 動態裁切函數
# -------------------------
def crop_digits_auto(crop_img, max_digits=2, min_digit_width=10):
    """
    自動裁切倒數區成單數字
    crop_img: 倒數區圖像
    max_digits: 最大可能位數
    min_digit_width: 每個數字最小寬度，避免切太小
    """
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    col_sum = np.sum(thresh, axis=0)
    if np.all(col_sum == 0):
        return []  # 沒有數字

    # 找到左右有內容的邊界
    left = np.argmax(col_sum > 0)
    right = len(col_sum) - np.argmax(col_sum[::-1] > 0)
    cropped = crop_img[:, left:right]
    w = right - left

    # 判斷要切幾個數字
    num_digits = min(max_digits, max(1, w // min_digit_width))
    digit_width = w // num_digits
    digits = [cropped[:, i*digit_width:(i+1)*digit_width] for i in range(num_digits)]
    return digits

# -------------------------
# 影片處理
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 偵測倒數區
    results = yolo_model(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            # 只抓倒數區 class=0
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            crop_img = frame[y1:y2, x1:x2]

            # -------------------------
            # 動態裁切成單數字
            # -------------------------
            digits = crop_digits_auto(crop_img, max_digits=2, min_digit_width=10)

            # -------------------------
            # 預處理 + 存檔
            # -------------------------
            for d in digits:
                gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, img_size)
                normalized = resized / 255.0

                cv2.imwrite(os.path.join(output_dir, f"digit_{digit_count}.png"),
                            (normalized*255).astype(np.uint8))
                digit_count += 1

cap.release()
cv2.destroyAllWindows()
print("完成處理，總共產生數字圖片:", digit_count)
