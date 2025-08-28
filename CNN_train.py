'''import os
import cv2
import numpy as np

images = []
labels = []

base_dir = "digit_dataset"

for label in range(10):
    folder = os.path.join(base_dir, str(label))
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (28,28))
        img = img / 255.0  # normalize
        images.append(img)
        labels.append(label)

images = np.array(images).reshape(-1,28,28,1)
labels = np.array(labels)
print("圖片數量:", len(images))'''



'''import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# -------------------------
# 1. 讀取圖片與標籤
# -------------------------
images = []
labels = []

base_dir = "digit_dataset"  # 你的資料夾路徑

for label in range(10):
    folder = os.path.join(base_dir, str(label))
    if not os.path.exists(folder):
        continue
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (28,28))
        img = img / 255.0  # normalize
        images.append(img)
        labels.append(label)

images = np.array(images).reshape(-1,28,28,1)
labels = np.array(labels)

print("圖片數量:", len(images))

# -------------------------
# 2. 建立 CNN 模型
# -------------------------
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# 3. 訓練模型
# -------------------------
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.1)

# -------------------------
# 4. 存檔與測試
# -------------------------
model.save("cnn_digit_model.h5")
print("模型已存檔: cnn_digit_model.h5")

# 測試第一張圖片
pred = model.predict(images[0].reshape(1,28,28,1))
print("預測數字:", np.argmax(pred))'''




'''import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -------------------------
# 1. 載入模型
# -------------------------
yolo_model = YOLO("Model/test.pt")  # 你的倒數區 YOLO 模型
cnn_model = load_model("cnn_digit_model.h5")  # 你剛訓練好的 CNN

img_size = (28,28)  # CNN 輸入大小

video_path = "video/light.mp4"
cap = cv2.VideoCapture(video_path)

# -------------------------
# 2. 動態裁切函數
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
# 3. 影片辨識
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) != 0:  # 只處理倒數區 class=0
                continue

            x1, y1, x2, y2 = map(int, box)
            crop_img = frame[y1:y2, x1:x2]

            # 動態裁切成單數字
            digits = crop_digits_auto(crop_img, max_digits=2)

            countdown = ""
            for d in digits:
                gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, img_size)
                normalized = resized / 255.0
                pred = cnn_model.predict(normalized.reshape(1,28,28,1))
                digit = np.argmax(pred)
                countdown += str(digit)

            # 顯示倒數時間
            cv2.putText(frame, countdown, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Countdown Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''





import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -------------------------
# 1️⃣ 載入模型
# -------------------------
yolo_model = YOLO("Model/test.pt")      # YOLO 偵測紅綠燈與倒數區
cnn_model = load_model("cnn_digit_model.h5")  # CNN 辨識倒數區數字
img_size = (28, 28)  # CNN 輸入大小

video_path = "video/light.mp4"
cap = cv2.VideoCapture(video_path)

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
# 3️⃣ 影片辨識
# -------------------------
cv2.namedWindow("Traffic + Countdown", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

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

                countdown = ""
                for d in digits:
                    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, img_size)
                    normalized = resized / 255.0
                    pred = cnn_model.predict(normalized.reshape(1,28,28,1), verbose=0)
                    digit = np.argmax(pred)
                    countdown += str(digit)

                cv2.putText(frame, countdown, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

            # -------------------------
            # 綠燈
            # -------------------------
            elif int(cls) == 1:
                cv2.putText(frame, "Green", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # -------------------------
            # 紅燈
            # -------------------------
            elif int(cls) == 2:
                cv2.putText(frame, "Red", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    cv2.imshow("Traffic + Countdown", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


