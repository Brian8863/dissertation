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



import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# -------------------------
# 1️⃣ 訓練資料路徑
# -------------------------
train_dir = "dataset/"  # 每個數字子資料夾 0~9

# -------------------------
# 2️⃣ 資料增強
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28,28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='sparse',  # 用數字標籤
    shuffle=True
)

# -------------------------
# 3️⃣ 建立 CNN 模型
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 0~9 共 10 類
])

# -------------------------
# 4️⃣ 編譯模型
# -------------------------
model.compile(
    optimizer=Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# 5️⃣ 訓練模型
# -------------------------
epochs = 20
model.fit(train_generator, epochs=epochs)

# -------------------------
# 6️⃣ 儲存模型
# -------------------------
model.save("cnn_digit_model_new.h5")
print("訓練完成，模型已儲存為 cnn_digit_model_new.h5")








