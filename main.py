# 載入 YOLOv8 模型套件
from ultralytics import YOLO

# 載入 OpenCV 套件，用來處理影像與攝影機
import cv2

# 載入 time 套件，用來記錄每個 ID 的出現與消失時間
import time

# 載入已訓練好的 YOLOv8 模型（這邊使用 yolov8n.pt，預設可辨識80種 COCO 類別）
model = YOLO("yolov8n.pt")

# 開啟攝影機（0 表示預設攝影機）
cap = cv2.VideoCapture(0)

# 用來儲存追蹤到的 ID 與其出現時間等資料
tracked_ids = {}

# 進入影像處理迴圈
while True:
    # 讀取攝影機畫面，一張一張影像
    ret, frame = cap.read()
    if not ret:
        break  # 若讀取失敗就結束

    # 用 YOLO 模型進行追蹤（track 模式會自動分配物體 ID，persist=True 可記住 ID）
    results = model.track(frame, persist=True, stream=True)

    # 逐張推論結果處理（這邊只會有一張）
    for r in results:
        # 取得偵測到的邊框資訊
        boxes = r.boxes

        # 取得每個物體的 ID（如果存在）
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

        # 逐一處理每個物體的 ID
        for obj_id in ids:
            # 如果是第一次看到這個 ID，就新增記錄
            if obj_id not in tracked_ids:
                tracked_ids[obj_id] = {
                    "first_seen": time.time(),     # 第一次出現時間
                    "last_seen": time.time(),      # 最近一次看到的時間
                    "seen_count": 1                # 出現次數
                }
                print(f"🔵 新 ID {obj_id} 被追蹤")
            else:
                # 如果這個 ID 曾經看過，就更新時間與次數
                tracked_ids[obj_id]["last_seen"] = time.time()
                tracked_ids[obj_id]["seen_count"] += 1
                print(f"🟢 ID {obj_id} 再次出現")

        # 將結果畫到影像上，包含邊框與 ID 標籤
        annotated_frame = r.plot()

    # 顯示處理後的畫面（含邊框、分類、ID）
    cv2.imshow("Tracking", annotated_frame)

    # 如果按下「q」鍵就離開程式
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 釋放攝影機與關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
