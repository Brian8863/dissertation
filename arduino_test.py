import os, cv2, numpy as np, threading, queue, time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import serial  # Arduino 傳送可取消註解

# ---------- 降低多線程負擔 ----------
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
try: cv2.setNumThreads(1)
except: pass
try: import torch; torch.set_num_threads(1)
except: pass

# ---------- 參數 ----------
VIDEO_PATH, YOLO_IMGSZ, PROCESS_EVERY_N, QUEUE_MAX = "video/light2.mp4", 1280, 5, 5
IMG_SIZE, CONF_THRESHOLD = (28,28), 0.5
CODES = {"green":65,"red":67,"lt10":66}  # 對應 Arduino 的單 byte

YOLO_MIN_CONF = 0.5
CNN_MIN_CONF  = 0.5

# ---------- 模型 ----------
yolo_model = YOLO("Model/traffic_1280.pt")
cnn_model  = load_model("cnn_digit_model_new.h5")

# ---------- Arduino ----------
# arduino = serial.Serial("COM3", 115200, timeout=1)
# time.sleep(2)  # 等待 Arduino 重置

# ---------- 佇列與控制 ----------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_q, result_q, STOP = queue.Queue(QUEUE_MAX), queue.Queue(QUEUE_MAX), threading.Event()

# ---------- 狀態 ----------
history, last_count, stable_count = deque(maxlen=15), None, None
countdown = {"active":False,"value":None,"last":time.time()}
prev_state = {"green":None,"red":None,"lt10":None,"txt":None}

# ---------- 工具函數 ----------
def crop_digits(img, max_digits=2, min_w=10):
    if img is None or img.size==0: return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g,50,255,cv2.THRESH_BINARY)
    col_sum = np.sum(th,axis=0)
    if np.all(col_sum==0): return []
    l,r = np.argmax(col_sum>0), len(col_sum)-np.argmax(col_sum[::-1]>0)
    if r<=l: return []
    w, num = r-l, min(max_digits,max(1,(r-l)//min_w))
    return [img[:,i*w//num:(i+1)*w//num] for i in range(num)]

def update_countdown():
    if countdown["active"] and countdown["value"] is not None:
        if time.time()-countdown["last"]>=1:
            countdown["value"] -= 1
            countdown["last"]=time.time()
            if countdown["value"]<=0:
                countdown.update({"active":False,"value":0})

def send_to_arduino(value):
    """發送單 byte 給 Arduino"""
    try:
        # arduino.write(bytes([value]))  # 取消註解可傳給 Arduino
        print(f"SEND: {value}")  # 測試用
    except Exception as e:
        print("Arduino 傳送錯誤:", e)

def handle_state(now, key, code):
    if now != prev_state[key]:
        prev_state[key]=now
        if now:
            send_to_arduino(code)

def handle_text(txt):
    if txt != prev_state["txt"] and txt is not None:
        prev_state["txt"] = txt
        # 文字訊號可不傳，或用單 byte 方式映射
        # send_to_arduino(txt_byte)

# ---------- 執行緒 ----------
def t_capture():
    while not STOP.is_set():
        ret, f = cap.read()
        if not ret:
            frame_q.put(None)
            break
        try:
            frame_q.put(f,timeout=0.1)
        except queue.Full:
            frame_q.get_nowait()
            frame_q.put(f)
    cap.release()

def t_detect():
    global last_count, stable_count
    idx=0
    while not STOP.is_set():
        try: frame = frame_q.get(timeout=0.2)
        except queue.Empty: continue
        if frame is None: 
            result_q.put((None,None))
            break
        idx+=1
        if idx % PROCESS_EVERY_N != 0: continue

        try: results = yolo_model(frame, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        except: continue

        dets=[]
        for r in results:
            if not r.boxes: continue
            for box,cls,score in zip(r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.cls.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy()):
                if score < YOLO_MIN_CONF:
                    continue
                x1,y1,x2,y2 = map(int, box); c=int(cls)
                if c==0:  # 倒數
                    crop = frame[y1:y2,x1:x2]
                    digits = crop_digits(crop)
                    batch=[]
                    for d in digits:
                        gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                        if np.mean(gray)>58:
                            batch.append(cv2.resize(gray, IMG_SIZE)/255.0)
                    cnn_conf=None
                    if batch:
                        X = np.array(batch).reshape(-1,28,28,1)
                        preds = cnn_model.predict(X, verbose=0)
                        digits_list, confs = [], []
                        for p in preds:
                            d = np.argmax(p)
                            c_val = np.max(p)
                            if c_val >= CNN_MIN_CONF:
                                digits_list.append(d)
                                confs.append(c_val)
                        if digits_list:
                            cur = int("".join(str(d) for d in digits_list))
                            cnn_conf = float(np.mean(confs))
                            history.append(cur)
                            stable_count = max(set(history), key=history.count)
                            if last_count and stable_count == last_count-1:
                                countdown.update({"active":True,"value":stable_count,"last":time.time()})
                            last_count = stable_count
                    if stable_count is not None:
                        dets.append(("lt10" if stable_count<10 else "txt",
                                     stable_count if stable_count>=10 else CODES["lt10"],
                                     (x1,y1,x2,y2),
                                     score, cnn_conf))
                elif c==1: dets.append(("green",CODES["green"],(x1,y1,x2,y2),score,None))
                elif c==2: dets.append(("red",CODES["red"],(x1,y1,x2,y2),score,None))
        try:
            result_q.put((frame,dets),timeout=0.1)
        except queue.Full:
            result_q.get_nowait()
            result_q.put((frame,dets))

# ---------- 啟動 ----------
cv2.namedWindow("Traffic+Countdown", cv2.WINDOW_NORMAL)
threading.Thread(target=t_capture, daemon=True).start()
threading.Thread(target=t_detect, daemon=True).start()

# ---------- 主迴圈 ----------
try:
    while True:
        try: frame,dets = result_q.get(timeout=0.5)
        except queue.Empty:
            update_countdown()
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        if frame is None: break

        # 狀態處理
        handle_state(any(k=="green" for k,_,_,_,_ in dets),"green",CODES["green"])
        handle_state(any(k=="red"   for k,_,_,_,_ in dets),"red",CODES["red"])
        handle_state(any(k=="lt10"  for k,_,_,_,_ in dets),"lt10",CODES["lt10"])
        handle_text(next((v for k,v,_,_,_ in dets if k=="txt"),None))

        # 繪製
        for k,v,(x1,y1,x2,y2),y_conf,c_conf in dets:
            color = (0,255,0) if k=="green" else (0,0,255) if k=="red" else (0,255,255)
            if k=="green": txt=f"Green {y_conf:.2f}"
            elif k=="red": txt=f"Red {y_conf:.2f}"
            elif k=="lt10": txt=f"Cnt<10 {y_conf:.2f}"
            else: txt=f"{v} (CNN {c_conf:.2f})" if c_conf is not None else f"{v}"
            cv2.putText(frame,txt,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        update_countdown()
        if countdown["active"]:
            cv2.putText(frame,f"Auto: {countdown['value']}",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,200,255),3)

        if frame is not None:
            cv2.imshow("Traffic+Countdown", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
finally:
    STOP.set()
    while not frame_q.empty(): frame_q.get_nowait()
    while not result_q.empty(): result_q.get_nowait()
    try:
        frame_q.put(None, timeout=0.1)
        result_q.put((None,None), timeout=0.1)
    except:
        pass
    cv2.destroyAllWindows()
    # arduino.close()
