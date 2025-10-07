import os, cv2, numpy as np, threading, queue, time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3

# ---------- 降低多線程負擔 ----------
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
try: cv2.setNumThreads(1)
except: pass
try: import torch; torch.set_num_threads(1)
except: pass

# ---------- 參數 ----------
VIDEO_PATH, YOLO_IMGSZ, PROCESS_EVERY_N, QUEUE_MAX = "video/light.mp4", 1280, 5, 5
IMG_SIZE, CONF_THRESHOLD = (28,28), 0.5
CODES = {"green":int("01000001",2),"red":int("01000011",2),"lt10":int("01000010",2)}

YOLO_MIN_CONF = 0.7
CNN_MIN_CONF  = 0.7

# ---------- 模型 ----------
yolo_model = YOLO("Model/traffic_1280.pt")
cnn_model  = load_model("cnn_digit_model_new.h5")

# ---------- 語音控制與全域引擎 (核心修正) ----------
MIN_INTERVAL = 2.0
last_play_time = {"fast":0}
tts_engine = None # 聲明全域 TTS 引擎變數

def send_tts_msg(msg):
    """直接向 pyttsx3 引擎發送語音訊息。"""
    global tts_engine
    if tts_engine:
        # 訊息被放入引擎的內部佇列
        tts_engine.say(msg) 

def speak_change(msg, key):
    """檢查間隔時間並發送語音訊息（用於控制 $10$ 秒提醒只播一次）。"""
    now = time.time()
    if now - last_play_time[key] > MIN_INTERVAL:
        send_tts_msg(msg)
        last_play_time[key] = now

# ---------- 佇列與控制 (移除 tts_q) ----------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_q, result_q, STOP = queue.Queue(QUEUE_MAX), queue.Queue(QUEUE_MAX), threading.Event()

# ---------- 狀態 ----------
history, last_count, stable_count = deque(maxlen=15), None, None
countdown = {"active":False,"value":None,"last":time.time()}
prev_state = {"lt10":None, "green":None, "red":None}
cnn_enabled = True
last_digit = None
prev_light = None 

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
    global cnn_enabled, last_digit
    if countdown["active"] and countdown["value"] is not None:
        if time.time()-countdown["last"] >= 1:
            countdown["value"] -= 1
            countdown["last"] = time.time()
            if countdown["value"] <= 0:
                countdown.update({"active":False,"value":0})
                cnn_enabled = True
                last_digit = None
                last_play_time["fast"] = 0 

def handle_state(now, key, code):
    if now != prev_state.get(key, None):
        prev_state[key]=now
        if now: 
            print(f"SEND: {code} ")

# ---------- 執行緒 (t_tts_worker 修正) ----------
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
    global stable_count, cnn_enabled, last_digit
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
                if score < YOLO_MIN_CONF: continue
                x1,y1,x2,y2 = map(int, box); c=int(cls)

                if c==0:  # 倒數
                    crop = frame[y1:y2,x1:x2]
                    if cnn_enabled:
                        digits = crop_digits(crop)
                        batch=[]
                        for d in digits:
                            gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
                            if np.mean(gray) > 58:
                                batch.append(cv2.resize(gray, IMG_SIZE)/255.0)
                        if batch:
                            X = np.array(batch).reshape(-1,28,28,1)
                            preds = cnn_model.predict(X, verbose=0)
                            digits_list = []
                            for p in preds:
                                d = np.argmax(p)
                                if np.max(p) >= CNN_MIN_CONF:
                                    digits_list.append(d)
                            if digits_list:
                                cur = int("".join(str(d) for d in digits_list))
                                stable_count = cur
                                if last_digit == 11 and cur == 10:
                                    countdown.update({"active":True,"value":10,"last":time.time()})
                                    cnn_enabled = False
                                last_digit = cur
                    if stable_count is not None and stable_count < 10:
                        dets.append(("lt10", CODES["lt10"], (x1,y1,x2,y2), score, None))
                elif c==1: dets.append(("green", CODES["green"], (x1,y1,x2,y2), score, None))
                elif c==2: dets.append(("red", CODES["red"], (x1,y1,x2,y2), score, None))

        try:
            result_q.put((frame,dets),timeout=0.1)
        except queue.Full:
            result_q.get_nowait()
            result_q.put((frame,dets))

def t_tts_worker():
    """專門用於運行 pyttsx3 引擎事件迴圈的執行緒，確保語音穩定播放。"""
    global tts_engine
    
    # 初始化引擎
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 82)
    tts_engine.setProperty('volume', 2.0)
    
    # 💥 核心修正：啟動引擎的內部事件迴圈，並設為非阻塞 (False) 💥
    try:
        tts_engine.startLoop(False)
    except Exception as e:
        # 如果 startLoop 失敗 (例如在某些 Mac/Linux 環境)，紀錄錯誤但繼續
        print(f"Failed to start TTS loop non-blocking: {e}")
        
    
    while not STOP.is_set():
        try:
            # 讓引擎處理佇列中的任務 (說話、等待等)
            tts_engine.iterate()
            time.sleep(0.05) # 短暫休眠，釋放 CPU 給其他執行緒
        except Exception as e:
            # 如果出現 "run loop not started" 以外的錯誤，繼續輪詢
            if "run loop not started" not in str(e):
                print(f"TTS Engine Iteration Error: {e}")
            time.sleep(0.05)
            continue
            
    # Clean up the engine when exiting
    try:
        # 停止引擎並結束內部迴圈
        tts_engine.stop()
    except:
        pass
    
    tts_engine = None


# ---------- 啟動 ----------
cv2.namedWindow("Traffic+Countdown", cv2.WINDOW_NORMAL)
threading.Thread(target=t_capture, daemon=True).start()
threading.Thread(target=t_detect, daemon=True).start()
threading.Thread(target=t_tts_worker, daemon=True).start() # 啟動專門的 TTS 執行緒

# ---------- 主迴圈 ----------
try:
    while True:
        try: frame,dets = result_q.get(timeout=0.5)
        except queue.Empty:
            update_countdown()
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        if frame is None: break

        # ---------- SEND 編碼：紅燈、綠燈、倒數<10 ----------
        for k, v, _, _, _ in dets:
            handle_state(True, k, v)

        # ---------- 語音控制 (實現不重複提醒) ----------
        if countdown["active"] is False or countdown["value"] > 5:
            current_light = "red" if any(k=="red" for k,_,_,_,_ in dets) else \
                            "green" if any(k=="green" for k,_,_,_,_ in dets) else None
            
            # 判斷燈號是否發生變化
            if current_light != prev_light and current_light is not None:
                if current_light == "red":
                    send_tts_msg("紅燈請停下！") # 偵測到紅燈
                elif current_light == "green":
                    send_tts_msg("綠燈可以走") # 偵測到綠燈
                
                prev_light = current_light # 更新燈號狀態

        # 倒數 10 秒提醒
        if countdown["active"] and countdown["value"] is not None and countdown["value"] == 10:
            # speak_change 會根據 last_play_time["fast"] 確保 $10$ 秒提醒只在計時剛開始時播放一次
            speak_change("綠燈剩餘10秒！", "fast")

        # ---------- 畫面繪製 ----------
        for k,v,(x1,y1,x2,y2),y_conf,c_conf in dets:
            color = (0,255,0) if k=="green" else (0,0,255) if k=="red" else (0,255,255)
            if k=="green": txt=f"Green {y_conf:.2f}"
            elif k=="red": txt=f"Red {y_conf:.2f}"
            elif k=="lt10": txt=f"Cnt<10 {y_conf:.2f}"
            cv2.putText(frame,txt,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)

        update_countdown()
        if countdown["active"]:
            cv2.putText(frame,f"Auto: {countdown['value']}",(50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,200,255),3)

        if frame is not None:
            cv2.imshow("Traffic+Countdown", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
finally:
    STOP.set()
    # 清理佇列
    while not frame_q.empty(): frame_q.get_nowait()
    while not result_q.empty(): result_q.get_nowait()
    try:
        frame_q.put(None, timeout=0.1)
        result_q.put((None,None), timeout=0.1)
    except:
        pass
    cv2.destroyAllWindows()