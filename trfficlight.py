import os, cv2, numpy as np, threading, queue, time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ---------- 降低多線程負擔 ----------
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
try: cv2.setNumThreads(1)
except: pass
try: import torch; torch.set_num_threads(1)
except: pass

# ---------- 參數 ----------
VIDEO_PATH, YOLO_IMGSZ, PROCESS_EVERY_N, QUEUE_MAX = "video/IMG_3180.MOV", 1280, 3, 5
IMG_SIZE, CONF_THRESHOLD = (28,28), 0.5
CODES = {"green":int("01000001",2),"red":int("01000011",2),"lt10":int("01000010",2)}

# ---------- 模型 ----------
yolo_model = YOLO("Model/traffic_1280.pt")
cnn_model  = load_model("cnn_digit_model_new.h5")

# ---------- 佇列 ----------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_q, result_q, STOP = queue.Queue(QUEUE_MAX), queue.Queue(QUEUE_MAX), threading.Event()

# ---------- 狀態 ----------
history, last_count, stable_count = deque(maxlen=15), None, None
countdown = {"active":False,"value":None,"last":time.time()}
prev_state = {"green":None,"red":None,"lt10":None,"txt":None}

# ---------- 工具 ----------
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

def handle_state(now, key, code):
    if now != prev_state[key]:
        prev_state[key]=now
        if now: print("SEND:", code)

def handle_text(txt):
    if txt != prev_state["txt"] and txt is not None:
        print("COUNT:", txt); prev_state["txt"]=txt

# ---------- 執行緒 ----------
def t_capture():
    while not STOP.is_set():
        ret, f = cap.read()
        if not ret: frame_q.put(None); break
        try: frame_q.put(f,timeout=0.1)
        except queue.Full: frame_q.get_nowait(); frame_q.put(f)
    cap.release()

def t_detect():
    global last_count, stable_count
    idx=0
    while not STOP.is_set():
        try: frame = frame_q.get(timeout=0.2)
        except queue.Empty: continue
        if frame is None: result_q.put((None,None)); break
        idx+=1
        if idx%PROCESS_EVERY_N: continue

        try: results = yolo_model(frame, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        except: continue

        dets=[]
        for r in results:
            if not r.boxes: continue
            for box,cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                x1,y1,x2,y2 = map(int, box); c=int(cls)
                if c==0: # 倒數
                    crop=frame[y1:y2,x1:x2]; digits=crop_digits(crop)
                    batch=[cv2.resize(cv2.cvtColor(d,cv2.COLOR_BGR2GRAY),IMG_SIZE)/255.0 for d in digits if np.mean(d)>58]
                    if batch:
                        preds=cnn_model.predict(np.array(batch).reshape(-1,28,28,1),verbose=0)
                        try:
                            cur=int("".join(str(np.argmax(p)) for p in preds))
                            history.append(cur); stable_count=max(set(history),key=history.count)
                            if last_count and stable_count==last_count-1:
                                countdown.update({"active":True,"value":stable_count,"last":time.time()})
                            last_count=stable_count
                        except: pass
                    if stable_count:
                        dets.append(("lt10" if stable_count<10 else "txt", stable_count if stable_count>=10 else CODES["lt10"], (x1,y1,x2,y2)))
                elif c==1: dets.append(("green",CODES["green"],(x1,y1,x2,y2)))
                elif c==2: dets.append(("red",CODES["red"],(x1,y1,x2,y2)))
        try: result_q.put((frame,dets),timeout=0.1)
        except queue.Full: result_q.get_nowait(); result_q.put((frame,dets))

# ---------- 啟動 ----------
cv2.namedWindow("Traffic+Countdown", cv2.WINDOW_NORMAL)
threading.Thread(target=t_capture,daemon=True).start()
threading.Thread(target=t_detect,daemon=True).start()

# ---------- 主迴圈 ----------
try:
    while True:
        try: frame,dets=result_q.get(timeout=0.5)
        except queue.Empty:
            update_countdown()
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        if frame is None: break

        # 狀態處理
        handle_state(any(k=="green" for k,_,_ in dets),"green",CODES["green"])
        handle_state(any(k=="red"   for k,_,_ in dets),"red",CODES["red"])
        handle_state(any(k=="lt10"  for k,_,_ in dets),"lt10",CODES["lt10"])
        handle_text(next((v for k,v,_ in dets if k=="txt"),None))

        # 繪製
        for k,v,(x1,y1,x2,y2) in dets:
            color = (0,255,0) if k=="green" else (0,0,255) if k=="red" else (0,255,255)
            txt   = "Green" if k=="green" else "Red" if k=="red" else "Cnt<10" if k=="lt10" else str(v)
            cv2.putText(frame,txt,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        update_countdown()
        if countdown["active"]:
            cv2.putText(frame,f"Auto: {countdown['value']}",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,200,255),3)

        cv2.imshow("Traffic+Countdown",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
finally:
    STOP.set()
    try: frame_q.put_nowait(None); result_q.put_nowait((None,None))
    except: pass
    cv2.destroyAllWindows()
