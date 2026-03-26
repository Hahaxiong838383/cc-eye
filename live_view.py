#!/usr/bin/env python3
"""
live_view.py — cc-eye 实时标注预览

YOLO 物体检测 + OpenCV Haar 人脸检测 + MediaPipe 手部/面部关键点
在摄像头画面上实时标注所有检测到的物体和人体特征。

用法（必须在终端前台运行，不能后台启动）：
  cd ~/mycc/2-Projects/cc-eye && source .venv/bin/activate
  python live_view.py                    # YOLO + 人脸
  python live_view.py --mediapipe        # 加上 MediaPipe 手部/面部

快捷键：
  ESC  退出
  M    切换 MediaPipe（手部+面部关键点）
  Y    切换 YOLO 物体检测
  S    截图保存到 /tmp/cc-eye-capture.jpg
"""

import cv2
import time
import sys
import argparse
import platform

# ── 参数解析 ──
parser = argparse.ArgumentParser(description="cc-eye 实时标注预览")
parser.add_argument("--mediapipe", action="store_true", help="启用 MediaPipe 手部+面部")
parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO")
parser.add_argument("--yolo-model", default="yolov8x.pt", help="YOLO 模型 (默认 yolov8x.pt，最高精度)")
parser.add_argument("--conf", type=float, default=0.15, help="YOLO 置信度阈值 (默认 0.15，检测更多物体)")
args = parser.parse_args()

# ── 摄像头 ──
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: 摄像头打不开")
    sys.exit(1)

# ── YOLO ──
yolo = None
yolo_enabled = not args.no_yolo
if yolo_enabled:
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model)
        print(f"YOLO 加载完成: {args.yolo_model} (conf={args.conf})")
    except Exception as e:
        print(f"YOLO 加载失败: {e}")
        yolo_enabled = False

# ── OpenCV 人脸 ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── MediaPipe ──
mp_enabled = args.mediapipe
face_mesh = None
hands = None
mp_draw = None

def init_mediapipe():
    global face_mesh, hands, mp_draw, mp_enabled
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5
        )
        hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        mp_draw = mp.solutions.drawing_utils
        mp_enabled = True
        print("MediaPipe 加载完成")
    except Exception as e:
        print(f"MediaPipe 加载失败: {e}")
        mp_enabled = False

if mp_enabled:
    init_mediapipe()

# ── 主循环 ──
cv2.namedWindow("cc-eye [live]", cv2.WINDOW_NORMAL)
cv2.resizeWindow("cc-eye [live]", 1280, 720)
print("=" * 40)
print("  cc-eye 实时标注预览")
print("  ESC:退出  M:MediaPipe  Y:YOLO  S:截图")
print("=" * 40)

frame_count = 0
yolo_results = []
fps = 0.0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1

    # ── YOLO 物体检测（每帧都跑，M4 32GB 性能充足）──
    if yolo_enabled and yolo:
        yolo_results = yolo(frame, verbose=False, conf=args.conf)

    # 类别配色表（全 80 类 COCO 按语义分 10 组）
    CATEGORY_COLORS = {
        # 人（绿色系）
        "person": (50, 200, 50),
        # 家具（蓝金）
        "chair": (200, 150, 50), "couch": (200, 150, 50), "bed": (200, 150, 50),
        "dining table": (200, 150, 50), "toilet": (200, 150, 50), "potted plant": (200, 150, 50),
        # 电子设备（浅红）
        "tv": (255, 100, 100), "laptop": (255, 100, 100), "cell phone": (255, 100, 100),
        "keyboard": (255, 100, 100), "mouse": (255, 100, 100), "remote": (255, 100, 100),
        "microwave": (255, 100, 100), "oven": (255, 100, 100), "toaster": (255, 100, 100),
        "refrigerator": (255, 100, 100),
        # 日用品（浅蓝）
        "cup": (100, 200, 255), "bottle": (100, 200, 255), "book": (100, 200, 255),
        "clock": (100, 200, 255), "vase": (100, 200, 255), "scissors": (100, 200, 255),
        "teddy bear": (100, 200, 255), "hair drier": (100, 200, 255), "toothbrush": (100, 200, 255),
        # 随身物品（紫色）
        "backpack": (180, 100, 255), "handbag": (180, 100, 255), "umbrella": (180, 100, 255),
        "suitcase": (180, 100, 255), "tie": (180, 100, 255),
        # 食物（橙色）
        "banana": (50, 150, 255), "apple": (50, 150, 255), "sandwich": (50, 150, 255),
        "orange": (50, 150, 255), "broccoli": (50, 150, 255), "carrot": (50, 150, 255),
        "hot dog": (50, 150, 255), "pizza": (50, 150, 255), "donut": (50, 150, 255),
        "cake": (50, 150, 255), "wine glass": (50, 150, 255), "fork": (50, 150, 255),
        "knife": (50, 150, 255), "spoon": (50, 150, 255), "bowl": (50, 150, 255),
        # 交通工具（青色）
        "car": (255, 200, 50), "bicycle": (255, 200, 50), "motorcycle": (255, 200, 50),
        "airplane": (255, 200, 50), "bus": (255, 200, 50), "train": (255, 200, 50),
        "truck": (255, 200, 50), "boat": (255, 200, 50),
        # 动物（粉红）
        "cat": (200, 100, 200), "dog": (200, 100, 200), "horse": (200, 100, 200),
        "sheep": (200, 100, 200), "cow": (200, 100, 200), "elephant": (200, 100, 200),
        "bear": (200, 100, 200), "zebra": (200, 100, 200), "giraffe": (200, 100, 200),
        "bird": (200, 100, 200),
        # 运动（亮黄）
        "frisbee": (0, 255, 255), "skis": (0, 255, 255), "snowboard": (0, 255, 255),
        "sports ball": (0, 255, 255), "kite": (0, 255, 255), "baseball bat": (0, 255, 255),
        "baseball glove": (0, 255, 255), "skateboard": (0, 255, 255), "surfboard": (0, 255, 255),
        "tennis racket": (0, 255, 255),
        # 交通标志（白色）
        "traffic light": (220, 220, 220), "fire hydrant": (220, 220, 220),
        "stop sign": (220, 220, 220), "parking meter": (220, 220, 220), "bench": (220, 220, 220),
    }
    DEFAULT_COLOR = (200, 200, 100)   # 其他类别：浅黄

    obj_count = {}
    if yolo_enabled and yolo_results:
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])
                color = CATEGORY_COLORS.get(label, DEFAULT_COLOR)

                # 统计物体数量
                obj_count[label] = obj_count.get(label, 0) + 1

                # 计算物体尺寸和面积占比
                obj_w = x2 - x1
                obj_h = y2 - y1
                area_pct = (obj_w * obj_h) / (w * h) * 100

                # 画框（大物体粗框，小物体细框）
                thickness = 3 if area_pct > 5 else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 主标签：类别 + 置信度
                txt = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # 副标签：尺寸 + 面积占比（框底部）
                size_txt = f"{obj_w}x{obj_h} ({area_pct:.1f}%)"
                (sw, sh), _ = cv2.getTextSize(size_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(frame, (x1, y2), (x1 + sw + 4, y2 + sh + 4), color, -1)
                cv2.putText(frame, size_txt, (x1 + 2, y2 + sh + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # 右下角物体统计面板
    if yolo_enabled and obj_count:
        panel_y = h - 30 - len(obj_count) * 22
        cv2.putText(frame, f"Objects: {sum(obj_count.values())}", (w - 200, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        for i, (lbl, cnt) in enumerate(sorted(obj_count.items(), key=lambda x: -x[1])):
            color = CATEGORY_COLORS.get(lbl, DEFAULT_COLOR)
            cy = panel_y + 22 * (i + 1)
            cv2.putText(frame, f"  {lbl}: {cnt}", (w - 200, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── 人脸检测 ──
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── MediaPipe ──
    if mp_enabled and face_mesh and hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 面部关键点
        fm = face_mesh.process(rgb)
        if fm.multi_face_landmarks:
            for fl in fm.multi_face_landmarks:
                # 眼睛
                for idx in [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,
                            362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]:
                    lm = fl.landmark[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 255), -1)
                # 虹膜
                for idx in [468, 473]:
                    lm = fl.landmark[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 255, 0), -1)
                # 嘴唇
                for idx in [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]:
                    lm = fl.landmark[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (200, 100, 255), -1)

        # 手部骨架
        hr = hands.process(rgb)
        if hr.multi_hand_landmarks:
            import mediapipe as mp
            for hl in hr.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

    # ── HUD ──
    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
    prev_time = now

    mode_str = []
    if yolo_enabled:
        mode_str.append("YOLO")
    if mp_enabled:
        mode_str.append("MP")
    mode_str.append("Face")
    mode_label = "+".join(mode_str)

    cv2.putText(frame, f"cc-eye [{mode_label}]", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, f"{time.strftime('%H:%M:%S')}  FPS:{fps:.0f}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("cc-eye [live]", frame)

    # ── 按键 ──
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("m") or key == ord("M"):
        if mp_enabled:
            mp_enabled = False
            print("MediaPipe OFF")
        else:
            if face_mesh is None:
                init_mediapipe()
            else:
                mp_enabled = True
            print(f"MediaPipe {'ON' if mp_enabled else 'OFF'}")
    elif key == ord("y") or key == ord("Y"):
        yolo_enabled = not yolo_enabled
        if yolo_enabled and yolo is None:
            from ultralytics import YOLO
            yolo = YOLO("yolov8n.pt")
        print(f"YOLO {'ON' if yolo_enabled else 'OFF'}")
    elif key == ord("s") or key == ord("S"):
        path = "/tmp/cc-eye-capture.jpg"
        cv2.imwrite(path, frame)
        print(f"截图已保存: {path}")

cap.release()
if face_mesh:
    face_mesh.close()
if hands:
    hands.close()
cv2.destroyAllWindows()
print("已退出")
