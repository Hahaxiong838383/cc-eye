"""
camera_daemon.py — cc 的常驻摄像头服务（1fps 高频监测版）

v2 架构：
- 1秒主循环，多分辨率分层处理
- 运动检测：160×120（极速，每帧）
- 人脸检测：320×240（分频，2s/静态5s）
- moondream 快扫：异步后台线程，10s
- minicpm-v 精扫：异步后台线程，事件触发/120s
- 变化门控：静态场景跳过重处理

用法：
  cd cc-eye && source .venv/bin/activate
  python camera_daemon.py &
"""

import cv2
import time
import json
import threading
import numpy as np
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── 配置 ────────────────────────────────────
CAPTURE_INTERVAL = 1.0          # 秒，主循环节拍（目标 1fps）
CAPTURE_WIDTH = 640             # 采集分辨率（从 1280 降到 640）
CAPTURE_HEIGHT = 480            # 采集分辨率（从 720 降到 480）
JPEG_QUALITY = 70               # 帧保存质量（从默认 95 降到 70）

LATEST_FRAME_PATH = "/tmp/cc-eye-latest.jpg"
EVENTS_PATH = "/tmp/cc-eye-events.jsonl"
SCENE_PATH = "/tmp/cc-eye-scene.json"

# 多分辨率处理
MOTION_SIZE = (160, 120)        # 运动检测用（极小，极快）
MOTION_BLUR_KERNEL = (5, 5)     # 对应小图的高斯核

# 检测阈值
MOTION_THRESHOLD = 8.0          # 帧差 > 此值 = 有运动
MOTION_BIG = 16.0               # 帧差 > 此值 = 大幅运动，触发精扫
MOTION_GATE = 2.5               # 帧差 < 此值 = 完全静态（跳过人脸检测）

# 分频控制
PERSON_CHECK_EVERY = 3          # 正常每 N tick 检测人体（YOLO）
PERSON_CHECK_STATIC = 6         # 静态场景每 N tick 检测人体
FACE_ID_EVERY = 20              # 每 N tick 做一次人脸身份识别（仅在有人时）
PERSON_LEFT_GRACE = 5           # 连续 N 次未检到人才判定离开（防抖）

# 双模型策略
FAST_SCAN_INTERVAL = 10         # moondream 快扫间隔（秒）
DETAIL_SCAN_INTERVAL = 60       # minicpm-v 定时精扫间隔（秒）
MAX_EVENTS = 200


# ── 异步扫描器 ────────────────────────────────
class AsyncScanner:
    """后台线程运行模型推理，不阻塞 1s 主循环"""

    def __init__(self, name: str):
        self.name = name
        self._busy = False
        self._result: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, fn, *args, **kwargs) -> bool:
        if self._busy:
            return False
        self._busy = True
        t = threading.Thread(
            target=self._run, args=(fn, args, kwargs), daemon=True
        )
        t.start()
        return True

    def _run(self, fn, args, kwargs):
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._result = result
        except Exception as e:
            print(f"[{self.name}] 扫描异常: {e}")
        finally:
            self._busy = False

    def poll(self) -> Optional[str]:
        with self._lock:
            r = self._result
            self._result = None
            return r


# ── 工具函数 ────────────────────────────────

def open_camera():
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def log_event(event_type: str, detail: str = ""):
    entry = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail,
    }
    events_file = Path(EVENTS_PATH)
    with open(events_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    lines = events_file.read_text().strip().split("\n")
    if len(lines) > MAX_EVENTS:
        events_file.write_text("\n".join(lines[-MAX_EVENTS:]) + "\n")


def init_vision_models() -> tuple:
    fast_model = None
    detail_model = None
    try:
        from vision_models import VisionModel
        vm_fast = VisionModel(mode="fast")
        if vm_fast.is_available:
            fast_model = vm_fast
            print(f"[cc-eye daemon] 快扫模型就绪: {vm_fast.model}")
        vm_detail = VisionModel(mode="balanced")
        if vm_detail.is_available:
            detail_model = vm_detail
            print(f"[cc-eye daemon] 精扫模型就绪: {vm_detail.model}")
        if not fast_model and not detail_model:
            print("[cc-eye daemon] 视觉模型不可用，仅运行基础检测")
    except Exception as e:
        print(f"[cc-eye daemon] 视觉模型加载失败: {e}")
    return fast_model, detail_model


FAST_SCAN_PROMPT = (
    "Describe the scene in one sentence: "
    "what is the person doing, and what key objects are visible."
)

# ── 人脸识别（懒加载）──
_face_recognizer = None
_recognized_people: list = []


def _get_face_recognizer():
    global _face_recognizer
    if _face_recognizer is not None:
        return _face_recognizer
    try:
        from cc_face import FaceRecognizer
        _face_recognizer = FaceRecognizer()
        print("[cc-eye daemon] LBPH 人脸识别器已加载")
    except Exception as e:
        print(f"[cc-eye daemon] 人脸识别器加载失败: {e}")
    return _face_recognizer


def recognize_faces(frame) -> list:
    recognizer = _get_face_recognizer()
    tmp_path = "/tmp/cc-eye-recognize-tmp.jpg"
    cv2.imwrite(tmp_path, frame)

    if recognizer is not None:
        try:
            result = recognizer.recognize(tmp_path)
            if result and len(result) == 2:
                name, confidence = result
                if name:
                    recognizer.auto_learn(name, tmp_path)
                    print(f"[cc-eye daemon] LBPH 识别: {name} (置信度 {confidence:.0f})")
                    return [name]
        except Exception as e:
            print(f"[cc-eye daemon] LBPH 识别异常: {e}")

    if recognizer is not None:
        try:
            recognizer.auto_learn("chuange", tmp_path)
        except Exception:
            pass
    return ["chuange"]


def describe_scene(vm, use_english: bool = False, people: list = None) -> Optional[str]:
    if use_english:
        prompt = FAST_SCAN_PROMPT
        if people:
            names = ", ".join(people)
            prompt = f"[Context: The person in the image is {names}. ] " + prompt
        return vm.describe_camera(prompt)
    try:
        from cc_context import build_vision_prompt
        prompt = build_vision_prompt()
    except ImportError:
        prompt = (
            "简洁描述这个场景："
            "1) 有谁在，在做什么 "
            "2) 桌上/房间里的重要物品 "
            "3) 有什么变化或异常。"
            "2-3句话。"
        )
    if people:
        names = "、".join(people)
        prompt = f"[重要提示：画面中的人是{names}（我的老板，叫川哥）。描述时请直接称呼他的名字，不要说'一名男子'。]\n{prompt}"
    return vm.describe_camera(prompt)


def save_scene(description: str, face_count: int, people: list = None) -> None:
    scene = {
        "ts": datetime.now().isoformat(),
        "description": description,
        "face_count": face_count,
        "people": people or [],
    }
    Path(SCENE_PATH).write_text(
        json.dumps(scene, ensure_ascii=False, indent=2)
    )


# ── 主循环 ────────────────────────────────────

def main():
    print(f"[cc-eye daemon] 启动摄像头服务（1fps 高频监测版）...")
    print(f"  采集: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT} → 运动检测 {MOTION_SIZE[0]}×{MOTION_SIZE[1]}")
    print(f"  节拍: {CAPTURE_INTERVAL}s | JPEG q{JPEG_QUALITY}")
    print(f"  快扫: {FAST_SCAN_INTERVAL}s (moondream, 异步)")
    print(f"  精扫: {DETAIL_SCAN_INTERVAL}s (minicpm-v, 异步)")

    cap = open_camera()
    if not cap.isOpened():
        print("[cc-eye daemon] 摄像头打开失败！")
        return

    # 预热
    for _ in range(5):
        cap.read()

    # YOLO 人体检测（替代 Haar 正脸，低头/侧身/背面都能检测）
    yolo_model = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("[cc-eye daemon] YOLO 人体检测器已加载 (yolov8n)")
    except Exception as e:
        print(f"[cc-eye daemon] YOLO 加载失败，降级到 Haar: {e}")

    # Haar Cascade（仅用于人脸身份识别，不再用于人体存在判断）
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 双模型 + 异步扫描器
    fast_model, detail_model = init_vision_models()
    fast_scanner = AsyncScanner("moondream") if fast_model else None
    detail_scanner = AsyncScanner("minicpm-v") if detail_model else None

    # 状态
    prev_motion_gray = None
    prev_person_present = False     # YOLO 人体检测结果（替代 prev_face_count）
    person_miss_count = 0           # 连续未检到人的次数（防抖用）
    tick = 0
    last_fast_scan = 0.0
    last_detail_scan = 0.0
    pending_detail = False
    fps_counter = 0
    fps_start = time.time()

    log_event("daemon_start", f"1fps 高频监测 {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
    print("[cc-eye daemon] 运行中... Ctrl+C 停止")

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[cc-eye daemon] 读帧失败，重试...")
                time.sleep(0.5)
                continue

            frame = cv2.flip(frame, 1)
            tick += 1

            # ── 1. 保存帧（压缩 JPEG）──
            cv2.imwrite(
                LATEST_FRAME_PATH, frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            # ── 2. 运动检测（160×120 极速）──
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_gray = cv2.resize(gray_full, MOTION_SIZE)
            motion_gray = cv2.GaussianBlur(motion_gray, MOTION_BLUR_KERNEL, 0)

            mean_diff = 0.0
            has_motion = False
            is_big_motion = False
            is_static = True

            if prev_motion_gray is not None:
                diff = cv2.absdiff(prev_motion_gray, motion_gray)
                mean_diff = float(np.mean(diff))
                is_static = mean_diff < MOTION_GATE
                has_motion = mean_diff > MOTION_THRESHOLD
                is_big_motion = mean_diff > MOTION_BIG

                if has_motion:
                    log_event("motion", f"diff={mean_diff:.1f}")

            prev_motion_gray = motion_gray

            # ── 3. 人体检测（YOLO person，分频 + 防抖）──
            # YOLO 检测整个人体，不依赖正脸朝向
            # 低头/侧身/背面都能检到，比 Haar 正脸更可靠
            person_interval = PERSON_CHECK_STATIC if is_static else PERSON_CHECK_EVERY
            should_check_person = (tick % person_interval == 0) or is_big_motion

            if should_check_person:
                person_detected = False
                person_count = 0

                if yolo_model is not None:
                    # YOLO 推理：只检测 class 0 (person)，320px 输入，静默模式
                    results = yolo_model(
                        frame, classes=[0], imgsz=320,
                        verbose=False, conf=0.4,
                    )
                    person_count = len(results[0].boxes)
                    person_detected = person_count > 0
                else:
                    # 降级：用 Haar 正脸检测（旧逻辑）
                    small_gray = cv2.resize(
                        gray_full, (0, 0), fx=0.5, fy=0.5
                    )
                    faces = face_cascade.detectMultiScale(
                        small_gray, scaleFactor=1.1,
                        minNeighbors=5, minSize=(30, 30),
                    )
                    person_count = len(faces)
                    person_detected = person_count > 0

                # 状态转换：无人 → 有人
                if person_detected and not prev_person_present:
                    person_miss_count = 0
                    _recognized_people.clear()
                    names = recognize_faces(frame)
                    _recognized_people.extend(names)
                    who = f"（{', '.join(names)}）" if names else ""
                    log_event("person_appeared", f"检测到 {person_count} 人{who}")
                    print(f"[cc-eye daemon] 有人来了！{who}({person_count} 人)")
                    Path("/tmp/cc-eye-person-arrived.flag").write_text(
                        datetime.now().isoformat()
                    )
                    pending_detail = True
                    prev_person_present = True

                # 定期更新身份识别（有人在时，每 FACE_ID_EVERY tick 识别一次）
                elif person_detected and tick % FACE_ID_EVERY == 0:
                    names = recognize_faces(frame)
                    if names:
                        _recognized_people.clear()
                        _recognized_people.extend(names)

                # 状态转换：有人 → 无人（带防抖，连续 N 次未检到才判定离开）
                elif not person_detected and prev_person_present:
                    person_miss_count += 1
                    if person_miss_count >= PERSON_LEFT_GRACE:
                        log_event("person_left", "画面中无人")
                        print("[cc-eye daemon] 人走了")
                        _recognized_people.clear()
                        prev_person_present = False
                        person_miss_count = 0

                # 检测到人，重置 miss 计数
                if person_detected:
                    person_miss_count = 0

            now = time.time()

            # ── 4. moondream 快扫（异步，10s）──
            if fast_scanner and (now - last_fast_scan) >= FAST_SCAN_INTERVAL:
                if fast_scanner.submit(
                    describe_scene, fast_model,
                    use_english=True, people=list(_recognized_people),
                ):
                    last_fast_scan = now

            # 收割快扫结果
            if fast_scanner:
                desc = fast_scanner.poll()
                if desc:
                    save_scene(desc, 1 if prev_person_present else 0, people=_recognized_people)
                    log_event("fast_scan", desc[:100])
                    print(f"[cc-eye daemon] 快扫: {desc[:60]}...")

            # ── 5. minicpm-v 精扫（异步，事件触发 / 120s）──
            should_detail = (
                pending_detail
                or (now - last_detail_scan) >= DETAIL_SCAN_INTERVAL
            )
            if detail_scanner and should_detail:
                if detail_scanner.submit(
                    describe_scene, detail_model,
                    people=list(_recognized_people),
                ):
                    last_detail_scan = now
                    pending_detail = False

            # 收割精扫结果
            if detail_scanner:
                desc = detail_scanner.poll()
                if desc:
                    save_scene(desc, 1 if prev_person_present else 0, people=_recognized_people)
                    log_event("detail_scan", desc[:100])
                    print(f"[cc-eye daemon] 精扫: {desc[:80]}...")

            # ── 6. FPS 统计（每 30 tick 打印一次）──
            fps_counter += 1
            if fps_counter >= 30:
                elapsed = time.time() - fps_start
                actual_fps = fps_counter / elapsed if elapsed > 0 else 0
                print(f"[cc-eye daemon] fps={actual_fps:.1f} diff={mean_diff:.1f} person={'Y' if prev_person_present else 'N'}")
                fps_counter = 0
                fps_start = time.time()

            # ── 精确 1s 节拍 ──
            loop_time = time.time() - t_start
            sleep_time = max(0, CAPTURE_INTERVAL - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[cc-eye daemon] 停止")
        log_event("daemon_stop", "用户中断")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
