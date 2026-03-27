"""
camera_daemon.py — cc 的常驻摄像头服务（轻量后台运行）

功能：
- 每 N 秒拍一帧，保存到 /tmp/cc-eye-latest.jpg
- 简单运动检测：对比前后帧差异，超阈值记录事件
- 人脸出现/消失检测（OpenCV Haar Cascade，不需要 GPU）
- 定期调用本地多模态模型描述场景（moondream/minicpm-v）
- 事件写入 /tmp/cc-eye-events.jsonl（供 cc 读取）
- 最新场景描述写入 /tmp/cc-eye-scene.json（供 cc 读取）
- cc 随时读 /tmp/cc-eye-latest.jpg 就能"看"

用法：
  cd cc-eye && source .venv/bin/activate
  python camera_daemon.py &
"""

import cv2
import time
import json
import numpy as np
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── 配置 ────────────────────────────────────
CAPTURE_INTERVAL = 3        # 秒，拍摄间隔（提速）
LATEST_FRAME_PATH = "/tmp/cc-eye-latest.jpg"
EVENTS_PATH = "/tmp/cc-eye-events.jsonl"
SCENE_PATH = "/tmp/cc-eye-scene.json"
MOTION_THRESHOLD = 8.0      # 帧差均值 > 此值 → 有运动
FACE_CHECK_INTERVAL = 2     # 每 N 次拍摄做一次人脸检测
# 双模型策略：moondream 快扫 + minicpm-v 事件精确分析
FAST_SCAN_INTERVAL = 10     # 秒，moondream 快速扫描间隔
DETAIL_SCAN_INTERVAL = 120  # 秒，minicpm-v 定时精扫间隔
MAX_EVENTS = 200            # 事件文件最大行数（自动滚动）

# ── 初始化 ────────────────────────────────────

def open_camera():
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def log_event(event_type: str, detail: str = ""):
    """追加事件到 JSONL 文件"""
    entry = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail,
    }
    events_file = Path(EVENTS_PATH)

    # 追加
    with open(events_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 滚动（保留最近 MAX_EVENTS 行）
    lines = events_file.read_text().strip().split("\n")
    if len(lines) > MAX_EVENTS:
        events_file.write_text("\n".join(lines[-MAX_EVENTS:]) + "\n")


def init_vision_models() -> tuple:
    """
    初始化双模型：moondream（快扫）+ minicpm-v（精确分析）。
    返回 (fast_model, detail_model)，不可用的为 None。
    """
    fast_model = None
    detail_model = None
    try:
        from vision_models import VisionModel
        # 快速模型（moondream 1.6B，~2s 响应）
        vm_fast = VisionModel(mode="fast")
        if vm_fast.is_available:
            fast_model = vm_fast
            print(f"[cc-eye daemon] 快扫模型就绪: {vm_fast.model}")
        # 精确模型（minicpm-v 8B，~8s 响应）
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
    "Describe what you see briefly: "
    "1) Is anyone present? What are they doing? "
    "2) Key objects visible. "
    "One sentence max."
)


def describe_scene(vm: "VisionModel", use_english: bool = False) -> Optional[str]:
    """用本地模型描述场景。moondream 用英文 prompt，minicpm-v 用中文。"""
    if use_english:
        return vm.describe_camera(FAST_SCAN_PROMPT)
    try:
        from cc_context import build_vision_prompt
        prompt = build_vision_prompt()
    except ImportError:
        prompt = (
            "Describe this scene concisely: "
            "1) People present and what they're doing "
            "2) Notable objects on desk/in room "
            "3) Any changes or unusual things. "
            "Be brief, 2-3 sentences max."
        )
    return vm.describe_camera(prompt)


def save_scene(description: str, face_count: int) -> None:
    """保存场景描述到 JSON 文件"""
    scene = {
        "ts": datetime.now().isoformat(),
        "description": description,
        "face_count": face_count,
    }
    Path(SCENE_PATH).write_text(
        json.dumps(scene, ensure_ascii=False, indent=2)
    )


def main():
    print(f"[cc-eye daemon] 启动摄像头服务（双模型策略）...")
    print(f"  拍摄间隔: {CAPTURE_INTERVAL}s")
    print(f"  快扫间隔: {FAST_SCAN_INTERVAL}s (moondream)")
    print(f"  精扫间隔: {DETAIL_SCAN_INTERVAL}s (minicpm-v)")
    print(f"  最新帧: {LATEST_FRAME_PATH}")
    print(f"  事件日志: {EVENTS_PATH}")
    print(f"  场景描述: {SCENE_PATH}")

    cap = open_camera()
    if not cap.isOpened():
        print("[cc-eye daemon] 摄像头打开失败！")
        return

    # 预热
    for _ in range(5):
        cap.read()

    # OpenCV 人脸检测器（Haar Cascade，CPU 轻量）
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 双模型初始化
    fast_model, detail_model = init_vision_models()

    prev_gray = None
    prev_face_count = 0
    frame_count = 0
    last_fast_scan_time = 0.0
    last_detail_scan_time = 0.0
    pending_detail_scan = False  # 事件触发的精确分析标记

    log_event("daemon_start", "摄像头服务启动（双模型策略）")
    print("[cc-eye daemon] 运行中... Ctrl+C 停止")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[cc-eye daemon] 读帧失败，重试...")
                time.sleep(1)
                continue

            frame = cv2.flip(frame, 1)  # 镜像
            frame_count += 1

            # 保存最新帧
            cv2.imwrite(LATEST_FRAME_PATH, frame)

            # ── 运动检测 ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = float(np.mean(diff))

                if mean_diff > MOTION_THRESHOLD:
                    log_event("motion", f"diff={mean_diff:.1f}")
                    # 大幅运动触发精确分析
                    if mean_diff > MOTION_THRESHOLD * 2:
                        pending_detail_scan = True

            prev_gray = gray

            # ── 人脸检测（每 N 帧）──
            if frame_count % FACE_CHECK_INTERVAL == 0:
                small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(
                    small, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                face_count = len(faces)

                if face_count > 0 and prev_face_count == 0:
                    log_event("person_appeared", f"检测到 {face_count} 张脸")
                    print(f"[cc-eye daemon] 有人来了！({face_count} 张脸)")
                    Path("/tmp/cc-eye-person-arrived.flag").write_text(
                        datetime.now().isoformat()
                    )
                    # 有人出现 → 立即触发精确分析
                    pending_detail_scan = True
                elif face_count == 0 and prev_face_count > 0:
                    log_event("person_left", "画面中无人")
                    print("[cc-eye daemon] 人走了")

                prev_face_count = face_count

            now = time.time()

            # ── moondream 快扫（每 10s，轻量级）──
            if fast_model and (now - last_fast_scan_time) >= FAST_SCAN_INTERVAL:
                last_fast_scan_time = now
                desc = describe_scene(fast_model, use_english=True)
                if desc:
                    save_scene(desc, prev_face_count)
                    log_event("fast_scan", desc[:100])
                    print(f"[cc-eye daemon] 快扫: {desc[:60]}...")

            # ── minicpm-v 精扫（事件触发 或 定时 120s）──
            should_detail = (
                pending_detail_scan
                or (detail_model and (now - last_detail_scan_time) >= DETAIL_SCAN_INTERVAL)
            )
            if detail_model and should_detail:
                last_detail_scan_time = now
                pending_detail_scan = False
                desc = describe_scene(detail_model)
                if desc:
                    save_scene(desc, prev_face_count)
                    log_event("detail_scan", desc[:100])
                    print(f"[cc-eye daemon] 精扫: {desc[:80]}...")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n[cc-eye daemon] 停止")
        log_event("daemon_stop", "用户中断")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
