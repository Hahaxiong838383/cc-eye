"""
main.py -- cc-eye 主循环
摄像头 → 身份识别 + 表情识别 + 面部特征 → 状态融合 → 互动
"""

import os
import sys
import time
import math
import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    FACE_RECOGNITION_INTERVAL,
    FACE_REGISTER_COUNT,
    EXPRESSION_INTERVAL,
    FACE_CONF_THRESHOLD,
    FACE_TRACK_THRESHOLD,
    EAR_BLINK_THRESHOLD,
    EAR_SMOOTHING_ALPHA,
    BLINK_CONFIRM_FRAMES,
    BLINK_REFRACTORY_FRAMES,
    MAR_OPEN_THRESHOLD,
    WINDOW_NAME,
    FACE_DATA_DIR,
)
from identity import FaceIdentity
from expression import ExpressionRecognizer, EMOTION_CN
from state_fusion import StateFusion, SignalSnapshot, UserState
from interaction import InteractionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("cc-eye")


# ==================================================================
# MediaPipe FaceMesh 面部特征提取（简化版，不依赖 gesture_drag_demo）
# ==================================================================

# 关键点索引
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [13, 14, 78, 308]  # 上唇、下唇、左嘴角、右嘴角


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_ear(landmarks: list, indices: List[int], w: int, h: int) -> float:
    """计算眼睛纵横比 (Eye Aspect Ratio)"""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    # 垂直距离
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    # 水平距离
    horiz = _dist(pts[0], pts[3])
    if horiz < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * horiz)


def compute_mar(landmarks: list, w: int, h: int) -> float:
    """计算嘴巴纵横比 (Mouth Aspect Ratio)"""
    top = (landmarks[13].x * w, landmarks[13].y * h)
    bottom = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[78].x * w, landmarks[78].y * h)
    right = (landmarks[308].x * w, landmarks[308].y * h)
    vert = _dist(top, bottom)
    horiz = _dist(left, right)
    if horiz < 1e-6:
        return 0.0
    return vert / horiz


def estimate_head_pose(landmarks: list, w: int, h: int) -> Tuple[float, float]:
    """
    简化版头部朝向估计，返回 (yaw, pitch) 度数
    基于鼻尖与面部中心的偏移
    """
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    # yaw: 鼻尖在左右耳连线上的位置
    ear_cx = (left_ear.x + right_ear.x) / 2.0
    ear_dist = abs(right_ear.x - left_ear.x)
    if ear_dist < 1e-6:
        yaw = 0.0
    else:
        yaw = (nose.x - ear_cx) / ear_dist * 90.0  # 粗略映射到度数

    # pitch: 鼻尖在额头到下巴连线上的位置
    face_cy = (forehead.y + chin.y) / 2.0
    face_h = abs(chin.y - forehead.y)
    if face_h < 1e-6:
        pitch = 0.0
    else:
        pitch = (nose.y - face_cy) / face_h * 90.0

    return (yaw, pitch)


# ==================================================================
# 注册流程
# ==================================================================

def run_registration(cap: cv2.VideoCapture, face_id: FaceIdentity) -> Optional[str]:
    """引导用户在摄像头前注册人脸"""
    logger.info("=== 人脸注册模式 ===")
    print("\n" + "=" * 50)
    print("  cc-eye 人脸注册")
    print("=" * 50)

    name = input("请输入你的名字: ").strip()
    if not name:
        print("名字不能为空，取消注册")
        return None

    print(f"\n好的 {name}，接下来请看着摄像头")
    print(f"慢慢转动头部，我需要采集 {FACE_REGISTER_COUNT} 张不同角度的照片")
    print("按 空格键 开始采集，ESC 取消\n")

    # 等待用户准备
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Press SPACE to start, ESC to cancel",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None
        if key == 32:  # SPACE
            break

    # 采集
    frames: List[np.ndarray] = []
    countdown_start = time.time()
    collect_interval = 0.5  # 每 0.5 秒采集一帧

    while len(frames) < FACE_REGISTER_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - countdown_start
        if elapsed >= collect_interval:
            frames.append(frame.copy())
            countdown_start = time.time()
            logger.info("采集 %d/%d", len(frames), FACE_REGISTER_COUNT)

        # 显示进度
        progress = f"Collecting: {len(frames)}/{FACE_REGISTER_COUNT}"
        cv2.putText(frame, progress, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, "Slowly turn your head", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None

    # 注册
    print(f"\n正在处理 {len(frames)} 帧图像...")
    success = face_id.register(name, frames)
    if success:
        print(f"注册成功！已记住 {name} 的脸")
        return name
    else:
        print("注册失败：未能提取足够的人脸特征，请重试")
        return None


# ==================================================================
# 主循环
# ==================================================================

def main() -> None:
    logger.info("cc-eye 启动中...")

    # 初始化各模块
    face_id = FaceIdentity()
    expr_rec = ExpressionRecognizer()
    fusion = StateFusion()
    interaction = InteractionEngine()

    # 注册状态变化回调
    def on_state_change(old: UserState, new: UserState, snapshot: object) -> None:
        interaction.get_response(old, new)

    fusion.on_state_change(on_state_change)

    # 打开摄像头（macOS 兼容）
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        logger.error("无法打开摄像头 %d", CAMERA_INDEX)
        sys.exit(1)

    logger.info("摄像头已打开 (%dx%d)", CAMERA_WIDTH, CAMERA_HEIGHT)

    # 首次运行检测
    os.makedirs(FACE_DATA_DIR, exist_ok=True)
    if not face_id.has_any_registered():
        logger.info("未检测到已注册人脸，进入注册流程")
        registered_name = run_registration(cap, face_id)
        if registered_name:
            interaction.get_greeting(registered_name, is_new=True)

    # MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=FACE_CONF_THRESHOLD,
        min_tracking_confidence=FACE_TRACK_THRESHOLD,
    )

    # 状态变量
    frame_count = 0
    current_identity = "unknown"
    current_identity_confidence = 0.0
    current_expression = "neutral"
    current_expression_cn = "平静"
    smoothed_ear = 0.3
    below_threshold_count = 0
    refractory_count = 0
    greeted_identities: dict = {}  # name -> last_greet_time

    logger.info("主循环开始，ESC 退出，R 重新注册")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("读帧失败")
            continue

        frame = cv2.flip(frame, 1)  # 镜像翻转
        h, w = frame.shape[:2]
        frame_count += 1
        now = time.time()

        # ----------------------------------------------------------
        # 1) 身份识别（每 N 帧）
        # ----------------------------------------------------------
        if frame_count % FACE_RECOGNITION_INTERVAL == 0:
            name, conf = face_id.identify(frame)
            if name != "unknown":
                current_identity = name
                current_identity_confidence = conf
                # 问候逻辑
                last_greet = greeted_identities.get(name, 0.0)
                if (now - last_greet) > 60.0:
                    interaction.get_greeting(name)
                    greeted_identities[name] = now

        # ----------------------------------------------------------
        # 2) 表情识别（每 N 帧）
        # ----------------------------------------------------------
        if frame_count % EXPRESSION_INTERVAL == 0:
            result = expr_rec.detect(frame)
            if result is not None:
                current_expression = result.dominant_emotion
                current_expression_cn = result.dominant_cn

        # ----------------------------------------------------------
        # 3) MediaPipe FaceMesh 面部特征
        # ----------------------------------------------------------
        ear = 0.3
        mar = 0.0
        yaw = 0.0
        pitch = 0.0
        face_detected = False
        is_blink = False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            face_detected = True
            lm = mesh_results.multi_face_landmarks[0].landmark

            # EAR
            left_ear = compute_ear(lm, LEFT_EYE_IDX, w, h)
            right_ear = compute_ear(lm, RIGHT_EYE_IDX, w, h)
            raw_ear = (left_ear + right_ear) / 2.0
            smoothed_ear = EAR_SMOOTHING_ALPHA * raw_ear + (1.0 - EAR_SMOOTHING_ALPHA) * smoothed_ear
            ear = smoothed_ear

            # 眨眼检测
            if refractory_count > 0:
                refractory_count -= 1
            else:
                if ear < EAR_BLINK_THRESHOLD:
                    below_threshold_count += 1
                else:
                    if below_threshold_count >= BLINK_CONFIRM_FRAMES:
                        is_blink = True
                        refractory_count = BLINK_REFRACTORY_FRAMES
                    below_threshold_count = 0

            # MAR
            mar = compute_mar(lm, w, h)

            # 头部朝向
            yaw, pitch = estimate_head_pose(lm, w, h)

        # ----------------------------------------------------------
        # 4) 状态融合
        # ----------------------------------------------------------
        signal = SignalSnapshot(
            timestamp=now,
            ear=ear,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
            blink=is_blink,
            dominant_emotion=current_expression,
            emotion_confidence=0.0,
            face_detected=face_detected,
            identity=current_identity,
        )
        state_snap = fusion.push(signal)

        # ----------------------------------------------------------
        # 5) 绘制
        # ----------------------------------------------------------

        # 面部框（使用 face_recognition 的位置）
        if frame_count % FACE_RECOGNITION_INTERVAL == 0 or frame_count <= 1:
            try:
                locations = face_id.get_face_locations(frame)
                for (top, right, bottom, left) in locations:
                    is_known = current_identity != "unknown"
                    color = (0, 255, 0) if is_known else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    label = current_identity if is_known else "unknown"
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            except Exception:
                pass

        # 左上角信息面板
        panel_y = 30
        line_gap = 28

        # 身份
        id_text = f"ID: {current_identity}"
        if current_identity != "unknown":
            id_text += f" ({current_identity_confidence:.0%})"
        cv2.putText(frame, id_text, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panel_y += line_gap

        # 表情
        expr_text = f"Expr: {current_expression} ({current_expression_cn})"
        cv2.putText(frame, expr_text, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panel_y += line_gap

        # 状态
        state_color = {
            UserState.FOCUSED: (0, 255, 0),
            UserState.FATIGUED: (0, 165, 255),
            UserState.HAPPY: (0, 255, 255),
            UserState.CONFUSED: (0, 128, 255),
            UserState.AWAY: (128, 128, 128),
        }.get(state_snap.state, (255, 255, 255))

        state_text = f"State: {state_snap.state.value} ({state_snap.state.cn})"
        cv2.putText(frame, state_text, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)
        panel_y += line_gap

        # EAR / 眨眼率
        info_text = f"EAR: {ear:.2f} | Blink: {state_snap.blink_rate:.0f}/min"
        cv2.putText(frame, info_text, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        panel_y += line_gap

        # Yaw / Pitch
        pose_text = f"Yaw: {yaw:.1f} | Pitch: {pitch:.1f}"
        cv2.putText(frame, pose_text, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # cc 对话气泡（右下角）
        interaction.draw_bubble(frame)

        # 显示
        cv2.imshow(WINDOW_NAME, frame)

        # ----------------------------------------------------------
        # 6) 按键处理
        # ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            logger.info("用户退出")
            break
        elif key == ord("r") or key == ord("R"):
            logger.info("用户请求重新注册")
            registered = run_registration(cap, face_id)
            if registered:
                interaction.get_greeting(registered, is_new=True)
                expr_rec.reset()
                fusion.reset()

    # 清理
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    logger.info("cc-eye 已退出")


if __name__ == "__main__":
    main()
