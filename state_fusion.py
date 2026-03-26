"""
state_fusion.py -- 多信号融合引擎
将表情、EAR、眨眼频率、面部朝向、MAR、身份等信号融合为综合状态
"""

import time
import logging
from enum import Enum
from typing import Optional, List, Callable, Dict, Deque
from dataclasses import dataclass, field
from collections import deque

from config import (
    STATE_WINDOW_SECONDS,
    FATIGUE_BLINK_RATE,
    FATIGUE_EAR_THRESHOLD,
    AWAY_CONFIDENCE_THRESHOLD,
    FOCUS_YAW_THRESHOLD,
)

logger = logging.getLogger(__name__)


class UserState(Enum):
    """用户综合状态"""
    FOCUSED = "focused"       # 专注
    FATIGUED = "fatigued"     # 疲劳
    HAPPY = "happy"           # 愉悦
    CONFUSED = "confused"     # 困惑
    AWAY = "away"             # 离开

    @property
    def cn(self) -> str:
        return _STATE_CN.get(self, self.value)


_STATE_CN: Dict[UserState, str] = {
    UserState.FOCUSED: "专注",
    UserState.FATIGUED: "疲劳",
    UserState.HAPPY: "愉悦",
    UserState.CONFUSED: "困惑",
    UserState.AWAY: "离开",
}


@dataclass
class SignalSnapshot:
    """单帧信号采样"""
    timestamp: float
    ear: float = 0.0                        # 眼睛纵横比
    mar: float = 0.0                        # 嘴巴纵横比
    yaw: float = 0.0                        # 偏航角（度）
    pitch: float = 0.0                      # 俯仰角（度）
    blink: bool = False                     # 本帧是否眨眼
    dominant_emotion: str = "neutral"       # 主导表情
    emotion_confidence: float = 0.0         # 表情置信度
    face_detected: bool = False             # 是否检测到人脸
    identity: str = "unknown"               # 身份


@dataclass
class StateSnapshot:
    """融合后的状态快照"""
    state: UserState
    confidence: float                       # 0.0 ~ 1.0
    identity: str = "unknown"
    dominant_emotion: str = "neutral"
    blink_rate: float = 0.0                 # 每分钟眨眼次数
    avg_ear: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    timestamp: float = field(default_factory=time.time)


# 状态变化回调类型
StateChangeCallback = Callable[[UserState, UserState, StateSnapshot], None]


class StateFusion:
    """多信号融合引擎"""

    def __init__(self) -> None:
        self._window: Deque[SignalSnapshot] = deque()
        self._current_state: UserState = UserState.AWAY
        self._callbacks: List[StateChangeCallback] = []
        self._last_snapshot: Optional[StateSnapshot] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> UserState:
        return self._current_state

    @property
    def last_snapshot(self) -> Optional[StateSnapshot]:
        return self._last_snapshot

    def on_state_change(self, callback: StateChangeCallback) -> None:
        """注册状态变化回调"""
        self._callbacks.append(callback)

    def push(self, signal: SignalSnapshot) -> StateSnapshot:
        """
        推入新的信号采样，返回融合后的状态快照

        Args:
            signal: 单帧信号

        Returns:
            StateSnapshot
        """
        now = signal.timestamp
        self._window.append(signal)

        # 清理超出窗口的旧数据
        cutoff = now - STATE_WINDOW_SECONDS
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

        snapshot = self._fuse()
        old_state = self._current_state

        if snapshot.state != old_state:
            self._current_state = snapshot.state
            logger.info(
                "状态变化: %s -> %s (置信度 %.2f)",
                old_state.cn, snapshot.state.cn, snapshot.confidence,
            )
            for cb in self._callbacks:
                try:
                    cb(old_state, snapshot.state, snapshot)
                except Exception as exc:
                    logger.warning("状态回调异常: %s", exc)

        self._last_snapshot = snapshot
        return snapshot

    def reset(self) -> None:
        self._window.clear()
        self._current_state = UserState.AWAY
        self._last_snapshot = None

    # ------------------------------------------------------------------
    # 内部融合逻辑
    # ------------------------------------------------------------------

    def _fuse(self) -> StateSnapshot:
        """根据窗口内信号判定综合状态"""
        if not self._window:
            return StateSnapshot(
                state=UserState.AWAY, confidence=1.0,
            )

        latest = self._window[-1]

        # 基本统计
        face_frames = [s for s in self._window if s.face_detected]
        face_ratio = len(face_frames) / len(self._window) if self._window else 0.0

        # 1) 离开判定：大部分帧没检测到人脸
        if face_ratio < AWAY_CONFIDENCE_THRESHOLD or not latest.face_detected:
            return StateSnapshot(
                state=UserState.AWAY,
                confidence=1.0 - face_ratio,
                identity=latest.identity,
                dominant_emotion=latest.dominant_emotion,
                timestamp=latest.timestamp,
            )

        # 统计眨眼率（每分钟）
        blink_count = sum(1 for s in face_frames if s.blink)
        window_duration = max(
            face_frames[-1].timestamp - face_frames[0].timestamp, 1.0
        )
        blink_rate = (blink_count / window_duration) * 60.0

        # 平均 EAR
        avg_ear = sum(s.ear for s in face_frames) / len(face_frames) if face_frames else 0.3

        # 表情统计
        emotion_counts: Dict[str, int] = {}
        for s in face_frames:
            emotion_counts[s.dominant_emotion] = emotion_counts.get(s.dominant_emotion, 0) + 1
        dominant_in_window = max(emotion_counts, key=lambda k: emotion_counts[k]) if emotion_counts else "neutral"

        # 2) 疲劳判定：眨眼率高 或 平均 EAR 低
        fatigue_signals = 0
        if blink_rate > FATIGUE_BLINK_RATE:
            fatigue_signals += 1
        if avg_ear < FATIGUE_EAR_THRESHOLD:
            fatigue_signals += 1
        if dominant_in_window == "neutral" and blink_rate > FATIGUE_BLINK_RATE * 0.8:
            fatigue_signals += 1

        if fatigue_signals >= 2:
            return StateSnapshot(
                state=UserState.FATIGUED,
                confidence=min(1.0, fatigue_signals / 3.0),
                identity=latest.identity,
                dominant_emotion=dominant_in_window,
                blink_rate=blink_rate,
                avg_ear=avg_ear,
                yaw=latest.yaw,
                pitch=latest.pitch,
                timestamp=latest.timestamp,
            )

        # 3) 愉悦判定：窗口内大部分是 happy
        happy_ratio = emotion_counts.get("happy", 0) / len(face_frames) if face_frames else 0.0
        if happy_ratio > 0.5:
            return StateSnapshot(
                state=UserState.HAPPY,
                confidence=happy_ratio,
                identity=latest.identity,
                dominant_emotion="happy",
                blink_rate=blink_rate,
                avg_ear=avg_ear,
                yaw=latest.yaw,
                pitch=latest.pitch,
                timestamp=latest.timestamp,
            )

        # 4) 困惑判定：眉头紧锁 (sad/fear/disgust) + 偏头
        confused_emotions = {"sad", "fear", "disgust", "surprise"}
        confused_count = sum(
            emotion_counts.get(e, 0) for e in confused_emotions
        )
        confused_ratio = confused_count / len(face_frames) if face_frames else 0.0
        avg_yaw = sum(abs(s.yaw) for s in face_frames) / len(face_frames) if face_frames else 0.0

        if confused_ratio > 0.4 or (confused_ratio > 0.25 and avg_yaw > FOCUS_YAW_THRESHOLD):
            return StateSnapshot(
                state=UserState.CONFUSED,
                confidence=confused_ratio,
                identity=latest.identity,
                dominant_emotion=dominant_in_window,
                blink_rate=blink_rate,
                avg_ear=avg_ear,
                yaw=latest.yaw,
                pitch=latest.pitch,
                timestamp=latest.timestamp,
            )

        # 5) 默认：专注
        focus_confidence = 1.0 - abs(latest.yaw) / 90.0  # yaw 越小越专注
        return StateSnapshot(
            state=UserState.FOCUSED,
            confidence=max(0.0, min(1.0, focus_confidence)),
            identity=latest.identity,
            dominant_emotion=dominant_in_window,
            blink_rate=blink_rate,
            avg_ear=avg_ear,
            yaw=latest.yaw,
            pitch=latest.pitch,
            timestamp=latest.timestamp,
        )
