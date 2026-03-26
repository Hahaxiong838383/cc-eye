"""
expression.py -- 表情识别模块
基于 DeepFace 内置表情分析（不依赖 FER / dlib）
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from config import EXPRESSION_SMOOTHING

logger = logging.getLogger(__name__)

# 7 类表情 -> 中文标签
EMOTION_CN: Dict[str, str] = {
    "happy": "开心",
    "sad": "难过",
    "angry": "生气",
    "surprise": "惊讶",
    "fear": "恐惧",
    "disgust": "厌恶",
    "neutral": "平静",
}

ALL_EMOTIONS = list(EMOTION_CN.keys())


@dataclass
class ExpressionResult:
    """表情识别结果"""
    dominant_emotion: str           # 主导表情 (英文 key)
    emotions: Dict[str, float]      # 7 类表情置信度
    confidence: float               # 主导表情的置信度

    @property
    def dominant_cn(self) -> str:
        return EMOTION_CN.get(self.dominant_emotion, self.dominant_emotion)


class ExpressionRecognizer:
    """表情识别器（基于 DeepFace），带 EMA 平滑"""

    def __init__(self) -> None:
        self._smoothed: Optional[Dict[str, float]] = None
        self._alpha = EXPRESSION_SMOOTHING
        self._ready = False
        self._init_detector()

    def _init_detector(self) -> None:
        """预加载 DeepFace 表情分析模型"""
        try:
            from deepface import DeepFace
            self._DeepFace = DeepFace
            self._ready = True
            logger.info("DeepFace 表情识别器初始化完成")
        except Exception as exc:
            logger.error("DeepFace 初始化失败: %s", exc)
            self._ready = False

    def detect(self, frame: np.ndarray) -> Optional[ExpressionResult]:
        """
        检测帧中最大人脸的表情

        Args:
            frame: BGR 格式 numpy 数组

        Returns:
            ExpressionResult 或 None（未检测到 / 不可用）
        """
        if not self._ready:
            return None

        try:
            results = self._DeepFace.analyze(
                frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:
            logger.debug("表情检测异常: %s", exc)
            return None

        if not results:
            return None

        # DeepFace.analyze 返回 list[dict] 或 dict
        if isinstance(results, list):
            analysis = results[0]
        else:
            analysis = results

        raw_emotions = analysis.get("emotion", {})
        if not raw_emotions:
            return None

        # DeepFace 的 emotion 值是百分比 (0-100)，归一化到 0-1
        normalized: Dict[str, float] = {}
        for emotion in ALL_EMOTIONS:
            normalized[emotion] = raw_emotions.get(emotion, 0.0) / 100.0

        # EMA 平滑
        smoothed = self._apply_ema(normalized)

        # 找主导表情
        dominant = max(smoothed, key=lambda k: smoothed[k])
        confidence = smoothed[dominant]

        return ExpressionResult(
            dominant_emotion=dominant,
            emotions=dict(smoothed),
            confidence=confidence,
        )

    def _apply_ema(self, raw: Dict[str, float]) -> Dict[str, float]:
        """指数移动平均平滑"""
        if self._smoothed is None:
            self._smoothed = dict(raw)
            return dict(raw)

        alpha = self._alpha
        smoothed = {}
        for emotion in ALL_EMOTIONS:
            old_val = self._smoothed.get(emotion, 0.0)
            new_val = raw.get(emotion, 0.0)
            smoothed[emotion] = alpha * new_val + (1.0 - alpha) * old_val

        self._smoothed = smoothed
        return dict(smoothed)

    def reset(self) -> None:
        """重置平滑状态"""
        self._smoothed = None
