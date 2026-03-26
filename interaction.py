"""
interaction.py -- 互动引擎
根据状态变化生成 cc 风格的回应文本，并在画面上绘制对话气泡
"""

import time
import random
import logging
from typing import Optional, Dict, List

import cv2
import numpy as np

from state_fusion import UserState
from config import INTERACTION_COOLDOWN, GREETING_ABSENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# cc 风格回应文本库
# ------------------------------------------------------------------

GREETINGS: Dict[str, List[str]] = {
    "known": [
        "{name}，回来了？继续干活",
        "嘿 {name}，准备好了吗",
        "{name}！坐稳了，开搞",
        "欢迎回来 {name}，我一直在",
    ],
    "first_time": [
        "嗨 {name}，认识你很高兴",
        "{name}，以后就是搭档了",
        "记住你了 {name}，下次秒认",
    ],
    "long_absence": [
        "{name}，好久不见啊",
        "以为你不来了呢 {name}",
        "{name} 终于回来了，想你了（并没有）",
    ],
}

STATE_RESPONSES: Dict[str, List[str]] = {
    "to_fatigued": [
        "眼睛都快睁不开了，休息一下？",
        "别硬撑，去喝杯水吧",
        "我看你在打瞌睡...",
        "疲劳驾驶可不行，先歇歇",
    ],
    "to_happy": [
        "心情不错嘛",
        "笑什么呢，分享一下？",
        "看你开心我也开心",
        "这表情，有好事？",
    ],
    "to_confused": [
        "哪里卡住了？说说看",
        "眉头都皱成一团了",
        "需要帮忙理一下思路吗",
        "别着急，一步一步来",
    ],
    "to_focused": [
        "进入状态了，不打扰你",
        "专注模式 ON",
        "这个状态可以，继续保持",
    ],
    "to_away": [
        "...",
        "人呢？",
    ],
    "fatigue_to_focused": [
        "回血了？继续",
        "精神头上来了，冲",
    ],
}


class InteractionEngine:
    """互动引擎：生成回应 + 绘制对话气泡"""

    def __init__(self) -> None:
        # 冷却管理：event_type -> last_trigger_time
        self._cooldowns: Dict[str, float] = {}
        # 当前显示的文本 + 显示开始时间
        self._current_text: str = ""
        self._text_show_time: float = 0.0
        self._text_duration: float = 5.0  # 文本显示持续秒数
        # 身份相关
        self._last_seen: Dict[str, float] = {}
        self._greeted: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_greeting(self, name: str, is_new: bool = False) -> Optional[str]:
        """
        生成问候语

        Args:
            name: 用户名
            is_new: 是否首次注册

        Returns:
            问候文本，如果冷却中返回 None
        """
        if not self._check_cooldown(f"greet_{name}"):
            return None

        now = time.time()
        last_seen = self._last_seen.get(name, 0.0)
        self._last_seen[name] = now

        if is_new:
            pool = GREETINGS["first_time"]
        elif (now - last_seen) > GREETING_ABSENCE_THRESHOLD and last_seen > 0:
            pool = GREETINGS["long_absence"]
        else:
            pool = GREETINGS["known"]

        text = random.choice(pool).format(name=name)
        self._set_text(text)
        return text

    def get_response(self, old_state: UserState, new_state: UserState) -> Optional[str]:
        """
        状态变化时生成回应

        Returns:
            回应文本，冷却中返回 None
        """
        event_key = f"{old_state.value}_to_{new_state.value}"

        if not self._check_cooldown(event_key):
            return None

        # 特殊组合
        if old_state == UserState.FATIGUED and new_state == UserState.FOCUSED:
            pool = STATE_RESPONSES.get("fatigue_to_focused", STATE_RESPONSES["to_focused"])
        else:
            pool = STATE_RESPONSES.get(f"to_{new_state.value}", [])

        if not pool:
            return None

        text = random.choice(pool)
        self._set_text(text)
        return text

    def draw_bubble(self, frame: np.ndarray, text: Optional[str] = None) -> np.ndarray:
        """
        在画面右下角绘制对话气泡

        Args:
            frame: BGR 帧（会被修改）
            text: 指定文本，None 则使用当前缓存文本

        Returns:
            带气泡的帧
        """
        display_text = text if text is not None else self._get_display_text()
        if not display_text:
            return frame

        h, w = frame.shape[:2]

        # 文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        max_width = min(400, w // 3)
        line_height = 28
        padding = 12

        # 文字分行（按字符宽度简单分割）
        lines = self._wrap_text(display_text, font, font_scale, thickness, max_width - 2 * padding)

        # 气泡尺寸
        text_h = len(lines) * line_height
        bubble_w = max_width
        bubble_h = text_h + 2 * padding

        # 位置：右下角，留 20px 边距
        x1 = w - bubble_w - 20
        y1 = h - bubble_h - 20
        x2 = x1 + bubble_w
        y2 = y1 + bubble_h

        # 半透明圆角矩形背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), cv2.FILLED)
        # 圆角效果（用圆填充四角）
        radius = 10
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, (40, 40, 40), cv2.FILLED)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, (40, 40, 40), cv2.FILLED)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, (40, 40, 40), cv2.FILLED)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, (40, 40, 40), cv2.FILLED)

        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # "cc:" 标签
        label_y = y1 + padding + 18
        cv2.putText(frame, "cc:", (x1 + padding, label_y),
                    font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        # 文字内容
        for i, line in enumerate(lines):
            ty = y1 + padding + 18 + (i + 1) * line_height
            cv2.putText(frame, line, (x1 + padding, ty),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _check_cooldown(self, event_key: str) -> bool:
        """检查冷却，通过则记录时间并返回 True"""
        now = time.time()
        last = self._cooldowns.get(event_key, 0.0)
        if (now - last) < INTERACTION_COOLDOWN:
            return False
        self._cooldowns[event_key] = now
        return True

    def _set_text(self, text: str, duration: float = 5.0) -> None:
        self._current_text = text
        self._text_show_time = time.time()
        self._text_duration = duration

    def _get_display_text(self) -> str:
        """获取当前应显示的文本（过期返回空串）"""
        if not self._current_text:
            return ""
        if (time.time() - self._text_show_time) > self._text_duration:
            return ""
        return self._current_text

    @staticmethod
    def _wrap_text(
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_pixel_width: int,
    ) -> List[str]:
        """将文本按像素宽度分行"""
        lines: List[str] = []
        current = ""
        for ch in text:
            test = current + ch
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw > max_pixel_width and current:
                lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        return lines if lines else [""]
