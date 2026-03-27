"""
cc_events.py — 统一感知事件流（实时上下文窗口）

把视觉事件、语音事件、系统事件合并成一条时间线。
所有模块通过这个事件流共享上下文，Gemini 每次回复都能看到最近发生了什么。

用法：
    from cc_events import post_event, get_context_window
    post_event("speech", "川哥说：你好")
    context = get_context_window(seconds=120)
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import deque

# ── 配置 ──
EVENT_FILE = Path("/tmp/cc-eye-unified-events.jsonl")
WINDOW_SECONDS = 120  # 上下文窗口：最近 2 分钟
MAX_EVENTS = 200      # 内存中最多保留的事件数

# 线程安全的事件队列
_events: deque = deque(maxlen=MAX_EVENTS)
_lock = threading.Lock()


def post_event(event_type: str, detail: str, source: str = "system") -> None:
    """
    发布一个事件到统一事件流。

    Args:
        event_type: 事件类型（vision/speech/response/face/scene/system）
        detail: 事件描述
        source: 来源（daemon/interact/brain）
    """
    event = {
        "ts": datetime.now().isoformat(),
        "type": event_type,
        "detail": detail,
        "source": source,
    }

    with _lock:
        _events.append(event)

    # 同时写文件（供其他进程读取）
    try:
        with open(EVENT_FILE, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def get_context_window(seconds: int = WINDOW_SECONDS) -> str:
    """
    获取最近 N 秒的事件，格式化为 LLM 可读的上下文。

    Returns:
        格式化的事件时间线文本
    """
    cutoff = datetime.now() - timedelta(seconds=seconds)
    cutoff_str = cutoff.isoformat()

    # 先从内存读
    recent = []
    with _lock:
        for e in _events:
            if e["ts"] >= cutoff_str:
                recent.append(e)

    # 如果内存为空（可能是新启动），从文件补充
    if not recent:
        recent = _load_from_file(cutoff_str)

    if not recent:
        return ""

    # 格式化为时间线
    lines = ["[最近事件时间线]"]
    for e in recent[-30:]:  # 最多 30 条，避免上下文太长
        ts = e["ts"][11:19]  # HH:MM:SS
        etype = e["type"]
        detail = e["detail"][:150]  # 截断过长的描述

        icon = {
            "vision": "👁",
            "speech": "🎤",
            "response": "🗣",
            "face": "👤",
            "scene": "📷",
            "system": "⚙",
        }.get(etype, "•")

        lines.append(f"  {ts} {icon} {detail}")

    return "\n".join(lines)


def get_last_event(event_type: str) -> Optional[dict]:
    """获取指定类型的最新事件"""
    with _lock:
        for e in reversed(_events):
            if e["type"] == event_type:
                return e
    return None


def get_last_speech() -> Optional[str]:
    """获取最近一次用户说话内容"""
    e = get_last_event("speech")
    return e["detail"] if e else None


def get_last_scene() -> Optional[str]:
    """获取最近一次场景描述"""
    e = get_last_event("scene")
    return e["detail"] if e else None


def seconds_since_last_interaction() -> float:
    """距离上次语音交互过了多少秒"""
    with _lock:
        for e in reversed(_events):
            if e["type"] in ("speech", "response"):
                last_ts = datetime.fromisoformat(e["ts"])
                return (datetime.now() - last_ts).total_seconds()
    return float("inf")


def _load_from_file(cutoff_str: str) -> list:
    """从文件加载事件（进程间共享）"""
    if not EVENT_FILE.exists():
        return []
    recent = []
    try:
        for line in EVENT_FILE.read_text().splitlines()[-100:]:
            e = json.loads(line)
            if e["ts"] >= cutoff_str:
                recent.append(e)
    except Exception:
        pass
    return recent


def sync_from_daemon_events() -> None:
    """
    从 camera_daemon 的事件日志同步到统一事件流。
    daemon 写 /tmp/cc-eye-events.jsonl，这里转换格式合并进来。
    """
    daemon_file = Path("/tmp/cc-eye-events.jsonl")
    if not daemon_file.exists():
        return

    cutoff = (datetime.now() - timedelta(seconds=WINDOW_SECONDS)).isoformat()

    try:
        for line in daemon_file.read_text().splitlines()[-50:]:
            e = json.loads(line)
            if e["ts"] < cutoff:
                continue

            # 转换 daemon 事件类型到统一类型
            dtype = e.get("type", "")
            if dtype in ("fast_scan", "detail_scan", "scene_described"):
                post_event("scene", e.get("detail", ""), source="daemon")
            elif dtype == "person_appeared":
                post_event("face", e.get("detail", ""), source="daemon")
            elif dtype == "person_left":
                post_event("face", "人离开了", source="daemon")
            elif dtype == "motion":
                pass  # 运动事件太频繁，不同步
    except Exception:
        pass
