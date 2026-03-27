"""
cc_voice.py — cc 贾维斯语音分身（edge-tts）

使用微软 edge-tts（云希 YunxiNeural），丝滑自然的中文男声。
共享 cc_context 的身份和记忆，通过 Mac 扬声器说话。

用法：
    from cc_voice import say
    say("川哥早上好")
"""

import asyncio
import subprocess
from typing import Optional

# 贾维斯音色：云希（沉稳、自然、丝滑）
VOICE = "zh-CN-YunjianNeural"
RATE = "-5%"
PITCH = "-10Hz"
AUDIO_PATH = "/tmp/cc-voice.mp3"

# 全局 AEC 实例（cc_interact 启动时注入）
_aec = None

# 最近播放的文本（用于文本层回声过滤）
import threading
_recent_tts_texts: list = []  # 最近播放的文本，cc_listen 用来过滤回声
_tts_lock = threading.Lock()


def set_aec(aec) -> None:
    """注入 AEC 回声消除器实例"""
    global _aec
    _aec = aec


def say(text: str, voice: Optional[str] = None) -> None:
    """cc 通过扬声器说话（edge-tts + afplay + AEC 联动）"""
    asyncio.run(_speak(text, voice or VOICE))


async def _speak(text: str, voice: str) -> None:
    """异步生成语音并播放，同步通知 AEC"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=RATE, pitch=PITCH)
    await communicate.save(AUDIO_PATH)

    # 记录播放文本（用于文本层回声过滤）
    with _tts_lock:
        _recent_tts_texts.append(text)
        if len(_recent_tts_texts) > 5:
            _recent_tts_texts.pop(0)

    subprocess.run(["afplay", AUDIO_PATH], check=True)


def is_echo(text: str, threshold: float = 0.5) -> bool:
    """判断 whisper 识别的文本是否是自己 TTS 的回声。
    用字符重叠率判断：如果识别文本中超过 50% 的字符出现在最近播放的文本里，认为是回声。"""
    if not text or len(text) < 3:
        return False
    with _tts_lock:
        recent = list(_recent_tts_texts)
    for tts_text in recent:
        # 计算字符重叠率
        common = sum(1 for c in text if c in tts_text)
        ratio = common / len(text)
        if ratio > threshold:
            return True
    return False


def greet() -> None:
    """cc 贾维斯打招呼"""
    try:
        from cc_context import get_scene_context
        scene = get_scene_context()
        face_count = scene.get("face_count", 0) if scene else 0
        if face_count > 0:
            say("川哥好，我是cc贾维斯，有什么需要帮忙的？")
        else:
            say("办公室目前没有人，我在后台值班。")
    except Exception:
        say("你好，我是cc贾维斯。")


if __name__ == "__main__":
    say("川哥早上好，cc贾维斯语音分身已上线，共享记忆和认知，随时听你指挥。")
