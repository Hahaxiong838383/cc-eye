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


def say(text: str, voice: Optional[str] = None) -> None:
    """cc 通过扬声器说话（edge-tts + afplay）"""
    asyncio.run(_speak(text, voice or VOICE))


async def _speak(text: str, voice: str) -> None:
    """异步生成语音并播放"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=RATE, pitch=PITCH)
    await communicate.save(AUDIO_PATH)
    subprocess.run(["afplay", AUDIO_PATH], check=True)


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
