"""
cc_voice.py — cc 的语音分身（TTS 输出）

共享 cc_context 的身份和记忆，通过 Mac 扬声器说话。

用法：
    from cc_voice import say
    say("川哥早上好")
    say("今天有3个待办事项")
"""

import pyttsx3
from typing import Optional

_engine: Optional[pyttsx3.Engine] = None


def _get_engine() -> pyttsx3.Engine:
    """获取或初始化 TTS 引擎（单例）"""
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 160)
        _engine.setProperty('volume', 1.0)
        # 贾维斯风格：Reed 中文男声（沉稳低沉）
        _engine.setProperty('voice', 'com.apple.eloquence.zh-CN.Reed')
    return _engine


def say(text: str) -> None:
    """cc 通过扬声器说话"""
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()


def greet() -> None:
    """cc 打招呼（读取当前状态生成问候语）"""
    try:
        from cc_context import get_scene_context
        scene = get_scene_context()
        face_count = scene.get('face_count', 0) if scene else 0
        if face_count > 0:
            say("川哥好，我在呢。有什么需要帮忙的？")
        else:
            say("办公室目前没有人，我在后台值班。")
    except Exception:
        say("你好，我是 cc。")


if __name__ == "__main__":
    say("川哥早上好，cc 语音分身已上线，共享记忆和认知，随时听你指挥。")
