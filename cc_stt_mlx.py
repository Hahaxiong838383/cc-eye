"""
cc_stt_mlx.py — Qwen3-ASR MLX 语音识别（M5 GPU 加速）

替代 SenseVoice，延迟从 500ms 降到 ~130ms。

用法：
    from cc_stt_mlx import transcribe
    text = transcribe("/tmp/audio.wav")
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

_MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"
_model = None


@dataclass
class STTResult:
    text: str
    language: str = "Chinese"
    emotion: Optional[str] = None
    emotion_cn: Optional[str] = None
    audio_events: Optional[list] = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from mlx_audio.stt.utils import load_model
    print("[cc-stt] 加载 Qwen3-ASR MLX...")
    _model = load_model(_MODEL_ID)
    # 预热
    import tempfile, numpy as np, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
        _model.generate(f.name)
    print("[cc-stt] Qwen3-ASR 就绪 (~130ms/句)")
    return _model


def transcribe(audio_path: str, duration: float = 0) -> Optional[STTResult]:
    """
    转写音频文件，返回 STTResult。

    Args:
        audio_path: WAV 文件路径
        duration: 音频时长（仅日志用）
    """
    model = _get_model()
    try:
        result = model.generate(audio_path)
        text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
        if not text:
            return None
        return STTResult(text=text)
    except Exception as e:
        print(f"[cc-stt] 转写失败: {e}")
        return None


def preload():
    """预加载模型"""
    _get_model()
