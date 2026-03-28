"""
cc_tts_local.py — 本地 TTS（Qwen3-TTS 1.7B VoiceDesign MLX）

主模型：VoiceDesign（定制贾维斯声音）
降级：CustomVoice aiden

用法：
    from cc_tts_local import local_tts_to_pcm
    pcm, sr = local_tts_to_pcm("你好")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from cc_voice_profile import JARVIS_VOICE_INSTRUCT, VOICE_DESIGN_MODEL, CUSTOM_VOICE_MODEL, FALLBACK_SPEAKER

_tts_model = None
_tts_mode = "voice_design"  # "voice_design" 或 "custom_voice"


def _get_model():
    """懒加载 TTS 模型（优先 VoiceDesign）"""
    global _tts_model, _tts_mode
    if _tts_model is not None:
        return _tts_model

    import warnings
    warnings.filterwarnings("ignore")
    from mlx_audio.tts.utils import load_model

    try:
        _tts_model = load_model(VOICE_DESIGN_MODEL)
        _tts_mode = "voice_design"
        print("[cc-tts] Qwen3-TTS 1.7B VoiceDesign 就绪（贾维斯定制声音）")
    except Exception as e:
        print(f"[cc-tts] VoiceDesign 加载失败: {e}，降级到 CustomVoice")
        _tts_model = load_model(CUSTOM_VOICE_MODEL)
        _tts_mode = "custom_voice"
        print(f"[cc-tts] CustomVoice 就绪 (speaker={FALLBACK_SPEAKER})")
    return _tts_model


def set_speaker(name: str) -> str:
    """切换音色（仅 custom_voice 模式）"""
    pass  # VoiceDesign 模式下不需要切换
    print(f"[cc-tts] 音色切换: {_current_speaker}")
    return _current_speaker


def local_tts_to_pcm(
    text: str,
    speaker: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """本地合成语音，优先查缓存（命中 <1ms）"""
    if text in _audio_cache:
        return _audio_cache[text]

    model = _get_model()

    if _tts_mode == "voice_design":
        results = list(model.generate_voice_design(
            text=text,
            language="Chinese",
            instruct=JARVIS_VOICE_INSTRUCT,
        ))
    else:
        results = list(model.generate_custom_voice(text=text, speaker=FALLBACK_SPEAKER))
    r = results[0]
    samples = np.array(r.audio, dtype=np.float32)

    # 归一化
    peak = np.abs(samples).max()
    if peak > 0.001:
        samples = samples * (0.95 / peak)

    # 开头加 50ms 静音 + 20ms 淡入（消除 sd.play 开流爆音）
    silence = np.zeros(int(r.sample_rate * 0.05), dtype=np.float32)
    fade_len = min(int(r.sample_rate * 0.02), len(samples))
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len, dtype=np.float32)
        samples[:fade_len] *= fade
    samples = np.concatenate([silence, samples])

    return samples, r.sample_rate


def local_tts_stream(text: str, speaker: Optional[str] = None):
    """
    流式合成，yield (pcm_chunk, sample_rate)。首包 ~230ms。
    """
    model = _get_model()
    spk = speaker or _current_speaker
    for result in model.generate_custom_voice(text=text, speaker=spk, stream=True, streaming_interval=0.3):
        samples = np.array(result.audio, dtype=np.float32)
        if len(samples) == 0:
            continue
        peak = np.abs(samples).max()
        if peak > 0.001:
            samples = samples * (0.9 / peak)
        yield samples, result.sample_rate


# ── 预缓存常用短回复 ──
_audio_cache: dict = {}

_PRECACHE_PHRASES = [
    # 基础应答
    "在。", "我在。", "嗯。", "好。", "好的。", "行。", "收到。",
    "说吧。", "嗯嗯。", "明白。", "知道了。", "没问题。",
    "好，我在。", "在的。", "怎么了。", "随时待命。",
    # 过渡语气
    "让我想想。", "这个嘛。", "好问题。", "有点意思。", "稍等。",
    "这个方向挺好的。", "我梳理一下思路。", "让我整理下信息。", "这个值得聊聊。",
    # 安静模式
    "好的，我听着。",
    # 时间问候
    "早。", "上午好。", "中午好。", "下午好。", "晚上好。", "还在忙？", "夜深了。",
    # 贾维斯经典台词风格
    "先生，所有系统运行正常。", "一切就绪，随时待命。",
    "先生，我不建议这么做。", "这不是个好主意，先生。",
    "已经为您准备好了。", "正在处理中。", "分析完成。",
    "先生，有个情况需要你注意。", "检测到异常，建议你看一下。",
    "遵命。", "马上处理。", "已完成。",
    "先生，我必须提醒你。", "数据分析显示。", "根据目前的信息。",
    "需要我做什么？", "还有别的事吗？", "随时可以开始。",
    "今天的行程已经更新。", "有新的消息需要处理。",
    "这超出了我的能力范围，但我可以试试。",
    "建议你休息一下。", "你已经连续工作很久了。",
    "一切尽在掌控。", "情况比预期的要好。", "情况有些复杂。",
]

# 缓存文件路径
_CACHE_DIR = Path(__file__).parent / ".venv" / "cache"
_CACHE_FILE = _CACHE_DIR / "tts_cache.npz"


def _load_cache_from_disk():
    """从磁盘加载缓存"""
    if not _CACHE_FILE.exists():
        return 0
    try:
        data = np.load(str(_CACHE_FILE), allow_pickle=True)
        meta = data["meta"].item()  # dict: {phrase: sample_rate}
        count = 0
        for phrase, sr in meta.items():
            key = f"audio_{count}"
            if key in data:
                _audio_cache[phrase] = (data[key], int(sr))
                count += 1
        print(f"[cc-tts] 从磁盘加载缓存: {count} 条")
        return count
    except Exception as e:
        print(f"[cc-tts] 缓存加载失败: {e}")
        return 0


def _save_cache_to_disk():
    """固化缓存到磁盘"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        save_dict = {}
        meta = {}
        for i, (phrase, (audio, sr)) in enumerate(_audio_cache.items()):
            save_dict[f"audio_{i}"] = audio
            meta[phrase] = sr
        save_dict["meta"] = np.array(meta, dtype=object)
        np.savez(str(_CACHE_FILE), **save_dict)
        print(f"[cc-tts] 缓存已固化: {len(_audio_cache)} 条 → {_CACHE_FILE}")
    except Exception as e:
        print(f"[cc-tts] 缓存固化失败: {e}")


def preload():
    """预加载模型 + 加载/生成缓存"""
    model = _get_model()

    # 先从磁盘加载
    loaded = _load_cache_from_disk()

    # 找出缺失的短句
    missing = [p for p in _PRECACHE_PHRASES if p not in _audio_cache]
    if not missing:
        print(f"[cc-tts] 缓存完整，{loaded} 条全部命中")
        return

    print(f"[cc-tts] 合成缺失短句: {len(missing)} 条...")
    for phrase in missing:
        try:
            if _tts_mode == "voice_design":
                results = list(model.generate_voice_design(
                    text=phrase, language="Chinese", instruct=JARVIS_VOICE_INSTRUCT,
                ))
            else:
                results = list(model.generate_custom_voice(text=phrase, speaker=FALLBACK_SPEAKER))
            samples = np.array(results[0].audio, dtype=np.float32)
            peak = np.abs(samples).max()
            if peak > 0.001:
                samples = samples * (0.95 / peak)
            silence = np.zeros(int(results[0].sample_rate * 0.05), dtype=np.float32)
            samples = np.concatenate([silence, samples])
            _audio_cache[phrase] = (samples, results[0].sample_rate)
        except Exception:
            pass

    # 固化到磁盘
    _save_cache_to_disk()
    print(f"[cc-tts] 预缓存完成: {len(_audio_cache)} 条")
