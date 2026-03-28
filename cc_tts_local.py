"""
cc_tts_local.py — 本地 TTS（Qwen3-TTS Base + 锁定音色）

音色锁定：Base 模型 + ref_audio（从 VoiceDesign 生成的参考音频）
每次合成使用同一段参考音频，speaker embedding 固定，音色不漂移。

用法：
    from cc_tts_local import local_tts_to_pcm, preload
    pcm, sr = local_tts_to_pcm("你好")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from cc_voice_profile import BASE_MODEL, REF_AUDIO_PATH, REF_TEXT

_tts_model = None


def _get_model():
    """懒加载 Base TTS 模型"""
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    import warnings
    warnings.filterwarnings("ignore")
    from mlx_audio.tts.utils import load_model

    _tts_model = load_model(BASE_MODEL)
    print("[cc-tts] Qwen3-TTS 1.7B Base 就绪（ref_audio 音色锁定）")
    return _tts_model


def _validate_ref_audio() -> bool:
    """检查参考音频是否存在"""
    if not REF_AUDIO_PATH.exists():
        print(f"[cc-tts] 警告：参考音频不存在 {REF_AUDIO_PATH}")
        print("[cc-tts] 请运行 python scripts/gen_jarvis_ref.py 生成")
        return False
    return True


def local_tts_to_pcm(
    text: str,
    speaker: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    本地合成语音（Base 模型 + ref_audio 音色锁定）。
    优先查缓存（命中 <1ms）。
    """
    if text in _audio_cache:
        return _audio_cache[text]

    model = _get_model()

    if not _validate_ref_audio():
        # 无参考音频时降级：直接用 base 模型无音色合成
        results = list(model.generate(text=text, lang_code="auto"))
    else:
        # 正常路径：ref_audio 锁定音色
        results = list(model.generate(
            text=text,
            ref_audio=str(REF_AUDIO_PATH),
            ref_text=REF_TEXT,
            lang_code="auto",
        ))

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
    流式合成，yield (pcm_chunk, sample_rate)。
    Base 模型 + ref_audio，支持 stream=True。
    """
    model = _get_model()

    if not _validate_ref_audio():
        return

    for result in model.generate(
        text=text,
        ref_audio=str(REF_AUDIO_PATH),
        ref_text=REF_TEXT,
        lang_code="auto",
        stream=True,
        streaming_interval=0.3,
    ):
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
    # 基础应答（≥3字）
    "我在的。", "好的呢。", "收到了。", "你说吧。", "嗯，你说。", "明白了。",
    "知道了。", "没问题。", "好，我在。", "在的呢。",
    "怎么了？", "随时待命。", "需要我做什么？", "还有别的事吗？",
    # 过渡语气
    "让我想想。", "这个嘛。", "好问题。", "有点意思。", "稍等一下。",
    "这个方向挺好的。", "我梳理一下思路。", "让我整理下信息。", "这个值得聊聊。",
    "我看看。", "等我想想。", "有意思。", "好，稍等。",
    # 云端衔接过渡（联网/搜索/深度）
    "我看看最新的。", "让我查一下。", "稍等，我确认下。", "我搜一下。",
    "这个值得展开说说。", "让我理一下。", "我想想怎么解决。",
    "让我看看方案。", "让我确认一下。", "我核实下。", "我分析一下。",
    # 安静模式
    "好的，我听着。",
    # 时间问候
    "早上好。", "上午好。", "中午好。", "下午好。", "晚上好。",
    "还在忙？", "夜深了。", "该休息了。",
    # 贾维斯经典台词风格
    "所有系统运行正常。", "一切就绪，随时待命。",
    "我不建议这么做。", "这不是个好主意。",
    "已经准备好了。", "正在处理中。", "分析完成。",
    "有个情况需要你注意。", "检测到异常，建议你看一下。",
    "遵命，马上处理。", "马上处理。",
    "我必须提醒你。", "数据分析显示。", "根据目前的信息。",
    "随时可以开始。", "今天的行程已经更新。", "有新的消息需要处理。",
    "这超出了我的能力范围，但我可以试试。",
    "建议你休息一下。", "你已经连续工作很久了。",
    "一切尽在掌控。", "情况比预期的要好。", "情况有些复杂。",
    "放心，交给我。", "收到，开始处理。", "明白你的意思。",
]

# 缓存文件路径
_CACHE_DIR = Path(__file__).parent / ".venv" / "cache"
_CACHE_FILE = _CACHE_DIR / "tts_cache_base.npz"  # 新文件名，和旧 VoiceDesign 缓存区分


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
    """预加载模型 + 加载/生成缓存（使用 Base + ref_audio 音色锁定）"""
    model = _get_model()

    if not _validate_ref_audio():
        print("[cc-tts] 无参考音频，跳过预缓存")
        return

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
            results = list(model.generate(
                text=phrase,
                ref_audio=str(REF_AUDIO_PATH),
                ref_text=REF_TEXT,
                lang_code="auto",
            ))
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
