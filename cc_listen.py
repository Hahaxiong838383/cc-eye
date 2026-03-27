"""
cc_listen.py — cc 贾维斯语音输入（VAD + SenseVoice）

基于能量的 VAD 自动切分语音段，配合 SenseVoice 做本地 STT。
SenseVoice 额外提供语音情感识别和音频事件检测。
不需要手动按键，说话就识别，停顿就结束。

用法：
    from cc_listen import listen_once, listen_loop

架构：
    麦克风 → sounddevice 持续采样 → 能量 VAD 检测说话
    → 录制语音段 → SenseVoice 转文字 + 情感 + 事件 → 返回结果
"""

import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import threading
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, field

# ── 配置 ──
SAMPLE_RATE = 16000          # whisper 最佳采样率
CHANNELS = 1
BLOCK_SIZE = 1024            # 每次采样帧数（~64ms @ 16kHz）
ENERGY_THRESHOLD = 0.008     # 说话能量阈值（根据 EMEET 麦克风调整）
SILENCE_DURATION = 0.8       # 静音多久判定说话结束（秒）— 流式管线需要更快响应
MIN_SPEECH_DURATION = 0.5    # 最短有效语音段（秒），过短的丢弃
MAX_SPEECH_DURATION = 30.0   # 最长语音段（秒），超过强制截断
PRE_SPEECH_BUFFER = 0.3      # 说话前缓冲（秒），保留起始音
AUDIO_PATH = "/tmp/cc-listen-segment.wav"

# ── SenseVoice 模型（懒加载）──
_sensevoice_model = None
_sensevoice_lock = threading.Lock()

# SenseVoice 情感标签映射（英文 → 中文）
EMOTION_MAP = {
    "HAPPY": "开心",
    "SAD": "难过",
    "ANGRY": "生气",
    "NEUTRAL": "平静",
    "FEARFUL": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}

# SenseVoice 音频事件标签
AUDIO_EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Crying", "Coughing", "Sneezing", "Breath"}

# 解析 SenseVoice 原始输出的正则
_TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")


def _get_sensevoice():
    """懒加载 SenseVoice 模型（首次调用时加载，后续复用）"""
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model
    with _sensevoice_lock:
        if _sensevoice_model is not None:
            return _sensevoice_model
        from funasr import AutoModel
        print("[cc-listen] 加载 SenseVoiceSmall 模型（ModelScope）...")
        _sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
            trust_remote_code=True,
            disable_update=True,
        )
        print("[cc-listen] SenseVoiceSmall 就绪")
        return _sensevoice_model


def _parse_sensevoice_tags(raw_text: str) -> tuple:
    """
    解析 SenseVoice 原始输出中的标签。
    输出格式: <|zh|><|HAPPY|><|Speech|><|withitn|>识别文本

    Returns:
        (clean_text, language, emotion, audio_events)
    """
    tags = _TAG_PATTERN.findall(raw_text)
    # 去掉所有标签得到纯文本
    clean = _TAG_PATTERN.sub("", raw_text).strip()

    language = "zh"
    emotion = "neutral"
    audio_events: List[str] = []

    for tag in tags:
        # 语言标签
        if tag in ("zh", "en", "yue", "ja", "ko"):
            language = tag
        # 情感标签（EMO_UNKNOWN 视为 neutral）
        elif tag.upper() in EMOTION_MAP:
            emotion = tag.lower()
        elif tag == "EMO_UNKNOWN":
            emotion = "neutral"
        # 音频事件标签
        elif tag in AUDIO_EVENTS:
            audio_events.append(tag.lower())
        # withitn / notitn 忽略

    return clean, language, emotion, audio_events


@dataclass
class SpeechSegment:
    """一段语音的识别结果（含情感和音频事件）"""
    text: str
    duration: float             # 语音时长（秒）
    confidence: float           # 平均置信度
    language: str               # 检测到的语言
    emotion: str = "neutral"    # 语音情感: happy/sad/angry/neutral/fearful/disgusted/surprised
    emotion_cn: str = "平静"     # 情感中文
    audio_events: List[str] = field(default_factory=list)  # 音频事件: laughter/applause/bgm...


def _calculate_energy(audio_block: np.ndarray) -> float:
    """计算音频块的 RMS 能量"""
    return float(np.sqrt(np.mean(audio_block ** 2)))


def listen_once(
    timeout: float = 30.0,
    aec: Optional[object] = None,
) -> Optional[SpeechSegment]:
    """
    监听一次语音输入（带 NLMS 回声消除）。

    等待用户开始说话 → 录制到停顿 → whisper 识别 → 返回文本。
    如果 timeout 秒内没有检测到说话，返回 None。

    Args:
        timeout: 等待说话超时（秒）
        aec: EchoCanceller 实例。播放 TTS 时自动用参考信号消除回声。
    """
    audio_buffer = []
    pre_buffer = []  # 说话前的缓冲帧
    is_speaking = False
    silence_start = 0.0
    speech_start = 0.0
    wait_start = time.time()

    pre_buffer_frames = int(PRE_SPEECH_BUFFER * SAMPLE_RATE / BLOCK_SIZE)

    # 用 sounddevice InputStream 持续采样
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    )
    stream.start()

    try:
        while True:
            data, _ = stream.read(BLOCK_SIZE)
            audio_chunk = data[:, 0] if data.ndim > 1 else data.flatten()

            # AEC：用 NLMS 自适应滤波消除 TTS 回声
            if aec is not None:
                audio_chunk = aec.process(audio_chunk)

            energy = _calculate_energy(audio_chunk)

            if not is_speaking:
                # 维护前缓冲（环形）
                pre_buffer.append(audio_chunk.copy())
                if len(pre_buffer) > pre_buffer_frames:
                    pre_buffer.pop(0)

                # 检测说话开始
                if energy > ENERGY_THRESHOLD:
                    is_speaking = True
                    speech_start = time.time()
                    # 把前缓冲加入录音
                    audio_buffer.extend(pre_buffer)
                    audio_buffer.append(audio_chunk.copy())
                    pre_buffer.clear()
                elif time.time() - wait_start > timeout:
                    return None  # 超时没说话
            else:
                audio_buffer.append(audio_chunk.copy())

                if energy > ENERGY_THRESHOLD:
                    silence_start = 0.0  # 还在说话，重置静音计时
                else:
                    if silence_start == 0.0:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break  # 静音够久，说话结束

                # 语音段过长，强制截断
                if time.time() - speech_start > MAX_SPEECH_DURATION:
                    break
    finally:
        stream.stop()
        stream.close()

    # 检查最短时长
    duration = time.time() - speech_start
    if duration < MIN_SPEECH_DURATION:
        return None  # 太短，可能是噪音

    # 拼接音频并保存
    audio = np.concatenate(audio_buffer)
    sf.write(AUDIO_PATH, audio, SAMPLE_RATE)

    # whisper 识别
    return _transcribe(AUDIO_PATH, duration)


def _transcribe(audio_path: str, duration: float) -> Optional[SpeechSegment]:
    """用 SenseVoice 识别语音文件（含情感 + 音频事件）"""
    model = _get_sensevoice()
    start = time.time()

    try:
        res = model.generate(
            input=audio_path,
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
    except Exception as e:
        print(f"[cc-listen] SenseVoice 推理失败: {e}")
        return None

    elapsed = time.time() - start

    if not res or not res[0].get("text"):
        return None

    raw_text = res[0]["text"]

    # 先从原始输出解析情感和事件标签
    clean_text, language, emotion, audio_events = _parse_sensevoice_tags(raw_text)

    # 再用官方后处理清理文本（保险起见）
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        clean_text = rich_transcription_postprocess(raw_text)
    except ImportError:
        pass  # 已经用正则清理过了

    if not clean_text:
        return None

    emotion_cn = EMOTION_MAP.get(emotion.upper(), "平静")

    print(f"[cc-listen] SenseVoice ({elapsed:.1f}s): {clean_text}")
    if emotion != "neutral":
        print(f"[cc-listen] 语音情感: {emotion_cn} ({emotion})")
    if audio_events:
        print(f"[cc-listen] 音频事件: {audio_events}")

    return SpeechSegment(
        text=clean_text,
        duration=duration,
        confidence=0.0,
        language=language,
        emotion=emotion,
        emotion_cn=emotion_cn,
        audio_events=audio_events,
    )


def listen_loop(
    callback: Callable[[SpeechSegment], None],
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    持续监听语音，每识别出一段就调用 callback。

    Args:
        callback: 回调函数，参数是 SpeechSegment
        stop_event: 外部停止信号（Event），set() 后退出循环
    """
    print("[cc-listen] 持续监听模式启动...")
    if stop_event is None:
        stop_event = threading.Event()

    while not stop_event.is_set():
        segment = listen_once(timeout=60.0)
        if segment and not stop_event.is_set():
            callback(segment)


def calibrate_mic(duration: float = 3.0) -> float:
    """
    校准麦克风环境噪音，返回建议的能量阈值。
    安静环境下运行 3 秒，取平均能量的 3 倍作为阈值。
    """
    print(f"[cc-listen] 校准麦克风（{duration}秒，请保持安静）...")
    energies = []

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
    )
    stream.start()

    end_time = time.time() + duration
    while time.time() < end_time:
        data, _ = stream.read(BLOCK_SIZE)
        audio_chunk = data[:, 0] if data.ndim > 1 else data.flatten()
        energies.append(_calculate_energy(audio_chunk))

    stream.stop()
    stream.close()

    avg_energy = np.mean(energies)
    suggested = avg_energy * 3.0
    print(f"[cc-listen] 环境噪音能量: {avg_energy:.6f}")
    print(f"[cc-listen] 建议阈值: {suggested:.6f}（当前设置: {ENERGY_THRESHOLD}）")
    return suggested


if __name__ == "__main__":
    print("=== cc 贾维斯语音监听测试（SenseVoice）===")
    print("先校准麦克风...")
    threshold = calibrate_mic()
    print(f"\n现在说话测试（30 秒超时）...")
    result = listen_once(timeout=30.0)
    if result:
        print(f"\n识别结果: {result.text}")
        print(f"时长: {result.duration:.1f}s | 语言: {result.language}")
        print(f"语音情感: {result.emotion_cn} ({result.emotion})")
        if result.audio_events:
            print(f"音频事件: {result.audio_events}")
    else:
        print("未检测到语音")
