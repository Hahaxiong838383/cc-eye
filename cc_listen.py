"""
cc_listen.py — cc 贾维斯语音输入（VAD + faster-whisper）

基于能量的 VAD 自动切分语音段，配合 faster-whisper 做本地 STT。
不需要手动按键，说话就识别，停顿就结束。

用法：
    from cc_listen import listen_once, listen_loop

架构：
    麦克风 → sounddevice 持续采样 → 能量 VAD 检测说话
    → 录制语音段 → faster-whisper 转文字 → 返回文本
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

# ── 配置 ──
SAMPLE_RATE = 16000          # whisper 最佳采样率
CHANNELS = 1
BLOCK_SIZE = 1024            # 每次采样帧数（~64ms @ 16kHz）
ENERGY_THRESHOLD = 0.008     # 说话能量阈值（根据 EMEET 麦克风调整）
SILENCE_DURATION = 1.5       # 静音多久判定说话结束（秒）
MIN_SPEECH_DURATION = 0.5    # 最短有效语音段（秒），过短的丢弃
MAX_SPEECH_DURATION = 30.0   # 最长语音段（秒），超过强制截断
PRE_SPEECH_BUFFER = 0.3      # 说话前缓冲（秒），保留起始音
AUDIO_PATH = "/tmp/cc-listen-segment.wav"

# ── whisper 模型（懒加载）──
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    """懒加载 faster-whisper 模型（首次调用时加载，后续复用）"""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        from faster_whisper import WhisperModel
        print("[cc-listen] 加载 whisper medium 模型（中文精度更高）...")
        _whisper_model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8",
        )
        print("[cc-listen] whisper medium 就绪")
        return _whisper_model


@dataclass
class SpeechSegment:
    """一段语音的识别结果"""
    text: str
    duration: float        # 语音时长（秒）
    confidence: float      # 平均置信度
    language: str          # 检测到的语言


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
    """用 faster-whisper 识别语音文件"""
    model = _get_whisper()
    start = time.time()

    segments, info = model.transcribe(
        audio_path,
        language="zh",
        beam_size=5,
        initial_prompt="贾维斯，川哥，你好，看看，环境，摄像头，帮我，打开，关闭，时间",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    texts = []
    confidences = []
    for seg in segments:
        texts.append(seg.text.strip())
        confidences.append(seg.avg_logprob)

    text = "".join(texts).strip()
    elapsed = time.time() - start

    if not text:
        return None

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print(f"[cc-listen] 识别完成 ({elapsed:.1f}s): {text}")
    return SpeechSegment(
        text=text,
        duration=duration,
        confidence=avg_conf,
        language=info.language,
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
    print("=== cc 贾维斯语音监听测试 ===")
    print("先校准麦克风...")
    threshold = calibrate_mic()
    print(f"\n现在说话测试（30 秒超时）...")
    result = listen_once(timeout=30.0)
    if result:
        print(f"\n识别结果: {result.text}")
        print(f"时长: {result.duration:.1f}s | 语言: {result.language}")
    else:
        print("未检测到语音")
