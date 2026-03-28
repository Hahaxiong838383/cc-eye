"""
cc_vad.py — Silero VAD 封装（ONNX Runtime 推理，无 torch 依赖）

替代 cc_listen.py 中基于能量阈值的 VAD，使用 Silero VAD 神经网络模型
精准判断人声，显著减少误触发（键盘声、风扇声等不再误判为说话）。

用法：
    from cc_vad import SileroVAD, SpeechSegmenter

    vad = SileroVAD(threshold=0.5)
    prob = vad.get_speech_prob(audio_chunk)  # [0, 1]
    is_speech = vad.is_speech(audio_chunk)   # bool

    segmenter = SpeechSegmenter(on_speech_end=callback)
    segmenter.feed(audio_chunk)

架构：
    麦克风 → sounddevice 采样 → SileroVAD(ONNX) 逐帧判断人声概率
    → SpeechSegmenter 维护状态机（SILENCE/SPEECH/TRAILING_SILENCE）
    → 检测到完整语音段后回调 on_speech_end(audio_segment)
"""

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import onnxruntime as ort

# ── 日志 ──
logger = logging.getLogger("cc-vad")

# ── 模型路径 ──
_PROJECT_DIR = Path(__file__).parent
_MODEL_DIR = _PROJECT_DIR / ".venv" / "models"
_MODEL_PATH = _MODEL_DIR / "silero_vad.onnx"
_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)


def _ensure_model() -> Path:
    """确保 ONNX 模型文件存在，不存在则自动下载。"""
    if _MODEL_PATH.exists():
        return _MODEL_PATH

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("下载 Silero VAD ONNX 模型: %s", _MODEL_URL)

    try:
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
    except Exception as exc:
        # 清理不完整的下载
        _MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"Silero VAD 模型下载失败: {exc}\n"
            f"请手动下载到 {_MODEL_PATH}:\n"
            f"  curl -fSL -o {_MODEL_PATH} {_MODEL_URL}"
        ) from exc

    size_kb = _MODEL_PATH.stat().st_size / 1024
    logger.info("模型已下载: %.0f KB -> %s", size_kb, _MODEL_PATH)
    return _MODEL_PATH


# ── Silero VAD ONNX 推理 ──


class SileroVAD:
    """
    Silero VAD 封装，基于 PyTorch 推理（官方推荐方式）。

    核心方法:
        is_speech(chunk)       — 判断是否包含人声
        get_speech_prob(chunk) — 返回人声概率 [0, 1]
        reset()                — 重置内部状态（新对话轮次时调用）

    Args:
        threshold:   人声判定阈值，默认 0.5
        sample_rate: 采样率，必须为 16000
        window_size: 每帧样本数，Silero VAD 要求 512 @ 16kHz（32ms）
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        window_size: int = 512,
    ):
        if sample_rate != 16000:
            raise ValueError(
                f"Silero VAD 仅支持 16kHz 采样率，收到 {sample_rate}"
            )

        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size = window_size

        # 加载 torch 版 Silero VAD
        import torch
        self._torch = torch
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()

        logger.info(
            "SileroVAD 就绪 (torch): threshold=%.2f, window=%d samples (%.0fms)",
            threshold,
            window_size,
            window_size / sample_rate * 1000,
        )

    def reset(self) -> None:
        """重置内部状态（新的对话轮次时调用）。"""
        self._model.reset_states()

    def get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        返回一个 audio chunk 的人声概率 [0, 1]。

        Args:
            audio_chunk: float32 数组，长度必须等于 window_size。
                         值域 [-1, 1]（16kHz 单声道）。

        Returns:
            人声概率，0.0 = 纯静音/噪音，1.0 = 确定是人声。
        """
        chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()

        if len(chunk) != self.window_size:
            raise ValueError(
                f"chunk 长度必须为 {self.window_size}，收到 {len(chunk)}"
            )

        tensor = self._torch.from_numpy(chunk)
        prob = self._model(tensor, self.sample_rate).item()
        return prob

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """判断 audio chunk 是否包含人声。"""
        return self.get_speech_prob(audio_chunk) >= self.threshold


# ── 语音段切分状态机 ──


class _SegState(Enum):
    """SpeechSegmenter 内部状态。"""
    SILENCE = auto()
    SPEECH = auto()
    TRAILING_SILENCE = auto()


@dataclass
class SpeechSegmenterConfig:
    """SpeechSegmenter 配置。"""
    threshold: float = 0.5          # 人声判定阈值
    min_speech_ms: int = 200        # 最短有效语音段（毫秒）
    min_silence_ms: int = 300       # 静音多久算说完（毫秒）— 极速响应
    sample_rate: int = 16000
    window_size: int = 512          # Silero VAD 帧大小
    pre_speech_ms: int = 200        # 语音前缓冲（保留起始音）


class SpeechSegmenter:
    """
    基于 SileroVAD 的连续语音切段器。

    通过 feed() 持续喂入音频，内部维护 ring buffer 和状态机，
    检测到"说话开始 → 静音结束"时回调 on_speech_end(audio_segment)。

    状态转换:
        SILENCE → SPEECH          当连续帧 > threshold
        SPEECH → TRAILING_SILENCE 当帧 < threshold
        TRAILING_SILENCE → SPEECH 如果静音期间又检测到人声（继续录）
        TRAILING_SILENCE → SILENCE 静音超过 min_silence_ms → 触发回调

    用法:
        def on_speech(audio: np.ndarray):
            print(f"检测到语音段: {len(audio)/16000:.1f}s")

        seg = SpeechSegmenter(on_speech_end=on_speech)
        # 在录音循环中:
        seg.feed(audio_chunk_512_samples)
    """

    def __init__(
        self,
        on_speech_end: Callable[[np.ndarray], None],
        config: Optional[SpeechSegmenterConfig] = None,
        vad: Optional[SileroVAD] = None,
    ):
        self._cfg = config or SpeechSegmenterConfig()
        self._vad = vad or SileroVAD(
            threshold=self._cfg.threshold,
            sample_rate=self._cfg.sample_rate,
            window_size=self._cfg.window_size,
        )
        self._on_speech_end = on_speech_end

        # 帧时长（毫秒）
        self._frame_ms = self._cfg.window_size / self._cfg.sample_rate * 1000

        # 状态
        self._state = _SegState.SILENCE
        self._speech_frames: list[np.ndarray] = []
        self._trailing_silence_ms: float = 0.0
        self._speech_ms: float = 0.0

        # 前缓冲（ring buffer，保留起始音）
        pre_frames = max(1, int(self._cfg.pre_speech_ms / self._frame_ms))
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=pre_frames)

    @property
    def state(self) -> str:
        """当前状态名称。"""
        return self._state.name

    def reset(self) -> None:
        """重置状态（新的对话轮次）。"""
        self._vad.reset()
        self._state = _SegState.SILENCE
        self._speech_frames.clear()
        self._trailing_silence_ms = 0.0
        self._speech_ms = 0.0
        self._pre_buffer.clear()

    def feed(self, audio_chunk: np.ndarray) -> None:
        """
        喂入一帧音频（长度 = window_size）。

        如果 chunk 长度不是 window_size，会自动按 window_size 拆分处理。
        尾部不足 window_size 的部分会被丢弃。
        """
        chunk = np.asarray(audio_chunk, dtype=np.float32).flatten()
        ws = self._cfg.window_size

        # 支持喂入任意长度，按 window_size 拆帧
        offset = 0
        while offset + ws <= len(chunk):
            frame = chunk[offset : offset + ws]
            self._process_frame(frame)
            offset += ws

    def _process_frame(self, frame: np.ndarray) -> None:
        """处理单帧音频。"""
        is_speech = self._vad.is_speech(frame)

        if self._state == _SegState.SILENCE:
            if is_speech:
                # 说话开始：把前缓冲和当前帧一起放入录音
                self._speech_frames.clear()
                self._speech_frames.extend(self._pre_buffer)
                self._speech_frames.append(frame.copy())
                self._speech_ms = len(self._speech_frames) * self._frame_ms
                self._state = _SegState.SPEECH
            else:
                # 持续静音，维护前缓冲
                self._pre_buffer.append(frame.copy())

        elif self._state == _SegState.SPEECH:
            self._speech_frames.append(frame.copy())
            self._speech_ms += self._frame_ms

            if not is_speech:
                # 可能要结束了，进入尾部静音
                self._trailing_silence_ms = self._frame_ms
                self._state = _SegState.TRAILING_SILENCE

        elif self._state == _SegState.TRAILING_SILENCE:
            self._speech_frames.append(frame.copy())

            if is_speech:
                # 又说话了，回到 SPEECH
                self._trailing_silence_ms = 0.0
                self._speech_ms += self._frame_ms
                self._state = _SegState.SPEECH
            else:
                self._trailing_silence_ms += self._frame_ms

                if self._trailing_silence_ms >= self._cfg.min_silence_ms:
                    # 静音够久，语音段结束
                    self._finalize_segment()

    def _finalize_segment(self) -> None:
        """结束当前语音段，触发回调。"""
        # 检查最短有效语音时长
        if self._speech_ms >= self._cfg.min_speech_ms:
            audio = np.concatenate(self._speech_frames)
            self._on_speech_end(audio)
        else:
            logger.debug(
                "丢弃过短语音段: %.0fms < %dms",
                self._speech_ms,
                self._cfg.min_speech_ms,
            )

        # 重置状态（不重置 VAD 内部状态，因为可能在同一轮对话中）
        self._speech_frames.clear()
        self._trailing_silence_ms = 0.0
        self._speech_ms = 0.0
        self._pre_buffer.clear()
        self._state = _SegState.SILENCE


# ── 测试入口 ──

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    print("=" * 60)
    print("Silero VAD 测试 — 对着麦克风说话，观察检测结果")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    try:
        import sounddevice as sd
    except ImportError:
        print("需要 sounddevice: pip install sounddevice")
        sys.exit(1)

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 512
    FRAME_MS = WINDOW_SIZE / SAMPLE_RATE * 1000  # 32ms

    # 初始化 VAD
    vad = SileroVAD(threshold=0.5, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE)

    # 测试1：逐帧打印人声概率
    print(f"\n--- 模式1：实时概率监控（{FRAME_MS:.0f}ms/帧）---")
    print("说话时概率应 > 0.5，静音时 < 0.1\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=WINDOW_SIZE,
        dtype="float32",
    )
    stream.start()

    speech_count = 0
    total_count = 0

    try:
        for _ in range(int(5000 / FRAME_MS)):  # 5 秒
            data, _ = stream.read(WINDOW_SIZE)
            chunk = data[:, 0] if data.ndim > 1 else data.flatten()
            prob = vad.get_speech_prob(chunk)
            total_count += 1
            if prob >= 0.5:
                speech_count += 1

            # 用进度条可视化
            bar_len = int(prob * 40)
            bar = "#" * bar_len + "-" * (40 - bar_len)
            tag = "SPEECH" if prob >= 0.5 else "      "
            print(f"\r  [{bar}] {prob:.3f} {tag}", end="", flush=True)
    except KeyboardInterrupt:
        pass

    stream.stop()
    stream.close()
    vad.reset()

    print(f"\n\n5 秒内检测到人声帧: {speech_count}/{total_count}")

    # 测试2：SpeechSegmenter 语音段切分
    print("\n--- 模式2：语音段切分（说话 → 停顿 → 回调）---")
    print("说一句话，停顿后会显示检测到的语音段时长")
    print("10 秒后自动结束\n")

    segments_found: list[float] = []

    def on_segment(audio: np.ndarray):
        dur = len(audio) / SAMPLE_RATE
        segments_found.append(dur)
        print(f"\n  >> 检测到语音段: {dur:.2f}s ({len(audio)} samples)")

    segmenter = SpeechSegmenter(on_speech_end=on_segment)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=WINDOW_SIZE,
        dtype="float32",
    )
    stream.start()

    start_time = time.time()
    try:
        while time.time() - start_time < 10:
            data, _ = stream.read(WINDOW_SIZE)
            chunk = data[:, 0] if data.ndim > 1 else data.flatten()
            segmenter.feed(chunk)

            # 显示状态
            state = segmenter.state
            indicator = {"SILENCE": ".", "SPEECH": "*", "TRAILING_SILENCE": "~"}
            print(indicator.get(state, "?"), end="", flush=True)
    except KeyboardInterrupt:
        pass

    stream.stop()
    stream.close()

    print(f"\n\n共检测到 {len(segments_found)} 个语音段")
    for i, dur in enumerate(segments_found):
        print(f"  段{i+1}: {dur:.2f}s")
    print("\n测试完成。")
