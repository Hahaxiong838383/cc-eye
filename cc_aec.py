"""
cc_aec.py — 回声消除模块（AEC: Acoustic Echo Cancellation）

原理：cc 自己 TTS 播放的音频波形是已知的（参考信号），
用 NLMS 自适应滤波器建模 扬声器→房间→麦克风 的传递函数，
从麦克风输入中减去估计的回声，只保留人声。

用法：
    aec = EchoCanceller()
    aec.set_reference("/tmp/cc-voice.mp3")  # 告诉 AEC 我要播什么
    # ... 播放音频 ...
    clean = aec.process(mic_chunk)  # 从麦克风数据中消除回声
"""

import numpy as np
import subprocess
import threading
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 16000


class EchoCanceller:
    """
    NLMS 自适应回声消除器。

    流程：
    1. say() 前调 set_reference(audio_path) 加载参考波形
    2. 播放开始时调 start_playback()
    3. 每次从麦克风读到数据，调 process(mic_chunk) 得到消除回声后的干净音频
    4. 播放结束后调 stop_playback()，进入尾部消散期
    """

    def __init__(
        self,
        filter_length: int = 4096,
        mu: float = 0.05,
        tail_duration: float = 0.5,
    ):
        """
        Args:
            filter_length: 自适应滤波器长度（采样点数）。越长能消除越远的回声，但越慢。
                          4096 @ 16kHz ≈ 256ms，足够覆盖普通房间的混响。
            mu: NLMS 步长。越大收敛越快但可能不稳定。0.05 是保守值。
            tail_duration: 播放结束后继续消除的时长（秒），等混响消散。
        """
        self.filter_length = filter_length
        self.mu = mu
        self.tail_duration = tail_duration

        # 自适应滤波器权重
        self._weights = np.zeros(filter_length, dtype=np.float32)
        # 参考信号缓冲（环形，用于卷积）
        self._ref_buffer = np.zeros(filter_length, dtype=np.float32)

        # 参考音频（完整波形）
        self._reference: Optional[np.ndarray] = None
        self._ref_pos = 0  # 当前读到哪里
        self._lock = threading.Lock()

        # 状态
        self._is_playing = False
        self._play_ended_time = 0.0
        self._active = False  # 是否有有效参考信号

    def set_reference(self, audio_path: str) -> bool:
        """
        加载 TTS 输出作为参考信号。支持 mp3/wav。
        在 say() 之前调用。
        """
        path = Path(audio_path)
        if not path.exists():
            return False

        try:
            # 用 ffmpeg 转为 16kHz mono PCM
            wav_path = "/tmp/cc-aec-ref.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "wav",
                    wav_path,
                ],
                capture_output=True,
                check=True,
            )

            import soundfile as sf
            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            with self._lock:
                self._reference = data
                self._ref_pos = 0
                self._active = True
                # 重置滤波器（新的参考信号，重新学习）
                self._weights = np.zeros(self.filter_length, dtype=np.float32)
                self._ref_buffer = np.zeros(self.filter_length, dtype=np.float32)

            return True
        except Exception as e:
            print(f"[cc-aec] 加载参考信号失败: {e}")
            return False

    def feed_reference_pcm(self, chunk: np.ndarray) -> None:
        """
        实时喂入参考信号 PCM（来自 InterruptablePlayer 的每帧回调）。

        用于新架构：播放器每帧回调 → 降采样到 16kHz → 喂给 AEC。
        替代旧的 set_reference(file) 方式。
        """
        with self._lock:
            if self._reference is None:
                self._reference = chunk.astype(np.float32)
            else:
                self._reference = np.concatenate([self._reference, chunk.astype(np.float32)])
            self._active = True

    def start_playback(self) -> None:
        """标记播放开始，重置参考信号"""
        with self._lock:
            self._is_playing = True
            self._ref_pos = 0
            self._reference = None  # 清空旧参考，等 feed_reference_pcm 实时填充
            self._active = True

    def stop_playback(self) -> None:
        """标记播放结束，进入尾部消散期"""
        import time
        with self._lock:
            self._is_playing = False
            self._play_ended_time = time.time()

    @property
    def is_active(self) -> bool:
        """AEC 是否处于激活状态（正在播放或在尾部消散期内）"""
        if self._is_playing:
            return True
        if not self._active:
            return False
        import time
        return (time.time() - self._play_ended_time) < self.tail_duration

    def process(self, mic_chunk: np.ndarray) -> np.ndarray:
        """
        处理一段麦克风音频，消除回声。

        Args:
            mic_chunk: 麦克风输入（float32，单声道）

        Returns:
            消除回声后的干净音频（同长度）
        """
        if not self._active or self._reference is None:
            return mic_chunk

        with self._lock:
            ref = self._reference
            ref_pos = self._ref_pos

        output = np.zeros_like(mic_chunk)
        ref_len = len(ref)

        for i in range(len(mic_chunk)):
            # 获取参考信号采样点
            if ref_pos < ref_len:
                ref_sample = ref[ref_pos]
                ref_pos += 1
            elif self.is_active:
                # 参考信号放完了但还在消散期，用 0
                ref_sample = 0.0
            else:
                # AEC 不再激活，直接透传
                output[i:] = mic_chunk[i:]
                break

            # 更新参考缓冲（移位 + 新样本）
            self._ref_buffer = np.roll(self._ref_buffer, 1)
            self._ref_buffer[0] = ref_sample

            # NLMS: 估计回声
            echo_estimate = np.dot(self._weights, self._ref_buffer)

            # 误差 = 麦克风 - 回声估计 = 干净人声（期望信号）
            error = mic_chunk[i] - echo_estimate

            # NLMS 权重更新
            norm = np.dot(self._ref_buffer, self._ref_buffer) + 1e-8
            self._weights += (self.mu * error / norm) * self._ref_buffer

            output[i] = error

        with self._lock:
            self._ref_pos = ref_pos

        return output

    def reset(self) -> None:
        """完全重置 AEC 状态"""
        with self._lock:
            self._weights = np.zeros(self.filter_length, dtype=np.float32)
            self._ref_buffer = np.zeros(self.filter_length, dtype=np.float32)
            self._reference = None
            self._ref_pos = 0
            self._is_playing = False
            self._active = False
