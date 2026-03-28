"""
cc_audio_out.py — 常驻音频输出流（解决打断+破音）

不再每句 sd.play() 开关流，而是维持一个持久 OutputStream。
TTS 音频写入缓冲区，打断时清空缓冲区即可（<50ms）。

用法：
    player = AudioPlayer()
    player.start()
    player.play(pcm, sr)         # 非阻塞，写入缓冲
    player.wait()                # 等播完
    player.interrupt()           # 立即停止
    player.stop()                # 关闭
"""

import collections
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable

DEFAULT_SR = 24000
BLOCK_SIZE = 1024  # ~42ms @ 24kHz


class AudioPlayer:
    """常驻音频输出流播放器"""

    def __init__(self, sample_rate: int = DEFAULT_SR):
        self._sr = sample_rate
        self._stream: Optional[sd.OutputStream] = None
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._playing = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._running = False
        self._current_energy: float = 0.0
        self.on_play_done: Optional[Callable] = None

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    @property
    def current_playback_energy(self) -> float:
        """当前正在播放的音频帧能量（用于回声估算）"""
        return self._current_energy

    def _update_energy(self, block):
        """更新当前播放能量"""
        self._current_energy = float(np.abs(block).mean()) if len(block) > 0 else 0.0

    def start(self):
        """启动常驻输出流"""
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """关闭输出流"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, pcm: np.ndarray, sample_rate: int = DEFAULT_SR):
        """
        播放 PCM（非阻塞）。
        如果采样率不同，简单重采样。
        """
        if sample_rate != self._sr:
            # 简单重采样
            ratio = self._sr / sample_rate
            indices = np.arange(0, len(pcm), 1 / ratio).astype(int)
            indices = indices[indices < len(pcm)]
            pcm = pcm[indices]

        # 切分为 block 放入缓冲
        self._done.clear()
        self._playing.set()
        with self._lock:
            for i in range(0, len(pcm), BLOCK_SIZE):
                block = pcm[i:i + BLOCK_SIZE]
                if len(block) < BLOCK_SIZE:
                    # 尾部补零
                    block = np.concatenate([block, np.zeros(BLOCK_SIZE - len(block), dtype=np.float32)])
                self._buffer.append(block)

    def wait(self, timeout: float = 30) -> bool:
        """等待播放完成"""
        return self._done.wait(timeout=timeout)

    def interrupt(self):
        """立即停止播放（清空缓冲区）"""
        with self._lock:
            self._buffer.clear()
        self._playing.clear()
        self._done.set()

    def _callback(self, outdata, frames, time_info, status):
        """sounddevice 回调：从缓冲区取数据"""
        with self._lock:
            if self._buffer:
                block = self._buffer.popleft()
                self._update_energy(block)
                outdata[:len(block), 0] = block[:frames]
                if len(block) < frames:
                    outdata[len(block):, 0] = 0
                # 检查是否播完
                if not self._buffer:
                    self._playing.clear()
                    self._done.set()
                    if self.on_play_done:
                        self.on_play_done()
            else:
                # 无数据，输出静音
                outdata[:, 0] = 0
                self._current_energy = 0.0
                if self._playing.is_set():
                    self._playing.clear()
                    self._done.set()
                    if self.on_play_done:
                        self.on_play_done()
