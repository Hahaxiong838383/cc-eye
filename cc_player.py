"""
cc_player.py — cc 贾维斯可打断音频播放器

替代 afplay/ffplay 阻塞式播放，使用 sounddevice.OutputStream 直接输出 PCM。
核心能力：正在播放时可以随时调用 stop() 在 50ms 内停止。

用法：
    from cc_player import InterruptablePlayer, tts_to_pcm_stream
    player = InterruptablePlayer()
    player.play_pcm(pcm_data)          # 非阻塞播放 PCM
    player.stop()                       # 立即停止（<50ms）

    # 流式 TTS 播放
    import asyncio
    chunks = asyncio.run(tts_to_pcm_stream("你好"))
    player.play_tts_stream(chunks)
"""

import asyncio
import io
import threading
import time
import collections
from typing import Optional, Callable, Generator

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# ── 默认配置 ──
DEFAULT_SAMPLE_RATE = 24000  # edge-tts 默认输出 24kHz
_MISSING = object()          # 队列空标记（区分 None sentinel 和队列空）
DEFAULT_CHANNELS = 1
FRAME_SIZE = 1024            # 每帧采样点数（~42ms @ 24kHz，足够保证 <50ms 打断响应）


def _mp3_bytes_to_pcm(mp3_bytes: bytes, target_sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    将 MP3 字节解码为 float32 PCM（mono）。

    Args:
        mp3_bytes: MP3 编码的原始字节
        target_sr: 目标采样率

    Returns:
        float32 ndarray，值域 [-1.0, 1.0]，mono
    """
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))

    # 转 mono
    if seg.channels > 1:
        seg = seg.set_channels(1)

    # resample（如果需要）
    if seg.frame_rate != target_sr:
        seg = seg.set_frame_rate(target_sr)

    # 提取 raw samples → float32
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    # pydub 输出 int16 范围，归一化到 [-1.0, 1.0]
    samples /= 32768.0

    return samples


class InterruptablePlayer:
    """
    可打断的音频播放器。

    使用 sounddevice.OutputStream 直接输出 PCM 到扬声器。
    播放在独立线程中进行，stop() 在 50ms 内生效。

    AEC 联动接口：
        on_play_start:  播放开始时回调
        on_play_stop:   播放停止时回调
        on_pcm_frame:   每帧 PCM 播放时回调（用于 AEC 参考信号）
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        on_play_start: Optional[Callable] = None,
        on_play_stop: Optional[Callable] = None,
        on_pcm_frame: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self._sample_rate = sample_rate
        self._on_play_start = on_play_start
        self._on_play_stop = on_play_stop
        self._on_pcm_frame = on_pcm_frame

        # 播放队列：TTS 流式往里放 PCM 帧，播放线程从里取
        self._queue: collections.deque = collections.deque()

        # 线程控制
        self._stop_flag = threading.Event()
        self._playing = threading.Event()     # 是否正在播放
        self._done_event = threading.Event()  # 播放完成信号
        self._done_event.set()                # 初始状态：无播放任务
        self._lock = threading.Lock()
        self._play_thread: Optional[threading.Thread] = None

    @property
    def is_playing(self) -> bool:
        """当前是否正在播放"""
        return self._playing.is_set()

    def play_pcm(
        self,
        pcm_data: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        on_pcm_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        播放 PCM 数据（非阻塞，开新线程播放）。

        Args:
            pcm_data: float32 ndarray，mono，值域 [-1.0, 1.0]
            sample_rate: 采样率
            on_pcm_frame: 每帧回调（覆盖构造函数设置）
        """
        # 先停止当前播放
        self.stop()

        frame_callback = on_pcm_frame or self._on_pcm_frame

        # 切分为帧放入队列
        with self._lock:
            self._queue.clear()
            for i in range(0, len(pcm_data), FRAME_SIZE):
                frame = pcm_data[i:i + FRAME_SIZE]
                self._queue.append(frame)

        self._stop_flag.clear()
        self._done_event.clear()
        self._playing.set()

        self._play_thread = threading.Thread(
            target=self._playback_worker,
            args=(sample_rate, frame_callback, False),
            daemon=True,
            name="cc-player",
        )
        self._play_thread.start()

    def play_tts_stream(
        self,
        tts_chunks: Generator,
        on_pcm_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        """
        从 edge-tts 流式消费音频并播放。

        tts_chunks 是 edge-tts communicate.stream() 产出的 mp3 chunk。
        解码为 PCM 后放入播放队列，播放线程实时消费。

        Args:
            tts_chunks: Generator，yield MP3 bytes
            on_pcm_frame: 每帧回调（覆盖构造函数设置）
        """
        # 先停止当前播放
        self.stop()

        frame_callback = on_pcm_frame or self._on_pcm_frame

        with self._lock:
            self._queue.clear()

        self._stop_flag.clear()
        self._done_event.clear()
        self._playing.set()

        # 启动播放线程（流式模式，等待队列数据）
        self._play_thread = threading.Thread(
            target=self._playback_worker,
            args=(self._sample_rate, frame_callback, True),
            daemon=True,
            name="cc-player-stream",
        )
        self._play_thread.start()

        # 在当前线程解码 TTS chunk 并喂入队列
        mp3_buffer = bytearray()
        for chunk in tts_chunks:
            if self._stop_flag.is_set():
                break
            mp3_buffer.extend(chunk)

        # 所有 chunk 收完，解码整段 mp3
        if mp3_buffer and not self._stop_flag.is_set():
            try:
                pcm = _mp3_bytes_to_pcm(bytes(mp3_buffer), self._sample_rate)
                frames = [pcm[i:i + FRAME_SIZE] for i in range(0, len(pcm), FRAME_SIZE)]
                with self._lock:
                    self._queue.extend(frames)
            except Exception as e:
                print(f"[cc-player] MP3 解码失败: {e}")

        # 标记流结束
        with self._lock:
            self._queue.append(None)  # sentinel：流结束

    def stop(self) -> None:
        """
        立即停止播放。

        设 stop_flag + 清空队列，播放线程检测到后在当前帧结束时退出。
        保证 50ms 内生效（FRAME_SIZE=1024 @ 24kHz ≈ 42ms）。
        """
        self._stop_flag.set()
        with self._lock:
            self._queue.clear()

        # 等待播放线程退出（超时保护）
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=0.2)
            self._play_thread = None

        self._playing.clear()

    def wait(self) -> None:
        """等待当前播放完成"""
        self._done_event.wait()

    def _playback_worker(
        self,
        sample_rate: int,
        frame_callback: Optional[Callable],
        is_stream: bool,
    ) -> None:
        """
        播放线程：从队列取帧，通过 sounddevice.OutputStream 输出。

        Args:
            sample_rate: 采样率
            frame_callback: 每帧 PCM 回调
            is_stream: 是否流式模式（等待数据）
        """
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=DEFAULT_CHANNELS,
                dtype="float32",
                blocksize=FRAME_SIZE,
            )
            stream.start()

            # 通知播放开始
            if self._on_play_start:
                try:
                    self._on_play_start()
                except Exception:
                    pass

            empty_count = 0
            max_empty_waits = 500  # 流式模式最多等 5 秒（500 * 10ms）
            got_sentinel = False

            while not self._stop_flag.is_set():
                frame = _MISSING
                with self._lock:
                    if self._queue:
                        frame = self._queue.popleft()

                # 从队列取到了 None sentinel → 流结束
                if frame is None:
                    got_sentinel = True
                    # 还有剩余帧就继续消费，否则退出
                    with self._lock:
                        if not self._queue:
                            break
                    continue

                # 队列空（未取到任何东西）
                if frame is _MISSING:
                    if not is_stream or got_sentinel:
                        break  # 非流式 / 已收到 sentinel 且队列空
                    empty_count += 1
                    if empty_count > max_empty_waits:
                        break  # 等太久了，退出
                    time.sleep(0.01)
                    continue

                empty_count = 0

                # 确保帧是正确形状（sounddevice 要求 2D）
                if frame.ndim == 1:
                    frame = frame.reshape(-1, 1)

                # 写入音频设备
                stream.write(frame)

                # AEC 参考信号回调
                if frame_callback:
                    try:
                        frame_callback(frame.flatten())
                    except Exception:
                        pass

        except Exception as e:
            print(f"[cc-player] 播放异常: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

            self._playing.clear()
            self._done_event.set()

            # 通知播放停止
            if self._on_play_stop:
                try:
                    self._on_play_stop()
                except Exception:
                    pass


async def tts_to_pcm_stream(
    text: str,
    voice: str = "zh-CN-YunjianNeural",
    rate: str = "-5%",
    pitch: str = "-10Hz",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Generator[bytes, None, None]:
    """
    调用 edge-tts 流式合成，yield MP3 字节块。

    不落盘，直接在内存中产出。配合 InterruptablePlayer.play_tts_stream() 使用。

    Args:
        text: 要合成的文本
        voice: edge-tts 音色
        rate: 语速调整
        pitch: 音调调整
        sample_rate: 目标采样率（用于解码时 resample）

    Yields:
        MP3 字节块
    """
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


def _collect_tts_chunks(text: str, voice: str, rate: str, pitch: str) -> list:
    """同步收集 edge-tts 产出的 mp3 chunk 列表"""
    chunks = []

    async def _collect():
        async for data in tts_to_pcm_stream(text, voice, rate, pitch):
            chunks.append(data)

    asyncio.run(_collect())
    return chunks


# ── 测试 ──

if __name__ == "__main__":
    print("=== cc 可打断播放器测试 ===\n")

    # 从 cc_voice 获取音色配置
    try:
        from cc_voice import VOICE, RATE, PITCH
    except ImportError:
        VOICE = "zh-CN-YunjianNeural"
        RATE = "-5%"
        PITCH = "-10Hz"

    player = InterruptablePlayer(
        on_play_start=lambda: print("[test] >>> 播放开始"),
        on_play_stop=lambda: print("[test] >>> 播放停止"),
        on_pcm_frame=lambda f: None,  # 静默回调，不打印
    )

    # ── 测试 1：正常播放一句话 ──
    print("[test] 合成语音...")
    chunks = _collect_tts_chunks("川哥你好，我是贾维斯，可打断播放器测试中。", VOICE, RATE, PITCH)

    # 拼接所有 mp3 chunk 解码为 PCM
    mp3_all = b"".join(chunks)
    pcm = _mp3_bytes_to_pcm(mp3_all)
    print(f"[test] PCM: {len(pcm)} samples, {len(pcm)/DEFAULT_SAMPLE_RATE:.2f}s")

    print("[test] 开始播放...")
    player.play_pcm(pcm)

    # ── 测试 2：2 秒后打断 ──
    time.sleep(2.0)
    print(f"\n[test] 2 秒到，打断！ is_playing={player.is_playing}")
    t0 = time.time()
    player.stop()
    elapsed_ms = (time.time() - t0) * 1000
    print(f"[test] stop() 耗时: {elapsed_ms:.1f}ms（要求 <50ms）")

    if elapsed_ms < 50:
        print("[test] PASS — 打断在 50ms 内生效")
    else:
        print(f"[test] WARN — 打断耗时 {elapsed_ms:.1f}ms，超过 50ms 目标")

    print(f"[test] is_playing={player.is_playing}")

    # ── 测试 3：流式 TTS 播放 ──
    print("\n[test] 流式 TTS 播放测试...")
    stream_chunks = _collect_tts_chunks("这是流式播放测试，你应该能听到这句话。", VOICE, RATE, PITCH)
    player.play_tts_stream(iter(stream_chunks))
    player.wait()

    print("\n[test] 全部测试完成")
