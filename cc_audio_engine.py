"""
cc_audio_engine.py — 全双工音频引擎（通过 Swift VP 桥接）

通过 cc_audio_bridge 子进程实现硬件级 AEC 全双工。
麦克风音频经 macOS Voice Processing 消除回声后送给 Python。
TTS 音频通过管道送给 Swift 播放。

用法：
    engine = AudioBridge()
    engine.start(on_mic_audio=my_callback)
    engine.play(pcm_float32, sample_rate=24000)
    engine.stop()
"""

import struct
import subprocess
import threading
import numpy as np
from pathlib import Path
from typing import Callable, Optional

# 协议常量（与 Swift 端一致）
MSG_PLAY: int     = 0x01
MSG_STOP: int     = 0x02
MSG_EXIT: int     = 0x03
MSG_MIC: int      = 0x10
MSG_PLAY_DONE: int = 0x11
MSG_READY: int    = 0x12
MSG_ERROR: int    = 0x13

BRIDGE_PATH = Path(__file__).parent / "cc_audio_bridge"


class AudioBridge:
    """
    全双工音频引擎。

    通过 Swift cc_audio_bridge 子进程获得 macOS 硬件级 AEC。
    麦克风数据经 VP 消回声后通过 on_mic_chunk 回调送出。
    TTS 音频通过 play() 方法送给扬声器播放。
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._on_mic_chunk: Optional[Callable[[np.ndarray], None]] = None
        self._on_play_done: Optional[Callable[[], None]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = threading.Event()
        self._play_done_event = threading.Event()
        self._play_done_event.set()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def start(
        self,
        on_mic_chunk: Callable[[np.ndarray], None],
        on_play_done: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        启动音频桥接。

        Args:
            on_mic_chunk: 麦克风回调，参数为 float32 ndarray (16kHz mono, 512 samples)
            on_play_done: 播放完成回调
        """
        if not BRIDGE_PATH.exists():
            print(f"[audio-bridge] 未找到 {BRIDGE_PATH}，请先编译")
            return False

        self._on_mic_chunk = on_mic_chunk
        self._on_play_done = on_play_done
        self._running = True

        # 启动 Swift 子进程
        self._proc = subprocess.Popen(
            [str(BRIDGE_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # 启动 stderr 日志线程
        threading.Thread(
            target=self._stderr_reader,
            daemon=True,
            name="bridge-stderr",
        ).start()

        # 启动 stdout 读取线程
        self._reader_thread = threading.Thread(
            target=self._stdout_reader,
            daemon=True,
            name="bridge-stdout",
        )
        self._reader_thread.start()

        # 等待就绪信号
        if not self._ready.wait(timeout=10):
            print("[audio-bridge] 启动超时")
            self.stop()
            return False

        print("[audio-bridge] 全双工音频引擎就绪（VP AEC）")
        return True

    def stop(self):
        """停止音频桥接"""
        self._running = False
        if self._proc and self._proc.poll() is None:
            try:
                self._send_message(MSG_EXIT)
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None

    # 播放由 Python sd.play 处理（VP 无播放节点时不能通过 bridge 播放）

    # ── 内部方法 ──

    def _send_message(self, msg_type: int, payload: bytes = b"") -> None:
        """发送消息给 Swift"""
        if not self._proc or not self._proc.stdin:
            return
        header = struct.pack("<BI", msg_type, len(payload))
        try:
            self._proc.stdin.write(header)
            if payload:
                self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _stdout_reader(self) -> None:
        """读取 Swift 发来的消息"""
        stdout = self._proc.stdout
        while self._running and self._proc.poll() is None:
            try:
                # 读 header
                header = self._read_exact(stdout, 5)
                if header is None:
                    break

                msg_type = header[0]
                length = struct.unpack("<I", header[1:5])[0]

                # 读 payload
                payload = b""
                if length > 0:
                    payload = self._read_exact(stdout, length)
                    if payload is None:
                        break

                self._handle_message(msg_type, payload)

            except Exception as e:
                if self._running:
                    print(f"[audio-bridge] 读取错误: {e}")
                break

    def _read_exact(self, stream, count: int) -> Optional[bytes]:
        """精确读取 N 字节"""
        data = b""
        while len(data) < count:
            chunk = stream.read(count - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _handle_message(self, msg_type: int, payload: bytes) -> None:
        """处理 Swift 发来的消息"""
        if msg_type == MSG_MIC:
            # 麦克风音频（16kHz mono float32, AEC 处理后）
            if self._on_mic_chunk and payload:
                pcm = np.frombuffer(payload, dtype=np.float32)
                self._on_mic_chunk(pcm)

        elif msg_type == MSG_PLAY_DONE:
            self._play_done_event.set()
            if self._on_play_done:
                self._on_play_done()

        elif msg_type == MSG_READY:
            self._ready.set()

        elif msg_type == MSG_ERROR:
            print(f"[audio-bridge] Swift 错误: {payload.decode('utf-8', errors='ignore')}")

    def _stderr_reader(self) -> None:
        """读取 Swift 的日志"""
        stderr = self._proc.stderr
        while self._running and self._proc.poll() is None:
            try:
                line = stderr.readline()
                if not line:
                    break
                print(f"[bridge] {line.decode('utf-8', errors='ignore').strip()}")
            except Exception:
                break
