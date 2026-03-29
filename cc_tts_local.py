"""
cc_tts_local.py — TTS 客户端（混合模式：本地缓存 + UDS 远程合成）

优先查本地内存缓存（<1ms），缓存 miss 走 UDS 发到 TTS 服务进程。
TTS 服务进程不可用时降级到本地直接推理。

用法：
    from cc_tts_local import local_tts_to_pcm, preload
    pcm, sr = local_tts_to_pcm("你好")
"""

import socket
import struct
import numpy as np
import msgpack
from pathlib import Path
from typing import Optional, Tuple

# ── UDS 客户端 ──

SOCK_PATH = "/tmp/cc-tts.sock"


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """精确读取 n 字节"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("TTS server closed")
        buf += chunk
    return buf


def _send_recv(req: dict) -> dict:
    """每次新建连接，避免 barge-in 后数据流错位"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(15.0)
    try:
        sock.connect(SOCK_PATH)
        payload = msgpack.packb(req, use_bin_type=True)
        sock.sendall(struct.pack(">I", len(payload)) + payload)

        header = _recv_exact(sock, 4)
        length = struct.unpack(">I", header)[0]
        body = _recv_exact(sock, length)
        return msgpack.unpackb(body, raw=False)
    finally:
        sock.close()


def _remote_synthesize(text: str) -> Tuple[np.ndarray, int]:
    """通过 UDS 远程合成（支持流式接收，拼接后返回）"""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(15.0)
    try:
        sock.connect(SOCK_PATH)
        payload = msgpack.packb({"action": "synthesize", "text": text}, use_bin_type=True)
        sock.sendall(struct.pack(">I", len(payload)) + payload)

        # 读第一个响应
        header = _recv_exact(sock, 4)
        length = struct.unpack(">I", header)[0]
        body = _recv_exact(sock, length)
        resp = msgpack.unpackb(body, raw=False)

        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "TTS server error"))

        # 非流式（缓存命中）→ 直接返回
        if not resp.get("stream"):
            pcm = np.frombuffer(resp["pcm"], dtype=np.float32).copy()
            return pcm, resp["sample_rate"]

        # 流式 → 收集所有 chunk 拼接
        chunks = []
        sr = resp.get("sample_rate", 24000)
        if "pcm" in resp:
            chunks.append(np.frombuffer(resp["pcm"], dtype=np.float32).copy())
            sr = resp["sample_rate"]

        while True:
            header = _recv_exact(sock, 4)
            length = struct.unpack(">I", header)[0]
            body = _recv_exact(sock, length)
            chunk_resp = msgpack.unpackb(body, raw=False)
            if chunk_resp.get("done"):
                break
            if "pcm" in chunk_resp:
                chunks.append(np.frombuffer(chunk_resp["pcm"], dtype=np.float32).copy())
                sr = chunk_resp["sample_rate"]

        if not chunks:
            raise RuntimeError("No audio chunks received")
        return np.concatenate(chunks), sr
    finally:
        sock.close()


# ── 本地降级（TTS 服务不可用时）──
# 注意：不能在主进程加载 MLX TTS 模型，会和视觉模型冲突导致 segfault。
# 降级策略：返回静音，等 TTS 服务恢复。


def _local_fallback(text: str) -> Tuple[np.ndarray, int]:
    """降级：返回静音（不在主进程加载 MLX 模型，避免和视觉冲突 crash）"""
    print(f"[cc-tts] TTS 服务不可用，跳过: {text[:30]}")
    sr = 24000
    silence = np.zeros(int(sr * 0.1), dtype=np.float32)
    return silence, sr


# ── 本地缓存 ──

_audio_cache: dict = {}
_CACHE_DIR = Path(__file__).parent / ".venv" / "cache"
_CACHE_FILE = _CACHE_DIR / "tts_cache_base.npz"


def _load_cache_from_disk() -> int:
    """从磁盘加载缓存"""
    if not _CACHE_FILE.exists():
        return 0
    try:
        data = np.load(str(_CACHE_FILE), allow_pickle=True)
        meta = data["meta"].item()
        count = 0
        for phrase, sr in meta.items():
            key = f"audio_{count}"
            if key in data:
                _audio_cache[phrase] = (data[key], int(sr))
                count += 1
        print(f"[cc-tts] 磁盘缓存: {count} 条")
        return count
    except Exception as e:
        print(f"[cc-tts] 缓存加载失败: {e}")
        return 0


# ── 公开接口 ──

def local_tts_to_pcm(
    text: str,
    speaker: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    合成语音（混合模式）：
    1. 本地缓存命中 → 直接返回（<1ms）
    2. 缓存 miss → UDS 远程合成
    3. UDS 不可用 → 降级本地直接推理
    """
    # L1: 本地内存缓存
    if text in _audio_cache:
        return _audio_cache[text]

    # L2: UDS 远程合成
    try:
        pcm, sr = _remote_synthesize(text)
        # 缓存到本地（下次不再走网络）
        _audio_cache[text] = (pcm, sr)
        return pcm, sr
    except (ConnectionError, FileNotFoundError, OSError) as e:
        _close_sock()  # 重置连接
        print(f"[cc-tts] UDS 不可用 ({e})，降级本地推理")

    # L3: 本地降级
    pcm, sr = _local_fallback(text)
    _audio_cache[text] = (pcm, sr)
    return pcm, sr


def local_tts_stream(text: str, speaker: Optional[str] = None):
    """流式合成（降级到非流式，因为 UDS 不支持流式）"""
    pcm, sr = local_tts_to_pcm(text, speaker)
    yield pcm, sr


def preload():
    """加载本地缓存 + 确认 TTS 服务就绪"""
    _load_cache_from_disk()

    # 健康检查
    try:
        resp = _send_recv({"action": "health"})
        if resp.get("ok"):
            print(f"[cc-tts] TTS 服务就绪，缓存 {resp.get('cached', '?')} 条")
            return
    except Exception:
        pass
    print("[cc-tts] 警告：TTS 服务未就绪，将使用降级模式")
