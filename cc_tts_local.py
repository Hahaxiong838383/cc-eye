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
_sock: Optional[socket.socket] = None


def _get_sock() -> socket.socket:
    """获取/重连 UDS 连接"""
    global _sock
    if _sock is not None:
        try:
            # 检测连接是否还活着
            _sock.getpeername()
            return _sock
        except (OSError, socket.error):
            _sock = None

    _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _sock.settimeout(15.0)  # 合成超时 15s
    _sock.connect(SOCK_PATH)
    return _sock


def _close_sock():
    """关闭 UDS 连接"""
    global _sock
    if _sock is not None:
        try:
            _sock.close()
        except Exception:
            pass
        _sock = None


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
    """发送请求，接收响应（长度前缀帧 + msgpack）"""
    sock = _get_sock()
    payload = msgpack.packb(req, use_bin_type=True)
    sock.sendall(struct.pack(">I", len(payload)) + payload)

    header = _recv_exact(sock, 4)
    length = struct.unpack(">I", header)[0]
    body = _recv_exact(sock, length)
    return msgpack.unpackb(body, raw=False)


def _remote_synthesize(text: str) -> Tuple[np.ndarray, int]:
    """通过 UDS 远程合成"""
    resp = _send_recv({"action": "synthesize", "text": text})
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "TTS server error"))
    pcm = np.frombuffer(resp["pcm"], dtype=np.float32).copy()
    sr = resp["sample_rate"]
    return pcm, sr


# ── 本地降级（TTS 服务不可用时）──

_fallback_model = None


def _local_fallback(text: str) -> Tuple[np.ndarray, int]:
    """降级：本地直接加载模型推理（有锁竞争但至少能工作）"""
    global _fallback_model
    from cc_voice_profile import BASE_MODEL, REF_AUDIO_PATH, REF_TEXT

    if _fallback_model is None:
        import warnings
        warnings.filterwarnings("ignore")
        from mlx_audio.tts.utils import load_model
        _fallback_model = load_model(BASE_MODEL)
        print("[cc-tts] 降级模式：本地加载 TTS 模型")

    if REF_AUDIO_PATH.exists():
        results = list(_fallback_model.generate(
            text=text, ref_audio=str(REF_AUDIO_PATH),
            ref_text=REF_TEXT, lang_code="auto",
        ))
    else:
        results = list(_fallback_model.generate(text=text, lang_code="auto"))

    r = results[0]
    samples = np.array(r.audio, dtype=np.float32)
    peak = np.abs(samples).max()
    if peak > 0.001:
        samples = samples * (0.95 / peak)
    silence = np.zeros(int(r.sample_rate * 0.05), dtype=np.float32)
    fade_len = min(int(r.sample_rate * 0.02), len(samples))
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len, dtype=np.float32)
        samples[:fade_len] *= fade
    samples = np.concatenate([silence, samples])
    return samples, r.sample_rate


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
