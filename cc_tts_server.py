"""
cc_tts_server.py — TTS 独立服务进程

独立进程加载 Qwen3-TTS 1.7B Base 模型，通过 Unix Domain Socket 接收文本，返回 PCM。
与主进程的视觉/STT 模型各占独立 Metal GPU 上下文，互不阻塞。

协议：长度前缀帧 + msgpack
  请求: {"action": "synthesize"|"health"|"shutdown", "text": "..."}
  响应: {"ok": bool, "pcm": bytes, "sample_rate": int, "shape": [N]}

用法：
    python cc_tts_server.py          # 前台启动
    python cc_tts_server.py --daemon # 后台启动（日志到 /tmp/cc-tts.log）
"""

import os
import sys
import signal
import socket
import struct
import time
import numpy as np
import msgpack
from pathlib import Path

SOCK_PATH = "/tmp/cc-tts.sock"
MAX_MSG_SIZE = 10 * 1024 * 1024  # 10MB，足够一句话的 PCM

# ── 模型和缓存（从 cc_tts_local 复用逻辑）──

_tts_model = None
_audio_cache: dict = {}

# 导入配置
sys.path.insert(0, str(Path(__file__).parent))
from cc_voice_profile import BASE_MODEL, REF_AUDIO_PATH, REF_TEXT


def _get_model():
    """懒加载 TTS 模型"""
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    import warnings
    warnings.filterwarnings("ignore")
    from mlx_audio.tts.utils import load_model

    _tts_model = load_model(BASE_MODEL)
    print("[tts-server] Qwen3-TTS 1.7B Base 就绪")
    return _tts_model


def _synthesize(text: str) -> tuple:
    """合成一句话，返回 (pcm_ndarray, sample_rate)"""
    # 先查缓存
    if text in _audio_cache:
        return _audio_cache[text]

    model = _get_model()

    if not REF_AUDIO_PATH.exists():
        results = list(model.generate(text=text, lang_code="auto"))
    else:
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

    # 50ms 静音 + 20ms 淡入
    silence = np.zeros(int(r.sample_rate * 0.05), dtype=np.float32)
    fade_len = min(int(r.sample_rate * 0.02), len(samples))
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len, dtype=np.float32)
        samples[:fade_len] *= fade
    samples = np.concatenate([silence, samples])

    return samples, r.sample_rate


# ── 缓存管理 ──

_CACHE_DIR = Path(__file__).parent / ".venv" / "cache"
_CACHE_FILE = _CACHE_DIR / "tts_cache_base.npz"

_PRECACHE_PHRASES = [
    # 基础应答
    "好的呢。", "收到了。", "嗯，你说。", "明白了。", "没问题。",
    # 过渡语（云端衔接）
    "让我想想。", "稍等一下。", "我看看。", "好，稍等。",
    "我看看最新的。", "让我查一下。", "稍等，我确认下。",
    "我想想怎么解决。", "让我确认一下。", "我分析一下。",
    # 安静模式
    "好的，我听着。", "好，我在。",
    # 问候
    "早上好。", "下午好。", "晚上好。",
    # 贾维斯风格
    "所有系统运行正常。", "一切就绪，随时待命。",
    "放心，交给我。", "马上处理。",
    "建议你休息一下。",
    # 音乐操作（工具调用秒回）
    "已停止播放。", "继续播放。", "下一首。", "上一首。",
    "音量调大了。", "音量调小了。",
    "当前没有在播放。", "马上安排。",
]


def _load_cache():
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
        print(f"[tts-server] 磁盘缓存: {count} 条")
        return count
    except Exception as e:
        print(f"[tts-server] 缓存加载失败: {e}")
        return 0


def _save_cache():
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
        print(f"[tts-server] 缓存固化: {len(_audio_cache)} 条")
    except Exception as e:
        print(f"[tts-server] 缓存固化失败: {e}")


def _precache():
    """合成缺失的预缓存短句"""
    missing = [p for p in _PRECACHE_PHRASES if p not in _audio_cache]
    if not missing:
        print(f"[tts-server] 缓存完整: {len(_audio_cache)} 条")
        return

    print(f"[tts-server] 合成缺失: {len(missing)} 条...")
    for phrase in missing:
        try:
            pcm, sr = _synthesize(phrase)
            _audio_cache[phrase] = (pcm, sr)
        except Exception:
            pass

    _save_cache()
    print(f"[tts-server] 预缓存完成: {len(_audio_cache)} 条")


# ── UDS 服务 ──

_running = True


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """精确读取 n 字节"""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf += chunk
    return buf


def _handle_request(data: dict) -> dict:
    """处理一个请求"""
    action = data.get("action", "synthesize")

    if action == "health":
        return {"ok": True, "cached": len(_audio_cache), "model": _tts_model is not None}

    if action == "shutdown":
        global _running
        _running = False
        return {"ok": True, "message": "shutting down"}

    if action == "synthesize":
        text = data.get("text", "")
        if not text:
            return {"ok": False, "error": "empty text"}

        # 缓存命中 → 直接返回完整 PCM
        if text in _audio_cache:
            pcm, sr = _audio_cache[text]
            print(f"[tts-server] cache | {text[:40]}")
            return {
                "ok": True,
                "pcm": pcm.tobytes(),
                "sample_rate": sr,
                "shape": list(pcm.shape),
                "stream": False,
            }

        # 缓存未命中 → 标记为流式，由 _handle_client 处理
        return {"_stream_text": text}

    if action == "synthesize_stream":
        # 显式流式请求（兼容）
        text = data.get("text", "")
        if not text:
            return {"ok": False, "error": "empty text"}
        return {"_stream_text": text}

    return {"ok": False, "error": f"unknown action: {action}"}


def _stream_to_client(conn: socket.socket, text: str):
    """流式合成并逐 chunk 发送给客户端"""
    model = _get_model()
    t0 = time.time()
    all_chunks = []

    if not REF_AUDIO_PATH.exists():
        gen = model.generate(text=text, lang_code="auto",
                             stream=True, streaming_interval=0.3)
    else:
        gen = model.generate(text=text, ref_audio=str(REF_AUDIO_PATH),
                             ref_text=REF_TEXT, lang_code="auto",
                             stream=True, streaming_interval=0.3)

    for result in gen:
        samples = np.array(result.audio, dtype=np.float32)
        if len(samples) == 0:
            continue
        # 不做独立归一化（避免 chunk 间音量跳变），最后统一处理
        all_chunks.append(samples)

        # 发送 chunk（stream=True 标记）
        chunk_resp = msgpack.packb({
            "ok": True,
            "stream": True,
            "pcm": samples.tobytes(),
            "sample_rate": result.sample_rate,
            "shape": list(samples.shape),
            "done": False,
        }, use_bin_type=True)
        conn.sendall(struct.pack(">I", len(chunk_resp)) + chunk_resp)

    # 发送结束标记
    done_resp = msgpack.packb({"ok": True, "stream": True, "done": True}, use_bin_type=True)
    conn.sendall(struct.pack(">I", len(done_resp)) + done_resp)

    elapsed_ms = (time.time() - t0) * 1000
    print(f"[tts-server] {elapsed_ms:.0f}ms stream | {text[:40]}")

    # 拼接完整音频加入缓存
    if all_chunks:
        full_pcm = np.concatenate(all_chunks)
        # 加 50ms 静音 + 20ms 淡入
        sr = 24000
        silence = np.zeros(int(sr * 0.05), dtype=np.float32)
        fade_len = min(int(sr * 0.02), len(full_pcm))
        if fade_len > 0:
            fade = np.linspace(0, 1, fade_len, dtype=np.float32)
            full_pcm[:fade_len] *= fade
        full_pcm = np.concatenate([silence, full_pcm])
        _audio_cache[text] = (full_pcm, sr)


def _handle_client(conn: socket.socket):
    """处理一个客户端连接（短连接，一次请求）"""
    try:
        # 读长度前缀
        try:
            header = _recv_exact(conn, 4)
        except ConnectionError:
            return
        length = struct.unpack(">I", header)[0]
        if length > MAX_MSG_SIZE:
            return

        body = _recv_exact(conn, length)
        req = msgpack.unpackb(body, raw=False)

        # 处理请求
        resp = _handle_request(req)

        # 如果需要流式合成
        if "_stream_text" in resp:
            _stream_to_client(conn, resp["_stream_text"])
            return

        # 普通响应
        resp_bytes = msgpack.packb(resp, use_bin_type=True)
        conn.sendall(struct.pack(">I", len(resp_bytes)) + resp_bytes)

    except BrokenPipeError:
        pass
    except ConnectionError:
        pass
    except Exception as e:
        print(f"[tts-server] 客户端错误: {e}")
    finally:
        conn.close()


def serve():
    """启动 UDS 服务"""
    global _running

    # 清理旧 socket 文件
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

    # 初始化模型和缓存
    print("[tts-server] 启动中...")
    _get_model()
    _load_cache()
    _precache()

    # 创建 UDS
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    server.listen(2)
    server.settimeout(1.0)  # 每秒检查一次 _running

    print(f"[tts-server] 就绪，监听 {SOCK_PATH}")

    def _signal_handler(sig, frame):
        global _running
        print(f"\n[tts-server] 收到信号 {sig}，关闭中...")
        _running = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    while _running:
        try:
            conn, _ = server.accept()
            _handle_client(conn)
        except socket.timeout:
            continue
        except Exception as e:
            if _running:
                print(f"[tts-server] accept 错误: {e}")

    # 清理
    server.close()
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    print("[tts-server] 已关闭")


if __name__ == "__main__":
    serve()
