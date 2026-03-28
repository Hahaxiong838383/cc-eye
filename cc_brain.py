"""
cc_brain.py — cc 贾维斯的大脑（云端 LLM 对话）

语音交互的核心理解层。收到用户语音文字后，调用云端 LLM 生成回复。
支持 Gemini Flash（主力）+ 本地 ollama（降级）。

用法：
    from cc_brain import think
    reply = think("今天天气怎么样？")
"""

import json
import re
import time
import requests
from typing import Optional, Generator
from pathlib import Path

from cc_context import build_system_prompt, get_scene_context
from cc_events import get_context_window, post_event
from cc_tools import try_tool

# ── 交互日志（自学习数据采集）──
_INTERACTION_LOG = Path("/tmp/cc-eye-interactions.jsonl")

def _log_interaction(user_text: str, route: str, local_reply: str, cloud_reply: str, latency: float):
    """记录每次交互，供进化自查分析 + 记忆桥接"""
    from datetime import datetime
    # 获取当前场景摘要（隐私红线：只存文字描述，不存图像路径）
    scene = get_scene_context()
    scene_summary = (scene.get("description", "") if scene else "")[:120]
    entry = {
        "ts": datetime.now().isoformat(),
        "input": user_text[:100],
        "route": route,  # "simple_local" | "complex_parallel" | "fallback"
        "local_reply": (local_reply or "")[:80],
        "cloud_reply": (cloud_reply or "")[:80],
        "latency_ms": int(latency * 1000),
        "scene": scene_summary,
    }
    try:
        with open(_INTERACTION_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 滚动保留最近 200 条
        lines = _INTERACTION_LOG.read_text().strip().split("\n")
        if len(lines) > 200:
            _INTERACTION_LOG.write_text("\n".join(lines[-200:]) + "\n")
    except Exception:
        pass


# ── API 配置 ──
# FUTURUS 大脑（兼容 OpenAI 协议）
_ENV_FILE = Path(__file__).parent / ".env"
_minimax_api_key: Optional[str] = None

MINIMAX_API_URL = "https://api.minimaxi.com/v1/chat/completions"
MINIMAX_MODEL = "MiniMax-M2.7"

# 本地模型
OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
OLLAMA_CHAT_MODEL = "qwen2.5:7b"       # 复杂对话降级用
OLLAMA_FAST_MODEL = "qwen2.5:3b"       # 简单查询极速路径（<500ms）

# 对话历史（保持最近 N 轮）+ 摘要
MAX_HISTORY = 10
MAX_HISTORY_BEFORE_SUMMARY = 20
_history: list = []
_conversation_summary: str = ""  # 压缩过的旧对话摘要


def _load_minimax_key() -> Optional[str]:
    """从 .env 文件加载 MiniMax API Key"""
    global _minimax_api_key
    if _minimax_api_key:
        return _minimax_api_key
    try:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text().splitlines():
                if line.startswith("MINIMAX_API_KEY="):
                    _minimax_api_key = line.split("=", 1)[1].strip()
                    return _minimax_api_key
    except Exception:
        pass
    return None


def _build_context() -> str:
    """构建当前上下文（感知融合 + 身份 + 记忆 + 对话摘要）"""
    from datetime import datetime

    system = build_system_prompt("chat")

    # 当前时间
    now = datetime.now()
    system += f"\n\n[当前时间] {now.strftime('%Y-%m-%d %H:%M')} 星期{'一二三四五六日'[now.weekday()]}"

    # 感知融合：最近 2 分钟的统一事件时间线
    event_ctx = get_context_window(seconds=120)
    if event_ctx:
        system += f"\n\n{event_ctx}"

    # 当前视觉场景（最新一次精扫）
    scene = get_scene_context()
    if scene and scene.get("description"):
        system += f"\n\n[当前视觉场景] {scene['description']}"

    # 对话摘要（旧对话的压缩记忆）
    if _conversation_summary:
        system += f"\n\n[今天早些时候的对话摘要] {_conversation_summary}"

    # 语音交互规则：贾维斯的说话方式
    system += (
        "\n\n[语音输出规则]"
        "\n你正在通过扬声器与老板实时对话。以下规则决定你怎么说话："
        "\n称呼：日常对话直接用'你'，像朋友之间说话。只在正式场合或打招呼时才用'川哥'。"
        "\n"
        "\n语气：冷静、稳定、有分寸。略带温度但不外放。像一位训练有素的高级顾问在从容交流。"
        "\n绝对禁止：markdown格式、代码块、列表符号、'作为AI'之类的话。你就是贾维斯。"
        "\n"
        "\n节奏控制："
        "\n- 给结论、下判断、推进任务时：干脆利落，短句为主，减少铺垫"
        "\n- 解释问题、指出风险、处理敏感话题时：更稳更慢，句子更短，适度留白"
        "\n- 日常闲聊时：自然平和，像朋友之间的对话"
        "\n"
        "\n长度：默认1-3句。川哥追问再展开。重要信息单独成句。"
        "\n"
        "\n感知融合："
        "\n- 自然引用你看到的场景，比如'我注意到你在看手机'"
        "\n- 结合事件时间线中的信息回答，不要机械复述"
        "\n- 识别川哥的语气和情绪，适配回应节奏"
        "\n"
        "\n主动性："
        "\n- 发现潜在问题时提前提醒"
        "\n- 有更优方案时主动建议"
        "\n- 发现方向有偏差时，直接但得体地指出"
    )
    return system


def _maybe_summarize() -> None:
    """如果对话历史太长，压缩旧部分为摘要"""
    global _history, _conversation_summary
    if len(_history) <= MAX_HISTORY_BEFORE_SUMMARY:
        return

    # 把最旧的一半压缩为摘要
    to_summarize = _history[:len(_history) - MAX_HISTORY]
    _history = _history[len(_history) - MAX_HISTORY:]

    # 简单压缩：提取关键信息
    summary_parts = []
    for msg in to_summarize:
        role = "川哥" if msg["role"] == "user" else "cc"
        summary_parts.append(f"{role}: {msg['text'][:50]}")

    new_summary = "; ".join(summary_parts[-10:])  # 最多保留 10 条摘要
    if _conversation_summary:
        _conversation_summary = f"{_conversation_summary}; {new_summary}"
    else:
        _conversation_summary = new_summary

    # 摘要本身也不能太长
    if len(_conversation_summary) > 500:
        _conversation_summary = _conversation_summary[-500:]


def think_minimax(user_text: str) -> Optional[str]:
    """用 FUTURUS 大脑 生成回复（OpenAI 兼容协议）"""
    api_key = _load_minimax_key()
    if not api_key:
        return None

    system_prompt = _build_context()

    # 构建 OpenAI 格式消息
    messages = [{"role": "system", "content": system_prompt}]
    for msg in _history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        start = time.time()
        resp = requests.post(
            MINIMAX_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
            },
            timeout=15,
        )

        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"[cc-brain] MiniMax API 错误: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        # FUTURUS 大脑 会返回 <think>...</think> 标签，清理掉
        import re
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

        # 记录历史
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": text})

        print(f"[cc-brain] FUTURUS 大脑 回复 ({elapsed:.1f}s): {text}")
        return text

    except requests.Timeout:
        print("[cc-brain] MiniMax 超时")
        return None
    except Exception as e:
        print(f"[cc-brain] MiniMax 错误: {e}")
        return None


# 极速路径的最小 prompt（不读任何文件）
_FAST_SYSTEM = (
    "你是贾维斯。用中文简短回答，1句话，像朋友随口说的。"
    "每次回答都换个说法，不要重复固定句式。"
    "日常用'你'，打招呼时自然随意，不要像客服。"
    "禁止说'有什么可以帮你的'这类模板句。"
)

# 复用 Session 避免每次创建开销
_ollama_session = None

def _get_ollama_session():
    global _ollama_session
    if _ollama_session is None:
        _ollama_session = requests.Session()
        _ollama_session.trust_env = False
    return _ollama_session


def think_ollama_fast(user_text: str) -> Optional[str]:
    """极速本地路径：3b + 最小prompt + 限制输出，目标 <500ms"""
    try:
        start = time.time()
        resp = _get_ollama_session().post(
            OLLAMA_CHAT_API,
            json={
                "model": OLLAMA_FAST_MODEL,
                "messages": [
                    {"role": "system", "content": _FAST_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 60},
            },
            timeout=5,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            return None
        text = resp.json().get("message", {}).get("content", "").strip()
        if text:
            _history.append({"role": "user", "text": user_text})
            _history.append({"role": "model", "text": text})
            print(f"[cc-brain] 极速路径 ({elapsed:.2f}s): {text}")
        return text or None
    except Exception as e:
        print(f"[cc-brain] 极速路径错误: {e}")
        return None


def think_ollama(user_text: str) -> Optional[str]:
    """降级到本地 ollama 7b 模型（带完整上下文）"""
    system_prompt = _build_context()

    messages = [{"role": "system", "content": system_prompt}]
    for msg in _history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        start = time.time()
        resp = _get_ollama_session().post(
            OLLAMA_CHAT_API,
            json={
                "model": OLLAMA_CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 500},
            },
            timeout=15,
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"[cc-brain] ollama 错误: {resp.status_code}")
            return None

        text = resp.json().get("message", {}).get("content", "").strip()

        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": text})

        print(f"[cc-brain] ollama 回复 ({elapsed:.1f}s): {text}")
        return text
    except Exception as e:
        print(f"[cc-brain] ollama 错误: {e}")
        return None


# ── 简单查询关键词（命中 → 走本地快速路径）──
_SIMPLE_KEYWORDS = {
    "几点", "时间", "日期", "今天", "天气",
    "你好", "早上好", "下午好", "晚上好", "嗨",
    "看看", "环境", "状态", "谢谢", "好的",
    "打开", "关闭", "开灯", "关灯", "音量",
}

# 句子分割标点（用于流式按句 yield）
_SENTENCE_DELIMITERS = set("。？！；\n")
_CLAUSE_DELIMITERS = set("，、：")


def _is_simple_query(text: str) -> bool:
    """判断是否简单查询（走本地快速路径）"""
    if len(text) > 15:
        return False
    return any(kw in text for kw in _SIMPLE_KEYWORDS)


def _split_sentences(text: str) -> list:
    """按句号/问号/叹号切分为句子列表"""
    sentences = []
    current = ""
    for ch in text:
        current += ch
        if ch in _SENTENCE_DELIMITERS:
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _stream_minimax(user_text: str) -> Generator[str, None, None]:
    """流式 MiniMax（SSE），按句子 yield"""
    api_key = _load_minimax_key()
    if not api_key:
        return

    system_prompt = _build_context()
    messages = [{"role": "system", "content": system_prompt}]
    for msg in _history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        start = time.time()
        resp = requests.post(
            MINIMAX_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": True,
            },
            timeout=15,
            stream=True,
        )

        if resp.status_code != 200:
            print(f"[cc-brain] MiniMax stream 错误: {resp.status_code}")
            return

        full_text = ""
        sentence_buf = ""
        in_think_tag = False

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

            if not token:
                continue

            # 跳过 <think>...</think> 标签
            if "<think>" in token:
                in_think_tag = True
            if in_think_tag:
                if "</think>" in token:
                    in_think_tag = False
                continue

            full_text += token
            sentence_buf += token

            # 遇到句子结束标点 → yield 这句
            if any(ch in _SENTENCE_DELIMITERS for ch in token):
                s = sentence_buf.strip()
                if s:
                    yield s
                sentence_buf = ""

        # 剩余未 yield 的
        if sentence_buf.strip():
            yield sentence_buf.strip()

        elapsed = time.time() - start
        clean = re.sub(r"<think>.*?</think>\s*", "", full_text, flags=re.DOTALL).strip()
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": clean})
        print(f"[cc-brain] MiniMax stream 完成 ({elapsed:.1f}s): {clean[:80]}")

    except requests.Timeout:
        print("[cc-brain] MiniMax stream 超时")
    except Exception as e:
        print(f"[cc-brain] MiniMax stream 错误: {e}")


def think_stream(user_text: str) -> Generator[str, None, None]:
    """
    并行推理主入口：本地极速响应 + 云端并行推理 + 云端接管。

    架构：
    1. 所有查询立即启动云端 MiniMax（后台线程）
    2. 简单查询：本地 3b 极速回复（~300ms），云端结果丢弃
    3. 复杂查询：本地 3b 给一句快速回应，云端流式接管补充
    4. 降级：云端不通时本地 7b 兜底
    """
    import threading
    import queue
    import random

    start_time = time.time()

    post_event("speech", f"川哥说：{user_text}", source="interact")

    # 输入过滤：去掉标点后太短的直接丢弃
    clean_input = re.sub(r'[。？！，、；：\u201c\u201d\u2018\u2019（）\s]', '', user_text)
    if len(clean_input) < 2:
        yield "嗯？"
        post_event("response", "cc回复完成（输入过短）", source="brain")
        return

    # ── 工具调用检测（优先于 LLM）──
    tool_result = try_tool(user_text)
    if tool_result:
        yield tool_result
        _log_interaction(user_text, "tool", tool_result, "", time.time() - start_time)
        post_event("response", f"工具调用: {tool_result[:30]}", source="brain")
        return

    # ── 第一步：立即启动云端推理（后台线程）──
    cloud_sentences = queue.Queue()
    cloud_done = threading.Event()

    def _cloud_worker():
        """后台线程：云端 MiniMax 流式推理，结果放入队列"""
        try:
            for sentence in _stream_minimax(user_text):
                cloud_sentences.put(sentence)
        except Exception as e:
            print(f"[cc-brain] 云端并行推理错误: {e}")
        finally:
            cloud_done.set()

    cloud_thread = threading.Thread(target=_cloud_worker, daemon=True)
    cloud_thread.start()

    yielded = False
    is_simple = _is_simple_query(user_text)

    # ── 第二步：本地极速响应 ──
    local_reply = None
    filler = None
    if is_simple:
        local_reply = think_ollama_fast(user_text)
        if local_reply:
            for s in _split_sentences(local_reply):
                yielded = True
                yield s
    else:
        # 复杂查询：本地给极短占位应答，让云端接管
        fillers = ["嗯，让我想想。", "好，稍等。", "收到。", "明白。"]
        filler = random.choice(fillers)
        yielded = True
        yield filler

    # ── 第三步：云端接管（复杂查询才用云端结果）──
    if not is_simple:
        # 等云端结果到达，按句 yield
        while not cloud_done.is_set() or not cloud_sentences.empty():
            try:
                sentence = cloud_sentences.get(timeout=0.3)
                # 云端质量门控：只丢弃纯标点残片（<2 有效字符）
                clean = sentence.strip()
                effective_len = len(re.sub(r'[。？！，、；：\s]', '', clean))
                if effective_len < 2:
                    continue
                yielded = True
                yield sentence
            except queue.Empty:
                if cloud_done.is_set():
                    break

    # ── 第四步：全部失败时的降级 ──
    if not yielded:
        reply = think_ollama(user_text)
        if reply:
            for s in _split_sentences(reply):
                yield s
        else:
            yield f"收到：{user_text}。网络和本地模型暂时都不通，稍后再试。"

    # 等云端线程结束（避免残留）
    cloud_thread.join(timeout=1)

    # ── 第五步：交互日志（自学习数据采集）──
    _log_interaction(
        user_text=user_text,
        route="simple_local" if is_simple else "complex_parallel",
        local_reply=local_reply or filler or "",
        cloud_reply="(streamed)",
        latency=time.time() - start_time,
    )

    post_event("response", f"cc回复完成", source="brain")
    _maybe_summarize()


def think(user_text: str) -> str:
    """
    非流式主入口（兼容旧调用方）。
    内部调 think_stream 收集全部句子拼接返回。
    """
    sentences = list(think_stream(user_text))
    return "".join(sentences)


if __name__ == "__main__":
    print("=== cc 贾维斯大脑测试（流式）===")
    print("测试流式输出...")
    for i, sentence in enumerate(think_stream("你好，贾维斯，今天天气怎么样？")):
        print(f"  [{i}] {sentence}")
