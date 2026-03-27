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

# ── API 配置 ──
# FUTURUS 大脑（兼容 OpenAI 协议）
_ENV_FILE = Path(__file__).parent / ".env"
_minimax_api_key: Optional[str] = None

MINIMAX_API_URL = "https://api.minimaxi.com/v1/chat/completions"
MINIMAX_MODEL = "MiniMax-M2.7"

# 本地快速路径
OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
OLLAMA_CHAT_MODEL = "qwen3:4b"

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

    # 语音交互专用指令
    system += (
        "\n\n[语音交互规则]"
        "\n- 你正在通过扬声器语音回复，回答要简洁口语化，像朋友聊天"
        "\n- 不要用 markdown 格式、代码块或列表"
        "\n- 不要说'作为AI'之类的话，你就是 cc 贾维斯"
        "\n- 1-3 句话回答，除非用户明确要求详细说明"
        "\n- 结合你看到的场景自然地回应，比如'我看到你在看手机'"
        "\n- 如果事件时间线里有相关信息，自然地融入回答"
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


def think_ollama(user_text: str) -> Optional[str]:
    """降级到本地 ollama 模型"""
    system_prompt = _build_context()

    messages = [{"role": "system", "content": system_prompt}]
    for msg in _history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        start = time.time()
        resp = requests.post(
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
    流式主入口：用户说了一句话，cc 按句子 yield 回复。

    路由：简单查询 → 本地 ollama 秒回 | 复杂对话 → MiniMax 流式
    """
    post_event("speech", f"川哥说：{user_text}", source="interact")

    yielded = False

    if _is_simple_query(user_text):
        # 快速路径：本地 ollama
        reply = think_ollama(user_text)
        if reply:
            for s in _split_sentences(reply):
                yielded = True
                yield s

    if not yielded:
        # 流式路径：MiniMax SSE
        for sentence in _stream_minimax(user_text):
            yielded = True
            yield sentence

    if not yielded:
        # 降级：本地 ollama 非流式
        reply = think_ollama(user_text)
        if reply:
            for s in _split_sentences(reply):
                yield s
        else:
            yield f"收到：{user_text}。网络和本地模型暂时都不通，稍后再试。"

    post_event("response", f"cc流式回复完成", source="brain")
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
