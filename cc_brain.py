"""
cc_brain.py — cc 贾维斯的大脑（云端 LLM 对话）

语音交互的核心理解层。收到用户语音文字后，调用云端 LLM 生成回复。
支持 Gemini Flash（主力）+ 本地 ollama（降级）。

用法：
    from cc_brain import think
    reply = think("今天天气怎么样？")
"""

import json
import time
import requests
from typing import Optional
from pathlib import Path

from cc_context import build_system_prompt, get_scene_context
from cc_events import get_context_window, post_event

# ── API 配置 ──
# FUTURUS 大脑（兼容 OpenAI 协议）
_ENV_FILE = Path(__file__).parent / ".env"
_minimax_api_key: Optional[str] = None

MINIMAX_API_URL = "https://api.minimaxi.com/v1/chat/completions"
MINIMAX_MODEL = "MiniMax-M2.7"

# 本地降级
OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
OLLAMA_CHAT_MODEL = "qwen2.5:3b"

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


def think(user_text: str) -> str:
    """
    主入口：用户说了一句话，cc 思考后回复。

    优先级：Gemini Flash → 本地 ollama → 兜底回复
    """
    # 记录用户语音到事件流
    post_event("speech", f"川哥说：{user_text}", source="interact")

    # 1. 尝试 FUTURUS 大脑
    reply = think_minimax(user_text)
    if not reply:
        # 2. 降级到本地 ollama
        reply = think_ollama(user_text)
    if not reply:
        # 3. 兜底
        reply = f"收到：{user_text}。网络和本地模型都不太通畅，稍后再试。"

    # 记录回复到事件流
    post_event("response", f"cc回复：{reply[:100]}", source="brain")

    # 检查是否需要压缩对话历史
    _maybe_summarize()

    return reply


if __name__ == "__main__":
    print("=== cc 贾维斯大脑测试 ===")
    print("测试 Gemini Flash...")
    reply = think("你好，贾维斯，现在几点了？")
    print(f"\n回复: {reply}")
