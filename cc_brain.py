"""
cc_brain.py — cc 贾维斯的大脑（云端 LLM 对话）

语音交互的核心理解层。收到用户语音文字后，调用云端 LLM 生成回复。
支持 Gemini Flash（主力）+ 本地 ollama（降级）。

用法：
    from cc_brain import think
    reply = think("今天天气怎么样？")
"""

import json
import queue
import re
import threading
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
_ENV_FILE = Path(__file__).parent / ".env"
_minimax_api_key: Optional[str] = None
_gemini_api_key: Optional[str] = None

# MiniMax（深度思考）
MINIMAX_API_URL = "https://api.minimaxi.com/v1/chat/completions"
MINIMAX_DEEP = "MiniMax-M2.7-highspeed"
MINIMAX_MODEL = MINIMAX_DEEP

# Gemini（快速响应，718ms 首 token）
GEMINI_MODEL = "models/gemini-2.5-flash-lite"

# 云端快速 = Gemini，云端深度 = MiniMax M2.7
CLOUD_FAST = "gemini"
CLOUD_DEEP = "minimax"

# 本地模型
# 本地 LLM：oMLX Qwen3.5-9B（OpenAI 兼容接口）
LOCAL_LLM_API = "http://localhost:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen3.5-4B-MLX-4bit"

# Ollama 降级（oMLX 不可用时）
OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
OLLAMA_CHAT_MODEL = "qwen2.5:3b"
OLLAMA_FAST_MODEL = "qwen2.5:3b"

# 对话历史（保持最近 N 轮）+ 摘要
MAX_HISTORY = 10
MAX_HISTORY_BEFORE_SUMMARY = 20
_history: list = []
_conversation_summary: str = ""  # 压缩过的旧对话摘要


def _load_gemini_key() -> Optional[str]:
    """加载 Gemini API Key"""
    global _gemini_api_key
    if _gemini_api_key:
        return _gemini_api_key
    try:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    _gemini_api_key = line.split("=", 1)[1].strip()
                    return _gemini_api_key
    except Exception:
        pass
    return None


_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    key = _load_gemini_key()
    if not key:
        return None
    from google import genai
    _gemini_client = genai.Client(api_key=key)
    return _gemini_client


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
        "\n自适应长度："
        "\n- 简短问候、确认、指令 → 1句话回答"
        "\n- 日常聊天、简单问题 → 1-3句"
        "\n- 复杂讨论、分析建议 → 展开说，但分段，每段2-3句"
        "\n- 川哥在思考或自言自语 → 简短回应或不说话"
        "\n- 川哥语速快/语气急 → 精简直接"
        "\n- 川哥语速慢/在探索 → 可以展开"
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
        "\n"
        "\n你的能力："
        "\n- 联网搜索：实时搜索互联网获取最新新闻、天气、股价、技术文档等"
        "\n- 多模态理解：你能理解图片、文档、代码截图等视觉内容"
        "\n- 记忆系统：记住对话历史、川哥的偏好、项目进展、过往决策"
        "\n- 深度分析：数据分析、方案对比、风险评估、竞品研究"
        "\n- 视觉感知：通过摄像头看到川哥的物理环境、表情、动作"
        "\n- 任务执行：处理任务、设置提醒、查询信息、代码审查"
        "\n- 工具调用：根据需要自动调用搜索、计算等工具"
        "\n当川哥问实时信息（新闻/天气/股价/赛事等）时，直接搜索并回答，不要说'我没有联网'。"
        "\n当川哥发图片或截图时，直接分析内容，不要说'我看不到'。"
        "\n"
        "\n重要：不要输出思考过程，不要使用<think>标签，直接给出回答。速度优先。"
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
        resp = _get_minimax_session().post(
            MINIMAX_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "tools": [
                    {"type": "web_search", "web_search": {"enable": True}},
                ],
            },
            timeout=20,
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


# 极速路径 prompt（保留贾维斯人格，但精简上下文）
_FAST_SYSTEM = (
    "你是贾维斯，川哥的长期AI副手。人格底色：冷静、精准、克制、可靠。"
    "兼具英式管家的分寸感、顶级参谋的判断力、长期陪伴者的信任感。"
    "\n"
    "你正在通过扬声器与川哥实时语音对话。"
    "规则：用中文，1-3句话。称呼用'你'，日常像朋友。"
    "语气冷静稳重，略带温度，像高级顾问从容交流。"
    "先给结论再说原因。有更优方案主动提出。"
    "\n"
    "川哥问你状态/自检时，汇报系统状态（视觉、听觉、大脑各模块运行情况）。"
    "禁止：markdown、列表符号、代码块、'作为AI'、'有什么可以帮你的'。你就是贾维斯。"
)

# 复用 Session 避免每次创建开销（trust_env=False 绕过系统代理）
_ollama_session = None
_minimax_session = None

def _get_ollama_session():
    global _ollama_session
    if _ollama_session is None:
        _ollama_session = requests.Session()
        _ollama_session.trust_env = False
    return _ollama_session

def _get_minimax_session():
    """MiniMax 是国内服务，必须直连不走 xray 代理"""
    global _minimax_session
    if _minimax_session is None:
        _minimax_session = requests.Session()
        _minimax_session.trust_env = False
    return _minimax_session


def _stream_local(user_text: str, max_tokens: int = 150) -> Generator[str, None, None]:
    """本地 oMLX 流式输出，按句 yield（首 token ~250ms）"""
    try:
        system_prompt = _build_context()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in _history[-MAX_HISTORY:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["text"]})
        messages.append({"role": "user", "content": user_text})

        start = time.time()
        resp = _get_ollama_session().post(
            LOCAL_LLM_API,
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
            timeout=15,
        )

        if resp.status_code != 200:
            print(f"[cc-brain] oMLX stream 错误: {resp.status_code}")
            return

        full_text = ""
        sentence_buf = ""

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
                token = json.loads(data_str)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

            if not token:
                continue

            full_text += token
            for ch in token:
                sentence_buf += ch
                if ch in _SENTENCE_DELIMITERS:
                    s = sentence_buf.strip()
                    if s:
                        yield s
                    sentence_buf = ""

        if sentence_buf.strip():
            yield sentence_buf.strip()

        elapsed = time.time() - start
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": full_text})
        print(f"[cc-brain] oMLX stream ({elapsed:.1f}s): {full_text[:60]}")

    except Exception as e:
        print(f"[cc-brain] oMLX stream 错误: {e}")
        # 降级非流式
        reply = think_local(user_text)
        if reply:
            yield reply


def think_local(user_text: str) -> Optional[str]:
    """本地 LLM（oMLX Qwen3.5-9B，~1s）"""
    try:
        system_prompt = _build_context()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in _history[-MAX_HISTORY:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["text"]})
        messages.append({"role": "user", "content": user_text})

        start = time.time()
        resp = _get_ollama_session().post(
            LOCAL_LLM_API,
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=10,
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"[cc-brain] oMLX 错误: {resp.status_code}")
            return None

        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text:
            _history.append({"role": "user", "text": user_text})
            _history.append({"role": "model", "text": text})
            print(f"[cc-brain] oMLX ({elapsed:.2f}s): {text}")
        return text or None
    except Exception as e:
        print(f"[cc-brain] oMLX 错误: {e}")
        return think_ollama_fast(user_text)  # 降级到 Ollama


def think_ollama_fast(user_text: str) -> Optional[str]:
    """降级路径：Ollama 3b"""
    try:
        # 用和 MiniMax 相同的完整上下文（人格+记忆+对话历史统一）
        system_prompt = _build_context()

        messages = [{"role": "system", "content": system_prompt}]
        for msg in _history[-MAX_HISTORY:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["text"]})
        messages.append({"role": "user", "content": user_text})

        start = time.time()
        resp = _get_ollama_session().post(
            OLLAMA_CHAT_API,
            json={
                "model": OLLAMA_FAST_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 80},
            },
            timeout=10,
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
_SENTENCE_DELIMITERS = set("。！？")  # 自然呼吸点断句
_CLAUSE_DELIMITERS = set("，、：")


# 纯本地的关键词（只有这些走本地，其余全走云端）
_LOCAL_ONLY_KEYWORDS = {
    "你好", "早上好", "下午好", "晚上好", "嗨", "早",
    "谢谢", "好的", "知道了", "明白", "行", "嗯",
    "再见", "拜拜", "晚安",
}

def _query_tier(text: str) -> str:
    """
    判断查询级别：
    - 'local': 纯问候/确认（≤8字且命中关键词）
    - 'cloud': 其余全走云端（质量+实时信息）
    """
    if len(text) <= 8 and any(kw in text for kw in _LOCAL_ONLY_KEYWORDS):
        return "local"
    return "cloud"

LOCAL_SHORT_MAX_TOKENS = 30


def _stream_gemini(user_text: str) -> Generator[str, None, None]:
    """Gemini 2.5 Flash-Lite 流式（~700ms 首 token）"""
    client = _get_gemini_client()
    if not client:
        return

    system_prompt = _build_context()

    # 构建对话历史
    contents = []
    contents.append({"role": "user", "parts": [{"text": f"[系统指令]{system_prompt}"}]})
    contents.append({"role": "model", "parts": [{"text": "明白，我是贾维斯。"}]})
    for msg in _history[-MAX_HISTORY:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["text"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    try:
        start = time.time()
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
        )

        full_text = ""
        sentence_buf = ""

        for chunk in response:
            if not chunk.text:
                continue
            token = chunk.text
            full_text += token

            for ch in token:
                sentence_buf += ch
                if ch in _SENTENCE_DELIMITERS:
                    s = sentence_buf.strip()
                    if s:
                        yield s
                    sentence_buf = ""

        if sentence_buf.strip():
            yield sentence_buf.strip()

        elapsed = time.time() - start
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": full_text})
        print(f"[cc-brain] Gemini ({elapsed:.1f}s): {full_text[:60]}")

    except Exception as e:
        print(f"[cc-brain] Gemini 错误: {e}")


def _stream_minimax_model(user_text: str, model: str) -> Generator[str, None, None]:
    """用指定模型流式调用 MiniMax"""
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
        resp = _get_minimax_session().post(
            MINIMAX_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": True,
                "tools": [{"type": "web_search", "web_search": {"enable": True}}],
            },
            timeout=20,
            stream=True,
        )
        if resp.status_code != 200:
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
                token = json.loads(data_str)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
            if not token:
                continue
            if in_think_tag:
                if "</think>" in token:
                    in_think_tag = False
                    token = token.split("</think>", 1)[1].lstrip("\n\r ")
                    if not token:
                        continue
                else:
                    continue
            if "<think>" in token:
                before = token.split("<think>", 1)[0]
                in_think_tag = True
                if not before.strip():
                    continue
                token = before

            full_text += token
            for ch in token:
                sentence_buf += ch
                if ch in _SENTENCE_DELIMITERS:
                    s = sentence_buf.strip()
                    if s:
                        yield s
                    sentence_buf = ""

        if sentence_buf.strip():
            yield sentence_buf.strip()

        elapsed = time.time() - start
        clean = re.sub(r"<think>.*?</think>\s*", "", full_text, flags=re.DOTALL).strip()
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": clean})
        print(f"[cc-brain] {model} ({elapsed:.1f}s): {clean[:60]}")

    except Exception as e:
        print(f"[cc-brain] {model} 错误: {e}")


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
        resp = _get_minimax_session().post(
            MINIMAX_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": True,
                "tools": [
                    {"type": "web_search", "web_search": {"enable": True}},
                ],
            },
            timeout=20,
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

            # 跳过 <think>...</think> 标签，保留标签外的文本
            if in_think_tag:
                if "</think>" in token:
                    in_think_tag = False
                    # 提取 </think> 之后的正文
                    token = token.split("</think>", 1)[1].lstrip("\n\r ")
                    if not token:
                        continue
                else:
                    continue
            if "<think>" in token:
                # 提取 <think> 之前的正文（通常没有）
                before = token.split("<think>", 1)[0]
                in_think_tag = True
                if not before.strip():
                    continue
                token = before

            full_text += token

            # 逐字符扫描，只在自然呼吸点断（。！）
            for ch in token:
                sentence_buf += ch
                if ch in _SENTENCE_DELIMITERS:
                    s = sentence_buf.strip()
                    if s:
                        yield s
                    sentence_buf = ""

        # 剩余
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

    # 输入过滤：去掉标点后为空才丢弃
    clean_input = re.sub(r'[。？！，、；：\u201c\u201d\u2018\u2019（）\s]', '', user_text)
    if len(clean_input) == 0:
        return

    # ── 工具调用检测（优先于 LLM）──
    tool_result = try_tool(user_text)
    if tool_result:
        yield tool_result
        _log_interaction(user_text, "tool", tool_result, "", time.time() - start_time)
        post_event("response", f"工具调用: {tool_result[:30]}", source="brain")
        return

    # ── 三级路由 ──
    tier = _query_tier(user_text)
    yielded = False

    if tier == "local":
        # 短句：本地流式秒回（限制输出长度，1句话搞定）
        for sentence in _stream_local(user_text, max_tokens=LOCAL_SHORT_MAX_TOKENS):
            yielded = True
            yield sentence
            break  # 短句只要第一句

    else:
        # 中长句：M2.5 快速响应，复杂问题（>25字）才启 M2.7 深度补充
        need_deep = len(user_text) > 25
        fast_q = queue.Queue()
        deep_q = queue.Queue()
        fast_done = threading.Event()
        deep_done = threading.Event()

        def _fast():
            try:
                for s in _stream_gemini(user_text):
                    fast_q.put(s)
            except Exception:
                pass
            finally:
                fast_done.set()

        def _deep():
            try:
                for s in _stream_minimax_model(user_text, MINIMAX_DEEP):
                    deep_q.put(s)
            except Exception:
                pass
            finally:
                deep_done.set()

        threading.Thread(target=_fast, daemon=True).start()
        if need_deep:
            threading.Thread(target=_deep, daemon=True).start()
        else:
            deep_done.set()

        # 先给过渡词（缓存秒播，填补云端等待）
        yield "__TRANSITION__"
        yielded = True

        # M2.5 先说（快）
        fast_text = ""
        while not fast_done.is_set() or not fast_q.empty():
            try:
                sentence = fast_q.get(timeout=0.2)
                clean = sentence.strip()
                if len(re.sub(r'[。？！，、；：\s]', '', clean)) < 2:
                    continue
                fast_text += clean
                yielded = True
                yield sentence
            except queue.Empty:
                # M2.7 已到且 M2.5 还在？让 M2.5 继续说
                if not deep_q.empty():
                    break  # M2.7 到了，切换
                if fast_done.is_set():
                    break

        # M2.7 接管深度回答（跳过重复）
        while not deep_done.is_set() or not deep_q.empty():
            try:
                sentence = deep_q.get(timeout=0.3)
                clean = sentence.strip()
                if len(re.sub(r'[。？！，、；：\s]', '', clean)) < 2:
                    continue
                # 跳过和 M2.5 重复的（子串匹配 + 字符重叠）
                clean_s = clean.strip("。？！，、；：.!? ")
                if fast_text and len(clean_s) >= 3:
                    # 子串包含
                    if clean_s in fast_text or fast_text in clean_s:
                        continue
                    # 字符重叠 > 50%
                    overlap = sum(1 for c in clean_s if c in fast_text)
                    if overlap / len(clean_s) > 0.5:
                        continue
                yielded = True
                yield sentence
            except queue.Empty:
                if deep_done.is_set():
                    break

    # 全部失败：本地兜底
    if not yielded:
        reply = think_local(user_text) or think_ollama(user_text)
        if reply:
            for s in _split_sentences(reply):
                yield s
        else:
            yield "网络不太好，稍后再试。"

    _log_interaction(user_text, tier, "", "(streamed)", time.time() - start_time)
    post_event("response", "cc回复完成", source="brain")
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
