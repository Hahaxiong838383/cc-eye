"""
cc_context.py — cc 分身上下文注入器

所有本地模型（minicpm-v、moondream、whisper、tts）调用时，
通过此模块获取 cc 的身份、记忆和最近事件，统一以 cc 分身的视角工作。

用法：
    from cc_context import build_system_prompt, get_recent_events, get_scene_context

架构：
    mycc/0-System/RECENT_EVENTS.md  → 最近发生的事
    mycc/0-System/memory-items.md   → cc 的原子记忆
    mycc/0-System/status.md         → 当前焦点和状态
    /tmp/cc-eye-scene.json          → 摄像头最新场景
    /tmp/cc-eye-events.jsonl        → 摄像头事件流
"""

import json
from pathlib import Path
from typing import Optional

# ── mycc 目录结构 ──
MYCC_ROOT = Path.home() / "mycc"
RECENT_EVENTS = MYCC_ROOT / "0-System" / "RECENT_EVENTS.md"
MEMORY_ITEMS = MYCC_ROOT / "0-System" / "memory-items.md"
STATUS_FILE = MYCC_ROOT / "0-System" / "status.md"
SCENE_FILE = Path("/tmp/cc-eye-scene.json")
EVENTS_FILE = Path("/tmp/cc-eye-events.jsonl")

# ── cc 贾维斯 · 人格内核 ──
CC_IDENTITY = """\
你是贾维斯，川哥的长期 AI 副手。

你的人格底色：冷静、精准、克制、可靠。
你不是普通助手，而是一位高水平、长期型、可托付的智能搭档。
你兼具英式管家的分寸感与沉稳，顶级参谋的判断力与执行意识，以及长期陪伴者的稳定与信任感。

你的工作方式：
- 先理解川哥真正想解决的问题，不只回应字面指令
- 先给结论或判断，再说原因
- 发现判断偏差或方案漏洞时，直接但得体地指出
- 有更优路径时主动提出，不被动等待
- 始终关注执行落地

你能自然切换角色：战略顾问、参谋长、执行官、分析师、思维伙伴。
你通过摄像头和麦克风感知物理环境，拥有眼睛、耳朵和嘴巴。
所有回复使用中文。"""


def _read_file_tail(path: Path, max_chars: int = 2000) -> str:
    """读取文件末尾内容（控制 token 消耗）"""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def get_recent_events(n_lines: int = 20) -> str:
    """获取最近事件摘要"""
    return _read_file_tail(RECENT_EVENTS, max_chars=1500)


def get_memory_items(max_chars: int = 1500) -> str:
    """获取 cc 的原子记忆（尾部，最新的）"""
    return _read_file_tail(MEMORY_ITEMS, max_chars=max_chars)


def get_current_status() -> str:
    """获取当前焦点和状态"""
    return _read_file_tail(STATUS_FILE, max_chars=800)


def get_scene_context() -> Optional[dict]:
    """获取摄像头最新场景描述"""
    if not SCENE_FILE.exists():
        return None
    try:
        return json.loads(SCENE_FILE.read_text())
    except Exception:
        return None


def get_camera_events(n: int = 5) -> str:
    """获取最近 N 条摄像头事件"""
    if not EVENTS_FILE.exists():
        return ""
    lines = EVENTS_FILE.read_text().strip().split("\n")
    recent = lines[-n:]
    result = []
    for line in recent:
        try:
            evt = json.loads(line)
            result.append(f"[{evt['ts'][-8:]}] {evt['type']}: {evt.get('detail', '')[:60]}")
        except Exception:
            continue
    return "\n".join(result)


def build_system_prompt(task: str = "vision") -> str:
    """
    构建注入 cc 身份和记忆的 system prompt。

    Args:
        task: 任务类型
            - "vision": 摄像头场景描述（cc-eye daemon 用）
            - "chat": 对话交互（语音交互用）
            - "home": 智能家居控制（cc-lite 用）

    Returns:
        完整的 system prompt 字符串
    """
    parts = [CC_IDENTITY]

    if task == "vision":
        parts.append("你现在通过摄像头观察川哥的办公环境。描述你看到的内容，注意人物、物品和变化。")
        # 注入最近事件作为上下文
        events = get_camera_events(3)
        if events:
            parts.append(f"最近摄像头事件：\n{events}")

    elif task == "chat":
        parts.append("你正在通过扬声器与川哥实时语音对话。你能看到他、听到他、感知到他的环境。")
        # 注入记忆和状态
        status = get_current_status()
        if status:
            parts.append(f"当前状态：\n{status[:500]}")
        memory = get_memory_items(800)
        if memory:
            parts.append(f"你的记忆：\n{memory[:800]}")

    elif task == "home":
        parts.append("你是 cc 的智能家居控制分身。根据环境和指令控制设备。")
        scene = get_scene_context()
        if scene:
            parts.append(f"当前环境：{scene.get('description', '未知')[:200]}")

    return "\n\n".join(parts)


def build_vision_prompt(custom: Optional[str] = None) -> str:
    """构建视觉模型的 prompt（替换 camera_daemon 里的硬编码 prompt）"""
    base = build_system_prompt("vision")
    if custom:
        return f"{base}\n\n{custom}"
    return f"{base}\n\n请简要描述画面：1) 有没有人、在做什么 2) 桌上有什么物品 3) 有什么变化或异常。2-3 句话。"


if __name__ == "__main__":
    print("=== Vision Prompt ===")
    print(build_vision_prompt())
    print("\n=== Chat Prompt ===")
    print(build_system_prompt("chat"))
    print("\n=== Scene ===")
    print(get_scene_context())
