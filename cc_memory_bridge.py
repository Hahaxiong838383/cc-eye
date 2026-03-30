"""
cc_memory_bridge.py — 贾维斯感知记忆桥接

每日汇总 cc-eye 的交互和视觉事件，沉淀到 mycc 记忆体系：
  1. 读 interactions.jsonl + events.jsonl（今天的数据）
  2. 用本地 LLM (qwen2.5:3b) 生成摘要
  3. 追加到 RECENT_EVENTS.md
  4. 有价值的事实提取为 memory-items 原子条目

隐私红线：
  - 只存文字摘要，不存音频/图像路径
  - 闲聊只存主题，决策/指令类可存具体内容
  - 不存连续监控数据

用法：
  python cc_memory_bridge.py          # 汇总今天的数据
  python cc_memory_bridge.py --dry    # 预览不写入
"""

import json
import sys
import requests
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# ── 路径 ──
MYCC_ROOT = Path.home() / "mycc"
RECENT_EVENTS = MYCC_ROOT / "0-System" / "RECENT_EVENTS.md"
INTERACTIONS_FILE = Path("/tmp/cc-eye-interactions.jsonl")
EVENTS_FILE = Path("/tmp/cc-eye-events.jsonl")

# ── 本地 LLM ──
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"


def _load_today_interactions() -> list[dict]:
    """读取今天的交互记录"""
    if not INTERACTIONS_FILE.exists():
        return []
    today_str = date.today().isoformat()
    entries = []
    for line in INTERACTIONS_FILE.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("ts", "").startswith(today_str):
                entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


def _load_today_events() -> list[dict]:
    """读取今天的视觉事件（只取有意义的：person_appeared/left, scene_described, detail_scan）"""
    if not EVENTS_FILE.exists():
        return []
    today_str = date.today().isoformat()
    significant_types = {"person_appeared", "person_left", "scene_described", "detail_scan"}
    entries = []
    for line in EVENTS_FILE.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            if (entry.get("ts", "").startswith(today_str)
                    and entry.get("type") in significant_types):
                entries.append(entry)
        except json.JSONDecodeError:
            continue
    return entries


def _summarize_with_llm(interactions: list[dict], events: list[dict]) -> Optional[str]:
    """用本地 LLM 生成今日贾维斯交互摘要"""
    # 构建摘要素材
    parts = []

    if interactions:
        parts.append(f"今天贾维斯语音交互 {len(interactions)} 次：")
        for i in interactions[-20:]:  # 最多取最近 20 条
            ts = i["ts"][11:16]  # HH:MM
            user_input = i.get("input", "")
            route = i.get("route", "")
            scene = i.get("scene", "")
            reply = i.get("cloud_reply", "") or i.get("local_reply", "")
            line = f"  {ts} 川哥说「{user_input}」→ 贾维斯回「{reply}」"
            if scene:
                line += f"（场景：{scene[:60]}）"
            parts.append(line)

    if events:
        # 去重：同类型事件合并计数
        type_counts: dict[str, int] = {}
        last_detail: dict[str, str] = {}
        for e in events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            last_detail[t] = e.get("detail", "")[:80]

        parts.append(f"今天视觉事件：")
        for t, count in type_counts.items():
            label = {
                "person_appeared": "人到达",
                "person_left": "人离开",
                "scene_described": "场景描述",
                "detail_scan": "精细扫描",
            }.get(t, t)
            parts.append(f"  {label} x{count}，最后一次：{last_detail[t]}")

    if not parts:
        return None

    raw_data = "\n".join(parts)

    prompt = (
        "你是 cc 的记忆摘要模块。根据以下贾维斯（cc-eye）今日交互和视觉事件数据，"
        "生成一段简洁的每日摘要（3-5 句话）。\n\n"
        "要求：\n"
        "1. 只保留有信息量的内容（决策、指令、异常事件、工作节奏）\n"
        "2. 闲聊只提主题，不记原文\n"
        "3. 用事件级语言（人来了/走了/聊了什么主题/场景变化）\n"
        "4. 不要用 markdown 格式，纯文本\n"
        "5. 如果数据太少或全是无意义的问候，直接回复「今日无有效交互」\n\n"
        f"原始数据：\n{raw_data}\n\n"
        "摘要："
    )

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.post(
            OLLAMA_API,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[memory-bridge] ollama 错误: {resp.status_code}")
            return None
        text = resp.json().get("message", {}).get("content", "").strip()
        return text if text else None
    except Exception as e:
        print(f"[memory-bridge] LLM 摘要失败: {e}")
        return None


def _extract_memory_items(interactions: list[dict], events: list[dict]) -> Optional[str]:
    """提取有价值的原子记忆条目（决策、指令、重要发现）"""
    # 过滤：只看非闲聊的交互（路由为 complex_parallel 或 tool 调用）
    significant = [
        i for i in interactions
        if i.get("route") in ("complex_parallel", "tool")
    ]
    if not significant and len(events) < 5:
        return None  # 不够有价值

    parts = []
    for i in significant[-10:]:
        parts.append(f"川哥问：{i.get('input', '')}，贾维斯答：{i.get('cloud_reply', '') or i.get('local_reply', '')}")

    if not parts:
        return None

    raw = "\n".join(parts)

    prompt = (
        "从以下贾维斯语音交互中，提取值得长期记住的原子事实。\n"
        "每条事实独立成行，前缀用 '- '。只提取：\n"
        "1. 决策/指令（川哥明确说了要做什么）\n"
        "2. 偏好发现（川哥喜欢/不喜欢什么）\n"
        "3. 日程/时间点（提到的安排）\n"
        "如果没有值得记的，回复「无」\n\n"
        f"数据：\n{raw}\n\n"
        "原子事实："
    )

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.post(
            OLLAMA_API,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 150},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        text = resp.json().get("message", {}).get("content", "").strip()
        if text and text != "无":
            return text
        return None
    except Exception:
        return None


def _append_to_recent_events(summary: str) -> None:
    """追加摘要到 RECENT_EVENTS.md"""
    today_str = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M")

    entry = f"- `{now_time}` **贾维斯感知记忆桥接**：{summary}\n"

    content = RECENT_EVENTS.read_text() if RECENT_EVENTS.exists() else ""

    # 找到今天的日期 section，追加
    date_header = f"## {today_str}"
    if date_header in content:
        # 在该 section 末尾追加
        idx = content.index(date_header)
        # 找下一个 ## 或文件末尾
        next_section = content.find("\n## ", idx + len(date_header))
        if next_section == -1:
            insert_pos = len(content)
        else:
            insert_pos = next_section

        content = content[:insert_pos].rstrip() + "\n" + entry + "\n" + content[insert_pos:]
    else:
        # 在 --- 后面插入新的日期 section
        if "---\n" in content:
            last_separator = content.rfind("---\n")
            # 在第一个日期 section 之前插入
            first_date = content.find("\n## 2026", last_separator)
            if first_date != -1:
                insert_pos = first_date + 1
            else:
                insert_pos = last_separator + 4
            content = content[:insert_pos] + f"\n{date_header}\n\n{entry}\n" + content[insert_pos:]
        else:
            content += f"\n{date_header}\n\n{entry}\n"

    RECENT_EVENTS.write_text(content)
    print(f"[memory-bridge] 已写入 RECENT_EVENTS.md")


def main() -> None:
    dry_run = "--dry" in sys.argv

    print(f"[memory-bridge] 开始汇总 {date.today()} 的贾维斯感知数据...")

    interactions = _load_today_interactions()
    events = _load_today_events()

    print(f"[memory-bridge] 交互 {len(interactions)} 条，视觉事件 {len(events)} 条")

    if not interactions and not events:
        print("[memory-bridge] 今天无感知数据，跳过")
        return

    # 1. 生成摘要
    summary = _summarize_with_llm(interactions, events)
    if not summary or "无有效交互" in summary:
        print(f"[memory-bridge] 摘要结果：{summary or '空'}，跳过写入")
        return

    print(f"[memory-bridge] 摘要：{summary}")

    # 2. 提取原子记忆
    memory_items = _extract_memory_items(interactions, events)
    if memory_items:
        print(f"[memory-bridge] 原子记忆：{memory_items}")

    if dry_run:
        print("[memory-bridge] --dry 模式，不写入文件")
        return

    # 3. 写入 RECENT_EVENTS
    _append_to_recent_events(summary)

    # 4. 原子记忆由 cc 的进化自查任务在 22:35 读取 RECENT_EVENTS 时自动提取
    if memory_items:
        print(f"[memory-bridge] 原子记忆已提取，等待进化自查任务处理")
        tmp_file = Path("/tmp/cc-eye-memory-candidates.txt")
        tmp_file.write_text(f"# {date.today()} 贾维斯感知提取\n\n{memory_items}\n")

    # 5. 合并视觉事实到候选记忆
    visual_facts_file = Path("/tmp/cc-eye-visual-facts.jsonl")
    if visual_facts_file.exists():
        try:
            facts = []
            seen = set()
            for line in visual_facts_file.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                fact = entry.get("fact", "").strip()
                if fact and fact != "无" and fact not in seen:
                    facts.append(fact)
                    seen.add(fact)
            if facts:
                # 追加到候选文件
                tmp_file = Path("/tmp/cc-eye-memory-candidates.txt")
                existing = tmp_file.read_text() if tmp_file.exists() else ""
                visual_section = "\n\n## [user/习惯] 视觉观察\n"
                for f in facts[-20:]:  # 最多保留 20 条
                    visual_section += f"- ★★ {f} [{date.today()}] #视觉\n"
                tmp_file.write_text(existing + visual_section)
                print(f"[memory-bridge] 视觉事实合并: {len(facts)} 条")
            # 清空已处理的事实文件
            visual_facts_file.write_text("")
        except Exception as e:
            print(f"[memory-bridge] 视觉事实合并失败: {e}")

    print("[memory-bridge] 完成")


if __name__ == "__main__":
    main()
