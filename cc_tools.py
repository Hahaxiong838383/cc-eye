"""
cc_tools.py — 贾维斯工具调用桥接层

语音输入 → 意图识别 → 飞书 REST API 直连 → 结果返回 TTS

架构：
  不走 MCP 中间层，Python 进程直接调用飞书 Open API。
  凭证从 ~/mycc/.env 读取（与 open-feishu-mcp 共用同一套）。

支持的工具（第一批）：
  1. feishu_send — 发消息到飞书群聊
  2. feishu_read_doc — 读飞书文档摘要（TODO）
  3. feishu_query_bitable — 查多维表格（TODO）
"""

import json
import re
import time
import requests
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

# ── 飞书 API 配置 ──
FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"
MYCC_ENV = Path.home() / "mycc" / ".env"

# 三机器人群聊（cc + codex + gemini）
DEFAULT_CHAT_ID = "oc_f4a09ffbd709b39c917b927ecb9d96b6"


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    message: str  # TTS 播报文本
    data: dict = field(default_factory=dict)


class FeishuClient:
    """飞书 API 直连客户端（绕过 MCP，Python 进程内直接调用）"""

    def __init__(self) -> None:
        self._app_id: Optional[str] = None
        self._app_secret: Optional[str] = None
        self._token: Optional[str] = None
        self._token_expires: float = 0
        self._session = requests.Session()
        self._session.trust_env = False  # 避免 SOCKS5 代理干扰
        self._load_credentials()

    def _load_credentials(self) -> None:
        """从 mycc/.env 加载飞书凭证"""
        if not MYCC_ENV.exists():
            print("[cc-tools] mycc/.env 不存在，飞书功能不可用")
            return
        for line in MYCC_ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("FEISHU_APP_ID="):
                self._app_id = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("FEISHU_APP_SECRET="):
                self._app_secret = line.split("=", 1)[1].strip().strip('"').strip("'")

        if self._app_id and self._app_secret:
            print("[cc-tools] 飞书凭证已加载")
        else:
            print("[cc-tools] 飞书凭证缺失，请检查 ~/mycc/.env")

    def _get_token(self) -> Optional[str]:
        """获取 tenant_access_token（自动缓存，过期前 5 分钟刷新）"""
        if self._token and time.time() < self._token_expires:
            return self._token

        if not self._app_id or not self._app_secret:
            return None

        try:
            resp = self._session.post(
                f"{FEISHU_BASE_URL}/auth/v3/tenant_access_token/internal/",
                json={
                    "app_id": self._app_id,
                    "app_secret": self._app_secret,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    self._token = data.get("tenant_access_token")
                    expire = data.get("expire", 7200)
                    self._token_expires = time.time() + expire - 300
                    print(f"[cc-tools] 飞书 token 已刷新，有效期 {expire}s")
                    return self._token
                print(f"[cc-tools] 飞书 token 获取失败: {data.get('msg')}")
        except Exception as e:
            print(f"[cc-tools] 飞书 token 请求异常: {e}")
        return None

    def send_message(
        self, text: str, chat_id: str = DEFAULT_CHAT_ID
    ) -> ToolResult:
        """发送文本消息到飞书群聊"""
        token = self._get_token()
        if not token:
            return ToolResult(False, "飞书连接不上，凭证可能有问题。")

        try:
            resp = self._session.post(
                f"{FEISHU_BASE_URL}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}),
                },
                timeout=10,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("code") == 0:
                msg_id = data.get("data", {}).get("message_id", "")
                print(f"[cc-tools] 飞书消息已发送: {msg_id}")
                return ToolResult(True, "消息已发到飞书群。")
            error = data.get("msg", resp.text[:100])
            print(f"[cc-tools] 飞书发送失败: {error}")
            return ToolResult(False, f"发送失败：{error[:30]}")
        except Exception as e:
            print(f"[cc-tools] 飞书发消息异常: {e}")
            return ToolResult(False, f"飞书发消息出错了。")

    def get_recent_messages(
        self, chat_id: str = DEFAULT_CHAT_ID, count: int = 5
    ) -> ToolResult:
        """获取群聊最近消息"""
        token = self._get_token()
        if not token:
            return ToolResult(False, "飞书连接不上。")

        try:
            resp = self._session.get(
                f"{FEISHU_BASE_URL}/im/v1/messages",
                params={
                    "container_id_type": "chat",
                    "container_id": chat_id,
                    "page_size": count,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("code") == 0:
                items = data.get("data", {}).get("items", [])
                summaries = []
                for item in items[:count]:
                    body = item.get("body", {}).get("content", "")
                    try:
                        content = json.loads(body)
                        text = content.get("text", body[:60])
                    except (json.JSONDecodeError, AttributeError):
                        text = body[:60]
                    summaries.append(text[:60])
                summary = "；".join(summaries) if summaries else "没有最近消息"
                return ToolResult(True, f"飞书群最近{len(summaries)}条消息：{summary}", {"messages": summaries})
            return ToolResult(False, "获取消息失败。")
        except Exception as e:
            print(f"[cc-tools] 获取飞书消息异常: {e}")
            return ToolResult(False, "获取飞书消息出错了。")


# ── 意图识别（关键词 + 正则）──

_TOOL_PATTERNS: list[tuple[str, str, int]] = [
    # (regex_pattern, tool_name, content_group_index)
    # 发消息（长匹配在前）
    (r"(?:发一条|发送|发个|发)(?:飞书|群里?)?(?:通知|消息)[：:，,\s]*(.+)", "feishu_send", 1),
    (r"(?:发一条|发送|发个|发)(?:飞书|群里?)[：:，,\s]*(.+)", "feishu_send", 1),
    (r"(?:通知|告诉)(?:群里|大家|飞书群?)[：:，,\s]*(.+)", "feishu_send", 1),
    (r"(?:飞书|群里?)(?:发|说|通知)[：:，,\s]*(.+)", "feishu_send", 1),
    (r"帮我发(?:个|条)?(?:消息|通知)?(?:说|[：:，,\s])(.+)", "feishu_send", 1),
    # 查消息
    (r"(?:飞书|群里?)(?:有什么|最近|新)?(?:消息|通知|动态)", "feishu_read", 0),
    (r"(?:看看|查看|读)(?:飞书|群里?)?(?:消息|通知|动态)", "feishu_read", 0),
]


def detect_tool_intent(text: str) -> Optional[Tuple[str, str]]:
    """
    检测语音输入是否包含工具调用意图。

    Returns:
        (tool_name, extracted_content) or None
    """
    for pattern, tool_name, group_idx in _TOOL_PATTERNS:
        match = re.search(pattern, text)
        if match:
            content = match.group(group_idx) if group_idx > 0 else ""
            content = content.strip() if content else ""
            # 发消息类需要有内容
            if tool_name == "feishu_send" and len(content) < 2:
                continue
            return (tool_name, content)
    return None


# ── 工具执行器（单例懒加载）──

_feishu: Optional[FeishuClient] = None


def _get_feishu() -> FeishuClient:
    global _feishu
    if _feishu is None:
        _feishu = FeishuClient()
    return _feishu


def execute_tool(tool_name: str, content: str) -> ToolResult:
    """执行工具调用，返回 TTS 播报文本"""
    client = _get_feishu()

    if tool_name == "feishu_send":
        return client.send_message(content)
    elif tool_name == "feishu_read":
        return client.get_recent_messages()

    return ToolResult(False, f"这个功能还没实现。")


def try_tool(user_text: str) -> Optional[str]:
    """
    尝试识别并执行工具调用。

    这是给 cc_brain.py 调的入口：
    - 识别到工具意图 → 执行 → 返回结果文本（TTS 播报）
    - 不是工具调用 → 返回 None（走正常 LLM 路径）

    Returns:
        工具执行结果文本，或 None
    """
    intent = detect_tool_intent(user_text)
    if not intent:
        return None

    tool_name, content = intent
    print(f"[cc-tools] 意图识别: {tool_name} → {content[:50]}")

    result = execute_tool(tool_name, content)
    status = "ok" if result.success else "fail"
    print(f"[cc-tools] 执行结果: [{status}] {result.message}")

    return result.message


# ── 测试 ──

if __name__ == "__main__":
    print("=== cc_tools 工具桥接测试 ===\n")

    # 意图识别测试
    test_inputs = [
        "发飞书消息：今天进展不错",
        "发个群通知，贾维斯上线了",
        "通知群里我在忙",
        "飞书发：测试一下",
        "飞书有什么消息",
        "看看群里消息",
        "今天天气怎么样",  # 非工具意图
        "你好",            # 非工具意图
    ]

    for text in test_inputs:
        intent = detect_tool_intent(text)
        if intent:
            print(f"  [{intent[0]}] {text} → 内容: {intent[1][:30]}")
        else:
            print(f"  [--] {text} → 非工具意图")

    print("\n--- 实际调用测试 ---")
    result = try_tool("发飞书消息：贾维斯语音工具桥接测试")
    print(f"结果: {result}")
