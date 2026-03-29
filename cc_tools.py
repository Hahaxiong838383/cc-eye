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
    # 音乐播放（优先匹配，避免被飞书模式吞掉）
    # 查询当前播放状态（放最前面）
    (r"(?:现在|正在)?(?:放|播)(?:的)?(?:什么|啥)(?:歌|音乐|曲)?", "music_state", 0),
    (r"(?:这是|这首)(?:什么|啥)(?:歌|音乐|曲)", "music_state", 0),
    # 推荐/每日推荐（在 play 前面，避免"播放推荐"被 play 吞掉）
    (r"(?:播放|放|听)?(?:每日|今日|今天的?)?推荐(?:的?歌曲?|的?音乐|的?歌)?", "music_random", 0),
    # 无具体歌名的通用请求 → 随机/推荐
    (r"(?:播放|放|听|来点|播点|放点|来一?首|点一?首|放一?首|播一?首)(?:歌曲?|音乐|的?歌|的?音乐)$", "music_random", 0),
    (r"(?:随便|随机)(?:播播|放放|听听|来点|放点)", "music_random", 0),
    # 有具体歌名/歌手 → 搜索播放
    (r"(?:播放|放|听|来一?首|点一?首|播一?首)(?:一下)?(?:一首)?(.+?)(?:的歌|的音乐)?$", "music_play", 1),
    (r"(?:搜|搜索|找|查)(?:一下)?(.+?)(?:的歌|的音乐|歌曲)$", "music_search", 1),
    (r"(?:搜|搜索|找)(?:一下)?(?:歌曲?|音乐|歌手)(.+)$", "music_search", 1),
    (r"(?:暂停|停止|停一下|别播了|关掉|停)(?:音乐|歌曲?|歌|播放)?", "music_stop", 0),
    (r"(?:继续|继续播|接着放|接着播|恢复播放)(?:音乐|歌曲?|歌)?", "music_resume", 0),
    (r"(?:下一首|切歌|换一首|跳过|下一曲)", "music_next", 0),
    (r"(?:上一首|上一曲|前一首)", "music_prev", 0),
    (r"(?:声音|音量)(?:大一?点|调大|加大|提高)", "music_vol_up", 0),
    (r"(?:声音|音量)(?:小一?点|调小|减小|降低)", "music_vol_down", 0),
    (r"(?:每日|今日)?推荐(?:歌曲?|音乐|歌)?", "music_recommend", 0),
    # 注意：安静/闭嘴等由 cc_jarvis_v3.py 的 QUIET_WORDS 直接处理，不走工具
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


# 音乐相关关键词（正则没匹配到但可能是音乐请求）
_MUSIC_HINTS = {"歌", "音乐", "曲", "嗨", "安静", "轻松", "助眠", "提神",
                "氛围", "伤感", "开心", "工作", "加班", "跑步", "运动",
                "古典", "爵士", "摇滚", "民谣", "电子", "嘻哈", "说唱",
                "播播", "放放", "听听"}


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

    # 兜底：正则没匹配到，但包含音乐关键词 → Gemini 智能调度
    if any(h in text for h in _MUSIC_HINTS):
        return ("music_smart", text)

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
    # 音乐智能调度（Gemini）
    if tool_name == "music_smart":
        return _execute_music_smart(content)

    # 音乐工具（正则直接执行）
    if tool_name.startswith("music_"):
        return _execute_music(tool_name, content)

    # 飞书工具
    client = _get_feishu()
    if tool_name == "feishu_send":
        return client.send_message(content)
    elif tool_name == "feishu_read":
        return client.get_recent_messages()

    return ToolResult(False, f"这个功能还没实现。")


# ── 音乐工具（ncm-cli 封装）──

import subprocess


def _ncm_env() -> dict:
    """清除代理环境变量，确保直连"""
    return {k: v for k, v in __import__("os").environ.items()
            if not k.lower().startswith(("http_proxy", "https_proxy", "all_proxy", "socks"))}


def _run_ncm_bg(args: list):
    """后台执行 ncm-cli 命令（不等结果，用于播放等长时间运行的命令）"""
    cmd = ["ncm-cli"] + args
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_ncm_env())
    except Exception as e:
        print(f"[cc-tools] ncm-cli 后台启动失败: {e}")


def _run_ncm(args: list, timeout: int = 15) -> dict:
    """执行 ncm-cli 命令，返回 JSON 结果（直连，不走代理）"""
    cmd = ["ncm-cli"] + args + ["--output", "json"]
    env = _ncm_env()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        return {"error": result.stderr.strip() or "命令执行失败"}
    except subprocess.TimeoutExpired:
        return {"error": "命令超时"}
    except json.JSONDecodeError:
        return {"error": "解析结果失败"}
    except Exception as e:
        return {"error": str(e)}


def _clean_keyword(content: str) -> str:
    """清洗音乐搜索关键词"""
    keyword = content.strip()
    keyword = re.sub(r"[。，！？、\s]+", "", keyword)
    keyword = re.sub(r"^(?:听|放|播放|来一?首|点一?首)(?:一下)?", "", keyword)
    keyword = re.sub(r"(?:的歌曲?|的音乐|歌曲?|音乐)$", "", keyword)
    return keyword.strip()


def _find_playable(records: list) -> list:
    """从搜索结果中过滤可播放的歌曲（visible != false）"""
    playable = []
    for s in records:
        if s.get("visible") is False:
            continue
        playable.append(s)
    return playable


def _play_song(song: dict) -> bool:
    """播放单曲，返回是否成功。播放后自动隐藏网易云 App 窗口。"""
    enc_id = str(song.get("id", ""))
    orig_id = str(song.get("originalId", ""))
    if not enc_id or not orig_id:
        return False
    _run_ncm_bg([
        "play", "--song",
        "--encrypted-id", enc_id,
        "--original-id", orig_id,
    ])
    # 延迟隐藏网易云 App 窗口（后台播放）
    import threading
    def _hide_app():
        import time as _t
        _t.sleep(2)
        try:
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to set visible of process "NeteaseMusic" to false'],
                capture_output=True, timeout=3,
            )
        except Exception:
            pass
    threading.Thread(target=_hide_app, daemon=True).start()
    return True


def _execute_music(tool_name: str, content: str) -> ToolResult:
    """执行音乐相关工具"""

    if tool_name == "music_play":
        keyword = _clean_keyword(content)
        if not keyword:
            return ToolResult(False, "你想听什么歌？告诉我歌名或歌手。")

        # 按 skill 规范加 --userInput
        data = _run_ncm(["search", "song", "--keyword", keyword,
                         "--userInput", f"播放{keyword}"])
        if "error" in data:
            return ToolResult(False, f"搜索出了点问题。")

        records = data.get("data", {}).get("records", [])
        playable = _find_playable(records)
        if not playable:
            return ToolResult(False, f"没找到{keyword}可播放的歌。")

        song = playable[0]
        song_name = song.get("name", "未知")
        artist = song["artists"][0]["name"] if song.get("artists") else "未知"

        _play_song(song)

        # 后续可播放的歌加到队列
        for s in playable[1:4]:
            enc = str(s.get("id", ""))
            orig = str(s.get("originalId", ""))
            if enc and orig:
                _run_ncm_bg(["queue", "add", "--encrypted-id", enc, "--original-id", orig])

        return ToolResult(True, f"正在播放{artist}的{song_name}。")

    elif tool_name == "music_random":
        data = _run_ncm(["recommend", "daily", "--limit", "10",
                         "--userInput", "随机播放推荐音乐"])
        records = data.get("data", []) if isinstance(data.get("data"), list) else data.get("data", {}).get("records", [])
        playable = _find_playable(records) if records else []

        if not playable:
            return ToolResult(False, "推荐没拿到，你告诉我想听什么吧。")

        song = playable[0]
        song_name = song.get("name", "未知")
        artist = song["artists"][0]["name"] if song.get("artists") else "未知"
        _play_song(song)

        for s in playable[1:5]:
            enc = str(s.get("id", ""))
            orig = str(s.get("originalId", ""))
            if enc and orig:
                _run_ncm_bg(["queue", "add", "--encrypted-id", enc, "--original-id", orig])

        return ToolResult(True, f"给你放一首{artist}的{song_name}。")

    elif tool_name == "music_search":
        keyword = _clean_keyword(content)
        if not keyword:
            return ToolResult(False, "你想搜什么歌？")

        data = _run_ncm(["search", "song", "--keyword", keyword,
                         "--userInput", f"搜索{keyword}"])
        records = data.get("data", {}).get("records", [])
        playable = _find_playable(records)

        if not playable:
            return ToolResult(False, f"没找到{keyword}相关的歌。")

        results = []
        for s in playable[:3]:
            name = s.get("name", "")
            artist = s["artists"][0]["name"] if s.get("artists") else ""
            results.append(f"{artist}的{name}")

        return ToolResult(True, f"找到了：{'，'.join(results)}。要播放哪首？")

    elif tool_name == "music_stop":
        _run_ncm(["stop"])
        return ToolResult(True, "已停止播放。")

    elif tool_name == "music_resume":
        _run_ncm(["resume"])
        return ToolResult(True, "继续播放。")

    elif tool_name == "music_next":
        result = _run_ncm(["next"])
        if result.get("success"):
            return ToolResult(True, "下一首。")
        # 队列没有下一首，自动播推荐
        return _execute_music("music_random", "")

    elif tool_name == "music_prev":
        _run_ncm(["prev"])
        return ToolResult(True, "上一首。")

    elif tool_name == "music_vol_up":
        _run_ncm(["volume", "80"])
        return ToolResult(True, "音量调大了。")

    elif tool_name == "music_vol_down":
        _run_ncm(["volume", "30"])
        return ToolResult(True, "音量调小了。")

    elif tool_name == "music_state":
        data = _run_ncm(["state"])
        if "error" in data:
            return ToolResult(True, "当前没有在播放。")
        state = data.get("data", data.get("state", {}))
        status = state.get("status", "stopped")
        if status == "playing":
            title = state.get("title", "")
            artist = state.get("artist", "")
            if title:
                return ToolResult(True, f"正在播放{artist}的{title}。")
        return ToolResult(True, "当前没有在播放。")

    elif tool_name == "music_recommend":
        data = _run_ncm(["recommend", "daily", "--limit", "5",
                         "--userInput", "今日推荐歌曲"])
        records = data.get("data", []) if isinstance(data.get("data"), list) else data.get("data", {}).get("records", [])
        playable = _find_playable(records) if records else []

        if not playable:
            return ToolResult(False, "今天的推荐还没出来。")

        results = []
        for s in playable[:3]:
            name = s.get("name", "")
            artist = s["artists"][0]["name"] if s.get("artists") else ""
            results.append(f"{artist}的{name}")

        return ToolResult(True, f"今天推荐：{'，'.join(results)}。要听哪首？")


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


# ── Gemini 智能音乐调度 ──

_MUSIC_SMART_PROMPT = """\
你是音乐搜索助手。根据用户请求，输出一个 JSON，用于 ncm-cli 搜索。
只输出 JSON，不要任何其他文字。

规则：
- keyword 是给网易云音乐搜索用的关键词，2-4个词，中文
- 理解用户的情绪、场景、风格偏好，转化为搜索关键词
- 如果用户提到具体歌手/歌名，直接用

示例：
用户：放点轻松的 → {"keyword": "轻音乐 放松 纯音乐"}
用户：来个适合加班的 → {"keyword": "工作 专注 轻音乐"}
用户：心情不好 → {"keyword": "治愈 温暖 民谣"}
用户：嗨一点的 → {"keyword": "电子 嗨曲 节奏"}
用户：古典音乐 → {"keyword": "古典 钢琴曲"}
"""


def _call_gemini_sync(prompt: str, user_text: str, timeout: int = 8) -> Optional[str]:
    """同步调用 Gemini 代理（非流式），返回文本结果"""
    from cc_brain import _load_gemini_proxy_config, _get_gemini_proxy_session
    config = _load_gemini_proxy_config()
    api_key = config.get("api_key")
    if not api_key:
        return None

    url = config["base_url"].rstrip("/") + "/chat/completions"
    try:
        resp = _get_gemini_proxy_session().post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.3,
                "max_tokens": 100,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[cc-tools] Gemini 调用失败: {e}")
    return None


def _execute_music_smart(user_text: str) -> ToolResult:
    """Gemini 智能音乐调度：理解模糊请求 → 生成搜索关键词 → 搜索播放"""
    print(f"[cc-tools] 音乐智能调度: {user_text}")

    # 1. 调 Gemini 获取搜索关键词
    result = _call_gemini_sync(_MUSIC_SMART_PROMPT, user_text)
    if not result:
        return ToolResult(False, "网络不太好，你直接告诉我歌名吧。")

    # 2. 解析 JSON
    try:
        # 清理可能的 markdown 代码块
        clean = result.strip().strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        data = json.loads(clean)
        keyword = data.get("keyword", "").strip()
    except (json.JSONDecodeError, AttributeError):
        # Gemini 返回的不是 JSON，直接当关键词用
        keyword = result.strip()[:20]

    if not keyword:
        return ToolResult(False, "没理解你想听什么，换个说法试试？")

    print(f"[cc-tools] Gemini → 关键词: {keyword}")

    # 3. 搜索播放（复用 _execute_music 的 music_play 逻辑）
    search_data = _run_ncm(["search", "song", "--keyword", keyword,
                            "--userInput", f"智能推荐: {user_text}"])
    if "error" in search_data:
        return ToolResult(False, "搜索出了点问题。")

    records = search_data.get("data", {}).get("records", [])
    playable = _find_playable(records)
    if not playable:
        return ToolResult(False, f"没找到合适的歌，换个说法试试？")

    song = playable[0]
    song_name = song.get("name", "未知")
    artist = song["artists"][0]["name"] if song.get("artists") else "未知"

    _play_song(song)

    # 后续加到队列
    for s in playable[1:4]:
        enc = str(s.get("id", ""))
        orig = str(s.get("originalId", ""))
        if enc and orig:
            _run_ncm_bg(["queue", "add", "--encrypted-id", enc, "--original-id", orig])

    return ToolResult(True, f"给你找了{artist}的{song_name}，听听看。")


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
