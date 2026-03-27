"""
cc_interact.py — cc 贾维斯多模态实时交互主循环

融合视觉感知 + 语音交互，实现完整的多模态实时交互：
  眼（摄像头）：看到环境变化 → 主动播报
  耳（麦克风）：听到说话 → whisper 识别 → 理解 → 语音回复
  嘴（扬声器）：edge-tts 贾维斯语音输出

用法：
  cd ~/mycc/2-Projects/cc-eye && source .venv/bin/activate
  python cc_interact.py

架构：
  主线程：语音监听循环（listen → understand → speak）
  视觉线程：读取 camera_daemon 的场景描述和事件，触发主动播报
  事件队列：统一调度语音输出（避免同时说话冲突）
"""

import json
import threading
import time
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional

from cc_voice import say, say_stream, set_aec, is_echo
from cc_listen import listen_once, SpeechSegment, calibrate_mic
from cc_aec import EchoCanceller
from cc_brain import think, think_stream
from state_fusion import inject_voice_emotion
from cc_events import (
    post_event,
    sync_from_daemon_events,
    seconds_since_last_interaction,
)
from cc_context import (
    build_system_prompt,
    get_scene_context,
    get_camera_events,
    CC_IDENTITY,
)

# ── 配置 ──
SCENE_FILE = Path("/tmp/cc-eye-scene.json")
EVENTS_FILE = Path("/tmp/cc-eye-events.jsonl")
PERSON_FLAG = Path("/tmp/cc-eye-person-arrived.flag")
SPEAK_QUEUE_TIMEOUT = 0.5    # 语音队列检查间隔
VISION_CHECK_INTERVAL = 5.0  # 视觉事件检查间隔（秒）


class CcInteract:
    """cc 贾维斯多模态交互引擎"""

    def __init__(self):
        self.speak_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._last_person_flag_ts = ""
        self._last_scene_ts = ""
        self._last_scene_desc = ""
        self._greeted = False
        self._start_time = time.time()
        # NLMS 回声消除器：用 TTS 波形做参考，从麦克风输入中减去回声
        self.aec = EchoCanceller(filter_length=4096, mu=0.05, tail_duration=0.5)

    def start(self) -> None:
        """启动多模态交互"""
        print("=" * 50)
        print("  cc 贾维斯 — 多模态实时交互")
        print("  眼：摄像头场景感知（camera_daemon 提供）")
        print("  耳：麦克风语音监听（whisper STT）")
        print("  嘴：扬声器语音输出（edge-tts）")
        print("  Ctrl+C 退出")
        print("=" * 50)

        # 初始化 AEC：注入到 cc_voice，播放时自动加载参考信号
        set_aec(self.aec)
        print("[cc-interact] NLMS 回声消除器已就绪")

        # 校准麦克风
        calibrate_mic(duration=2.0)

        # 启动视觉事件监控线程
        vision_thread = threading.Thread(
            target=self._vision_monitor_loop,
            daemon=True,
            name="vision-monitor",
        )
        vision_thread.start()

        # 启动语音输出线程（统一调度）
        speak_thread = threading.Thread(
            target=self._speak_loop,
            daemon=True,
            name="speaker",
        )
        speak_thread.start()

        # 启动事件（开场白由 _time_aware_greeting 处理）
        post_event("system", "贾维斯多模态交互启动", source="interact")

        # 主线程：语音监听循环
        self._listen_loop()

    def _listen_loop(self) -> None:
        """主线程：持续监听语音输入"""
        print("[cc-interact] 语音监听已启动，请说话...")

        while not self.stop_event.is_set():
            try:
                segment = listen_once(timeout=60.0, aec=self.aec)
                if segment and segment.text:
                    self._handle_speech(segment)
            except KeyboardInterrupt:
                print("\n[cc-interact] 用户中断")
                self.stop_event.set()
                break
            except Exception as e:
                print(f"[cc-interact] 监听异常: {e}")
                time.sleep(1)

    # 唤醒词：只有包含"贾维斯"时才响应
    WAKE_WORDS = ["贾维斯", "jarvis", "嘉维斯", "贾维丝"]

    def _handle_speech(self, segment: SpeechSegment) -> None:
        """处理识别出的语音（需要唤醒词触发）"""
        text = segment.text.strip()
        if not text:
            return

        # 文本层回声过滤：如果识别出的是自己 TTS 播放的内容，丢弃
        if is_echo(text):
            print(f"[cc-interact] 回声过滤: {text[:30]}...")
            return

        # 检查唤醒词
        text_lower = text.lower()
        wake_hit = False
        for w in self.WAKE_WORDS:
            if w in text_lower:
                wake_hit = True
                # 去掉唤醒词，保留指令部分
                text = text_lower.replace(w, "").strip()
                break

        if not wake_hit:
            # 没有唤醒词，忽略（不打印，减少噪音）
            return

        print(f"\n[cc-interact] 川哥说: {text}")

        # 注入语音情感到融合引擎 + 发布事件流
        if segment.emotion:
            inject_voice_emotion(segment.emotion)
        if segment.emotion and segment.emotion != "neutral":
            post_event(
                "speech",
                f"语音情感: {segment.emotion_cn} ({segment.emotion})",
                source="interact",
            )
        if segment.audio_events:
            for evt in segment.audio_events:
                if evt != "speech":
                    post_event("speech", f"音频事件: {evt}", source="interact")

        # 获取当前视觉上下文
        scene = get_scene_context()
        scene_desc = ""
        if scene:
            scene_desc = f"（当前场景：{scene.get('description', '')[:100]}）"

        # 流式管线：LLM 按句 yield → 每句立即 TTS 播放
        for sentence in think_stream(text):
            print(f"[cc-interact] 🗣️ {sentence}")
            try:
                say_stream(sentence)
            except Exception:
                say(sentence)  # ffplay 不可用时降级到 afplay

    def _generate_response(self, user_text: str, scene_context: str) -> str:
        """
        生成回复。

        当前版本：基于关键词的简单规则引擎。
        后续升级：接入本地 LLM（qwen2.5）或 Claude API 做真正的对话。
        """
        text_lower = user_text.lower()

        # 问候类
        if any(w in text_lower for w in ["你好", "早上好", "下午好", "晚上好", "嗨"]):
            return f"川哥好！我在线，随时听你指挥。{scene_context}"

        # 询问视觉
        if any(w in text_lower for w in ["看看", "看到", "环境", "周围", "画面"]):
            scene = get_scene_context()
            if scene and scene.get("description"):
                return f"我看到：{scene['description']}"
            return "摄像头正在工作，但还没有最新的场景描述。稍等一下。"

        # 时间
        if any(w in text_lower for w in ["几点", "时间", "现在"]):
            now = datetime.now().strftime("%H:%M")
            return f"现在是{now}。"

        # 状态查询
        if any(w in text_lower for w in ["状态", "怎么样", "还好"]):
            return "系统运行正常。摄像头、语音、视觉模型都在线。"

        # 默认回复
        return f"收到：{user_text}。目前我的语音理解还比较简单，复杂指令请在终端里找我。"

    # 主动交互配置
    PROACTIVE_COOLDOWN = 30.0    # 主动说话冷却（秒），避免太吵
    SILENCE_MUTE_AFTER = 300.0   # 5 分钟无互动后停止主动说话

    def _vision_monitor_loop(self) -> None:
        """视觉事件监控 + 主动交互线程"""
        print("[cc-interact] 视觉事件监控 + 主动交互已启动")
        last_proactive_time = 0.0

        while not self.stop_event.is_set():
            try:
                # 同步 daemon 事件到统一事件流
                sync_from_daemon_events()

                now = time.time()
                idle_seconds = seconds_since_last_interaction()
                # 启动后首次允许主动说话；之后 5 分钟无互动才静默
                is_first_run = not self._greeted
                can_proactive = (
                    (now - last_proactive_time) > self.PROACTIVE_COOLDOWN
                    and (is_first_run or idle_seconds < self.SILENCE_MUTE_AFTER)
                )

                # 检测人到达 → 主动打招呼
                if can_proactive and self._check_person_arrived():
                    last_proactive_time = now

                # 检测场景重大变化 → 主动播报
                if can_proactive and self._check_scene_change():
                    last_proactive_time = now

                # 时间感知问候（首次启动后 10 秒内不触发）
                if can_proactive and not self._greeted and now - self._start_time > 10:
                    self._time_aware_greeting()
                    self._greeted = True
                    last_proactive_time = now

            except Exception as e:
                print(f"[cc-interact] 视觉监控异常: {e}")

            self.stop_event.wait(VISION_CHECK_INTERVAL)

    def _check_person_arrived(self) -> bool:
        """检测是否有人到达，主动打招呼（只问候一次，人走了再来才重新触发）。"""
        if not PERSON_FLAG.exists():
            self._person_greeted = False  # 人走了，重置
            return False

        flag_ts = PERSON_FLAG.read_text().strip()
        if flag_ts == self._last_person_flag_ts:
            return False  # flag 没变

        self._last_person_flag_ts = flag_ts

        # 已经问候过了，不重复
        if getattr(self, '_person_greeted', False):
            return False

        self._person_greeted = True
        post_event("face", "检测到有人到达", source="interact")

        # 本地秒回 + Gemini 异步补充
        self._enqueue_speak("有人来了。")

        def _gen_greeting():
            scene = get_scene_context()
            scene_desc = scene.get("description", "")[:80] if scene else ""
            greeting = think(f"[系统提示：摄像头检测到有人出现。场景：{scene_desc}。用1句话自然打招呼。]")
            if greeting:
                self._enqueue_speak(greeting)

        threading.Thread(target=_gen_greeting, daemon=True).start()
        return True

    def _check_scene_change(self) -> bool:
        """检测场景重大变化，主动播报。返回是否触发了播报。"""
        scene = get_scene_context()
        if not scene:
            return False

        scene_ts = scene.get("ts", "")
        if scene_ts == self._last_scene_ts:
            return False

        old_desc = self._last_scene_desc
        new_desc = scene.get("description", "")
        self._last_scene_ts = scene_ts
        self._last_scene_desc = new_desc

        # 只在描述差异很大时播报（避免重复说"有个人坐在桌前"）
        if old_desc and new_desc and old_desc != new_desc:
            # 简单判断：如果新描述和旧描述的前 20 字不同，认为有变化
            if old_desc[:20] != new_desc[:20]:
                post_event("scene", f"场景变化：{new_desc[:100]}", source="interact")
                # 不每次都说，只记录事件
        return False

    def _time_aware_greeting(self) -> None:
        """基于时间的智能问候：本地秒回 + 云端补充"""
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 6:
            period = "凌晨了还在忙"
        elif hour < 9:
            period = "早上好"
        elif hour < 12:
            period = "上午好"
        elif hour < 14:
            period = "中午好"
        elif hour < 18:
            period = "下午好"
        elif hour < 22:
            period = "晚上好"
        else:
            period = "夜深了"

        # 第一句：本地固定短句，秒出（不等网络）
        self._enqueue_speak(f"川哥{period}，贾维斯在线。")

        # 第二句：Gemini 结合视觉场景生成（异步，自然衔接）
        def _gen_scene_comment():
            scene = get_scene_context()
            scene_desc = scene.get("description", "")[:80] if scene else ""
            if scene_desc:
                comment = think(f"[系统提示：你刚启动，看到场景：{scene_desc}。用1句话自然描述你看到了什么，像朋友随口说的。]")
                if comment:
                    self._enqueue_speak(comment)

        threading.Thread(target=_gen_scene_comment, daemon=True).start()

    def _enqueue_speak(self, text: str) -> None:
        """把要说的话放入队列（线程安全）"""
        self.speak_queue.put(text)

    def _speak_loop(self) -> None:
        """语音输出线程：从队列取出文本，依次播放"""
        while not self.stop_event.is_set():
            try:
                text = self.speak_queue.get(timeout=SPEAK_QUEUE_TIMEOUT)
                print(f"[cc-interact] 🗣️ {text}")
                say(text)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[cc-interact] 语音输出异常: {e}")


def main() -> None:
    """启动多模态交互"""
    engine = CcInteract()
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\n[cc-interact] 贾维斯下线")


if __name__ == "__main__":
    main()
