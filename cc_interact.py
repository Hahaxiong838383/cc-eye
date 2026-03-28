"""
cc_interact.py — cc 贾维斯全双工多模态交互引擎

核心改变：麦克风永远不关，播放器可随时停，AEC 实时消除回声。

架构：
  AudioEngine — 全双工音频引擎
    常驻麦克风 InputStream（callback 模式，32ms/帧，不阻塞）
    SileroVAD + SpeechSegmenter — 实时人声检测与语音段切分
    InterruptablePlayer — 可打断 TTS 流式播放（<50ms 打断）
    EchoCanceller — 播放时实时消除回声
    StateMachine — 状态驱动（IDLE/LISTENING/PROCESSING/SPEAKING/INTERRUPTED）
    barge-in — 播放中连续 3 帧人声 → 打断（~96ms 防抖）

  处理线程 — 从 queue 取语音段，做 STT → LLM → TTS 管道
  视觉线程 — 复用现有逻辑：人到达、场景变化、时间感知问候

用法：
  cd ~/mycc/2-Projects/cc-eye && source .venv/bin/activate
  python cc_interact.py
"""

import asyncio
import logging
import queue
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from cc_aec import EchoCanceller
from cc_brain import think, think_stream
from cc_context import (
    CC_IDENTITY,
    build_system_prompt,
    get_camera_events,
    get_scene_context,
)
from cc_events import (
    post_event,
    seconds_since_last_interaction,
    sync_from_daemon_events,
)
from cc_listen import SpeechSegment, _transcribe
from cc_player import InterruptablePlayer, tts_to_pcm_stream, _mp3_bytes_to_pcm
from cc_state import Event, State, StateMachine
from cc_vad import SileroVAD, SpeechSegmenter
from cc_voice import is_echo
from state_fusion import inject_voice_emotion

# ── 日志 ──
logger = logging.getLogger("cc-interact")

# ── 配置 ──
SAMPLE_RATE = 16000           # 麦克风采样率
CHANNELS = 1
WINDOW_SIZE = 512             # Silero VAD 帧大小（32ms @ 16kHz）
AUDIO_PATH = "/tmp/cc-listen-segment.wav"

# 视觉监控
SCENE_FILE = Path("/tmp/cc-eye-scene.json")
EVENTS_FILE = Path("/tmp/cc-eye-events.jsonl")
PERSON_FLAG = Path("/tmp/cc-eye-person-arrived.flag")
VISION_CHECK_INTERVAL = 5.0   # 视觉事件检查间隔（秒）

# TTS 配置
VOICE = "zh-CN-YunjianNeural"
RATE = "-5%"
PITCH = "-10Hz"

# barge-in 防抖：连续多少帧人声才算真正打断
BARGE_IN_FRAMES_REQUIRED = 3  # 3 帧 × 32ms ≈ 96ms

# 唤醒词
WAKE_WORDS = ["贾维斯", "jarvis", "嘉维斯", "贾维丝"]

# 主动交互
PROACTIVE_COOLDOWN = 30.0     # 主动说话冷却（秒）
SILENCE_MUTE_AFTER = 300.0    # 5 分钟无互动后停止主动说话


class AudioEngine:
    """全双工音频引擎 — 麦克风永远开，播放器可随时停"""

    def __init__(self):
        # ── 核心组件 ──
        self.vad = SileroVAD(threshold=0.5, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE)
        self.segmenter = SpeechSegmenter(
            on_speech_end=self._on_speech_segment_from_audio_thread,
        )
        self.state_machine = StateMachine(on_transition=self._on_state_change)
        self.player = InterruptablePlayer(
            on_play_start=self._on_play_start,
            on_play_stop=self._on_play_stop,
            on_pcm_frame=self._on_player_pcm_frame,
        )
        self.aec = EchoCanceller(filter_length=4096, mu=0.05, tail_duration=0.5)

        # ── 常驻麦克风 InputStream（callback 模式，不阻塞）──
        self.mic_stream: Optional[sd.InputStream] = None

        # ── 处理队列（音频线程 → 处理线程）──
        self._segment_queue: queue.Queue = queue.Queue()

        # ── barge-in 防抖计数器 ──
        self._barge_in_count = 0

        # ── 停止信号 ──
        self.stop_event = threading.Event()

        # ── 视觉监控状态 ──
        self._last_person_flag_ts = ""
        self._last_scene_ts = ""
        self._last_scene_desc = ""
        self._greeted = False
        self._person_greeted = False
        self._start_time = time.time()

        # ── 主动播报队列（视觉线程 → 播放）──
        self._proactive_queue: queue.Queue = queue.Queue()

        # ── TTS 回声过滤：记录最近播放的文本 ──
        self._recent_tts_texts: list = []
        self._tts_lock = threading.Lock()

    # ════════════════════════════════════════════
    #  启动 / 停止
    # ════════════════════════════════════════════

    def start(self) -> None:
        """启动引擎：开麦克风 + 视觉监控 + 处理线程"""
        print("=" * 50)
        print("  cc 贾维斯 — 全双工多模态交互")
        print("  耳：常驻麦克风（Silero VAD + AEC）")
        print("  嘴：可打断播放器（InterruptablePlayer）")
        print("  眼：摄像头场景感知（camera_daemon）")
        print("  Ctrl+C 退出")
        print("=" * 50)

        # 启动常驻麦克风
        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=WINDOW_SIZE,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.mic_stream.start()
        print("[cc-interact] 常驻麦克风已启动（Silero VAD + AEC）")

        # 启动处理线程（STT → LLM → TTS 管道）
        process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="process-pipeline",
        )
        process_thread.start()

        # 启动主动播报线程
        proactive_thread = threading.Thread(
            target=self._proactive_speak_loop,
            daemon=True,
            name="proactive-speaker",
        )
        proactive_thread.start()

        # 启动视觉事件监控线程
        vision_thread = threading.Thread(
            target=self._vision_monitor_loop,
            daemon=True,
            name="vision-monitor",
        )
        vision_thread.start()

        # 发布启动事件
        post_event("system", "贾维斯全双工交互启动", source="interact")
        print("[cc-interact] 全双工引擎就绪，说「贾维斯」唤醒...")

        # 主线程：等待停止信号
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            print("\n[cc-interact] 用户中断")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """关闭引擎"""
        self.stop_event.set()
        self.player.stop()
        if self.mic_stream is not None:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
        print("[cc-interact] 贾维斯下线")

    # ════════════════════════════════════════════
    #  音频回调（sounddevice 音频线程，每 32ms 调用）
    # ════════════════════════════════════════════

    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice 回调，每 32ms 自动调用。

        职责：
        1. AEC：播放中消除回声
        2. 喂给 VAD segmenter
        3. barge-in 检测（播放中检测到人声 → 打断）
        """
        if status:
            logger.debug("sounddevice status: %s", status)

        chunk = indata[:, 0].copy()  # copy 避免 sounddevice 回收

        # AEC：如果正在播放，消除回声
        if self.player.is_playing or self.aec.is_active:
            chunk = self.aec.process(chunk)

        # barge-in 检测：播放中检测到人声 → 打断
        if self.state_machine.current_state == State.SPEAKING:
            if self.vad.is_speech(chunk):
                self._barge_in_count += 1
                if self._barge_in_count >= BARGE_IN_FRAMES_REQUIRED:
                    # 连续 3 帧人声，确认打断
                    self.player.stop()
                    self.state_machine.transition(Event.BARGE_IN)
                    self._barge_in_count = 0
                    self.segmenter.reset()  # 重置 segmenter，准备接收新语音
                    logger.info("barge-in 触发：用户打断播放")
            else:
                self._barge_in_count = 0
            # 播放中不喂 segmenter，避免录到自己的声音
            return

        # 非播放状态：重置 barge-in 计数
        self._barge_in_count = 0

        # 喂给 VAD segmenter（segmenter 内部的回调会在检测到完整语音段时触发）
        self.segmenter.feed(chunk)

    # ════════════════════════════════════════════
    #  语音段回调（从 segmenter 触发）
    # ════════════════════════════════════════════

    def _on_speech_segment_from_audio_thread(self, audio: np.ndarray):
        """
        SpeechSegmenter 检测到完整语音段时回调。

        注意：此回调在音频线程中执行，不能做耗时操作！
        只负责把音频数据放入队列，由处理线程异步处理。
        """
        # 状态转移：通知检测到语音结束
        # SPEECH_START 在 segmenter 内部隐含（开始录音时），这里处理 SPEECH_END
        current = self.state_machine.current_state
        if current == State.IDLE:
            # 先转到 LISTENING 再转到 PROCESSING
            self.state_machine.transition(Event.SPEECH_START)
        self.state_machine.transition(Event.SPEECH_END)

        # 放入处理队列（不阻塞音频线程）
        try:
            self._segment_queue.put_nowait(audio.copy())
        except queue.Full:
            logger.warning("处理队列已满，丢弃语音段")

    # ════════════════════════════════════════════
    #  处理线程（STT → LLM → TTS 管道）
    # ════════════════════════════════════════════

    def _process_loop(self) -> None:
        """处理线程：从 queue 取语音段，做 STT → 唤醒词 → LLM → TTS"""
        logger.info("处理管道线程已启动")

        while not self.stop_event.is_set():
            try:
                audio = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._handle_segment(audio)
            except Exception as e:
                logger.error("处理语音段异常: %s", e, exc_info=True)
                self.state_machine.transition(Event.ERROR)

    def _handle_segment(self, audio: np.ndarray) -> None:
        """处理一个完整语音段：STT → 唤醒词 → LLM → TTS"""

        # 保存音频 → SenseVoice STT
        sf.write(AUDIO_PATH, audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        segment = _transcribe(AUDIO_PATH, duration)

        if not segment or not segment.text:
            self.state_machine.transition(Event.ERROR)
            return

        text = segment.text.strip()
        if not text:
            self.state_machine.transition(Event.ERROR)
            return

        # 文本层回声过滤
        if is_echo(text):
            print(f"[cc-interact] 回声过滤: {text[:30]}...")
            self.state_machine.transition(Event.TIMEOUT)
            return

        # 也检查 _recent_tts_texts（自己维护的记录）
        if self._is_own_echo(text):
            print(f"[cc-interact] 回声过滤(local): {text[:30]}...")
            self.state_machine.transition(Event.TIMEOUT)
            return

        # 唤醒词检查
        wake_hit, clean_text = self._check_wake_word(text)
        if not wake_hit:
            self.state_machine.transition(Event.TIMEOUT)
            return

        print(f"\n[cc-interact] 川哥说: {clean_text}")

        # 注入语音情感
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

        # 处理语音指令
        self._process_speech(clean_text, segment)

    def _check_wake_word(self, text: str) -> tuple:
        """
        检查唤醒词。

        Returns:
            (wake_hit: bool, clean_text: str)
        """
        text_lower = text.lower()
        for w in WAKE_WORDS:
            if w in text_lower:
                clean = text_lower.replace(w, "").strip()
                return True, clean
        return False, text

    def _is_own_echo(self, text: str, threshold: float = 0.5) -> bool:
        """检查是否是自己播放的 TTS 回声"""
        if not text or len(text) < 3:
            return False
        with self._tts_lock:
            recent = list(self._recent_tts_texts)
        for tts_text in recent:
            common = sum(1 for c in text if c in tts_text)
            ratio = common / len(text)
            if ratio > threshold:
                return True
        return False

    def _record_tts_text(self, text: str) -> None:
        """记录 TTS 播放的文本（用于回声过滤）"""
        with self._tts_lock:
            self._recent_tts_texts.append(text)
            if len(self._recent_tts_texts) > 10:
                self._recent_tts_texts.pop(0)

    def _process_speech(self, text: str, segment: SpeechSegment) -> None:
        """处理语音：LLM 推理 → 流式 TTS 播放"""

        first_audio_sent = False

        for sentence in think_stream(text):
            # 如果在处理过程中被打断，停止后续处理
            if self.state_machine.current_state == State.INTERRUPTED:
                logger.info("处理中被打断，停止后续 TTS")
                break
            if self.state_machine.current_state == State.IDLE:
                # 可能 barge-in 导致状态已回到 IDLE
                break

            print(f"[cc-interact] 回复: {sentence}")

            # 记录 TTS 文本用于回声过滤
            self._record_tts_text(sentence)

            # 用 edge-tts 合成 → InterruptablePlayer 播放
            try:
                self._play_sentence(sentence)

                # 第一句播放开始时，转换状态
                if not first_audio_sent:
                    self.state_machine.transition(Event.FIRST_AUDIO)
                    first_audio_sent = True

                # 等当前句播完再播下一句（除非被打断）
                self.player.wait()

            except Exception as e:
                logger.error("TTS 播放异常: %s", e)

        # 所有句子播完，状态回到 IDLE
        if self.state_machine.current_state == State.SPEAKING:
            self.state_machine.transition(Event.PLAY_DONE)
        elif self.state_machine.current_state == State.PROCESSING:
            # 如果没有任何音频产出（think_stream 没 yield 或全被跳过），回到 IDLE
            self.state_machine.transition(Event.ERROR)

    def _play_sentence(self, text: str) -> None:
        """
        合成一句话并通过 InterruptablePlayer 播放。

        使用 edge-tts 合成 mp3 → 解码为 PCM → 播放。
        """
        # 收集 edge-tts mp3 chunks
        mp3_chunks = []

        async def _collect():
            async for chunk_data in tts_to_pcm_stream(text, VOICE, RATE, PITCH):
                mp3_chunks.append(chunk_data)

        # 在新的 event loop 中运行（处理线程没有 event loop）
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_collect())
            loop.close()
        except Exception as e:
            logger.error("edge-tts 合成失败: %s", e)
            return

        if not mp3_chunks:
            return

        # 拼接 mp3 → 解码 PCM
        mp3_all = b"".join(mp3_chunks)
        try:
            pcm = _mp3_bytes_to_pcm(mp3_all, 24000)
        except Exception as e:
            logger.error("MP3 解码失败: %s", e)
            return

        # 通知 AEC 开始播放
        self.aec.start_playback()

        # 播放 PCM
        self.player.play_pcm(pcm, sample_rate=24000)

    # ════════════════════════════════════════════
    #  播放器回调（AEC 联动）
    # ════════════════════════════════════════════

    def _on_play_start(self) -> None:
        """播放开始回调"""
        logger.debug("播放开始")

    def _on_play_stop(self) -> None:
        """播放停止回调"""
        self.aec.stop_playback()
        logger.debug("播放停止")

    def _on_player_pcm_frame(self, frame: np.ndarray) -> None:
        """
        播放器每帧 PCM 回调 — 喂给 AEC 作为参考信号。

        注意：播放器在 24kHz，AEC 在 16kHz。这里做简单的最近邻降采样。
        """
        # 简单降采样 24kHz → 16kHz（取 2/3 的样本）
        if len(frame) > 0:
            ratio = 16000 / 24000
            indices = np.arange(0, len(frame), 1 / ratio).astype(int)
            indices = indices[indices < len(frame)]
            resampled = frame[indices]
            # 喂给 AEC 参考缓冲
            # AEC 的 process() 会在 _audio_callback 中使用这些参考信号
            # 这里不需要额外操作，AEC.start_playback() 已经标记了状态

    # ════════════════════════════════════════════
    #  状态机回调
    # ════════════════════════════════════════════

    def _on_state_change(self, prev: State, event: Event, target: State) -> None:
        """状态转移通知"""
        print(f"[state] {prev.name} + {event.name} → {target.name}")

    # ════════════════════════════════════════════
    #  主动播报线程
    # ════════════════════════════════════════════

    def _proactive_speak_loop(self) -> None:
        """主动播报线程：从队列取文本，通过 InterruptablePlayer 播放"""
        while not self.stop_event.is_set():
            try:
                text = self._proactive_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 只在 IDLE 状态才播报（避免打断正在进行的对话）
            if self.state_machine.current_state != State.IDLE:
                continue

            try:
                print(f"[cc-interact] 主动播报: {text}")
                self._record_tts_text(text)
                self._play_sentence(text)
                self.player.wait()
            except Exception as e:
                logger.error("主动播报异常: %s", e)

    def _enqueue_speak(self, text: str) -> None:
        """把要说的话放入主动播报队列（线程安全）"""
        try:
            self._proactive_queue.put_nowait(text)
        except queue.Full:
            logger.warning("主动播报队列已满，丢弃: %s", text[:20])

    # ════════════════════════════════════════════
    #  视觉监控线程（复用现有逻辑）
    # ════════════════════════════════════════════

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
                is_first_run = not self._greeted
                can_proactive = (
                    (now - last_proactive_time) > PROACTIVE_COOLDOWN
                    and (is_first_run or idle_seconds < SILENCE_MUTE_AFTER)
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
                logger.error("视觉监控异常: %s", e)

            self.stop_event.wait(VISION_CHECK_INTERVAL)

    def _check_person_arrived(self) -> bool:
        """检测是否有人到达，主动打招呼"""
        if not PERSON_FLAG.exists():
            self._person_greeted = False
            return False

        flag_ts = PERSON_FLAG.read_text().strip()
        if flag_ts == self._last_person_flag_ts:
            return False

        self._last_person_flag_ts = flag_ts

        if self._person_greeted:
            return False

        self._person_greeted = True
        post_event("face", "检测到有人到达", source="interact")

        # 本地秒回
        self._enqueue_speak("有人来了。")

        # 异步补充问候
        def _gen_greeting():
            scene = get_scene_context()
            scene_desc = scene.get("description", "")[:80] if scene else ""
            greeting = think(
                f"[系统提示：摄像头检测到有人出现。场景：{scene_desc}。用1句话自然打招呼。]"
            )
            if greeting:
                self._enqueue_speak(greeting)

        threading.Thread(target=_gen_greeting, daemon=True).start()
        return True

    def _check_scene_change(self) -> bool:
        """检测场景重大变化，主动播报"""
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

        if old_desc and new_desc and old_desc != new_desc:
            if old_desc[:20] != new_desc[:20]:
                post_event("scene", f"场景变化：{new_desc[:100]}", source="interact")
        return False

    def _time_aware_greeting(self) -> None:
        """基于时间的智能问候：本地秒回 + 云端补充"""
        hour = datetime.now().hour

        if hour < 6:
            greetings = ["这么晚还在忙？", "凌晨了，注意休息。", "夜猫子模式？"]
        elif hour < 9:
            greetings = ["早。", "起了？", "早上好。"]
        elif hour < 12:
            greetings = ["在的。", "上午好。", "嗯，随时待命。"]
        elif hour < 14:
            greetings = ["中午好。", "吃了吗？", "午休时间到了。"]
        elif hour < 18:
            greetings = ["下午好。", "在的。", "继续？"]
        elif hour < 22:
            greetings = ["晚上好。", "在。", "还在忙？"]
        else:
            greetings = ["夜深了。", "还没休息？", "在的。"]

        self._enqueue_speak(random.choice(greetings))

        # 第二句：结合视觉场景生成
        def _gen_scene_comment():
            scene = get_scene_context()
            scene_desc = scene.get("description", "")[:80] if scene else ""
            if scene_desc:
                comment = think(
                    f"[系统提示：你刚启动，看到场景：{scene_desc}。用1句话自然描述你看到了什么，像朋友随口说的。]"
                )
                if comment:
                    self._enqueue_speak(comment)

        threading.Thread(target=_gen_scene_comment, daemon=True).start()


def main() -> None:
    """启动全双工多模态交互"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    engine = AudioEngine()
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\n[cc-interact] 贾维斯下线")


if __name__ == "__main__":
    main()
