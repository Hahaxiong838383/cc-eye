"""
cc_jarvis_v3.py — 贾维斯 V3 全双工语音交互

重写版：解决 V2 的打断/破音/唤醒词问题。

架构：
  Swift VP AEC → 常驻麦克风
  → openwakeword（音频级唤醒）+ 能量 VAD（语音切分）
  → Qwen3-ASR MLX（STT 130ms）
  → 三级路由（oMLX 本地 / MiniMax 云端）
  → Qwen3-TTS MLX → 常驻 OutputStream（零 click，可打断）
"""

import logging
import numpy as np
import soundfile as sf
import threading
import time
import queue
from datetime import datetime
from pathlib import Path

from cc_audio_engine import AudioBridge
from cc_audio_out import AudioPlayer
from cc_tts_local import local_tts_to_pcm, preload as preload_tts
from cc_stt_mlx import transcribe as mlx_transcribe, preload as preload_stt
from cc_brain import think_stream
from cc_events import post_event

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("jarvis")

# ── 配置 ──
STT_SAMPLE_RATE = 16000
VAD_WINDOW = 512              # 32ms @ 16kHz
ENERGY_THRESHOLD = 0.008
MIN_SPEECH_MS = 300
MIN_SILENCE_MS = 400
BARGE_IN_ENERGY = 0.05        # 提高门槛，避免 TTS 回声误触发
BARGE_IN_FRAMES = 6           # 需要更持续的语音才触发
ACTIVE_SESSION_TIMEOUT = 120
AUDIO_PATH = "/tmp/cc-jarvis-segment.wav"

# 唤醒词
WAKE_WORDS = ["贾维斯", "jarvis", "嘉维斯", "贾维丝",
              "小维斯", "蒋维斯", "夏维斯", "茶维斯", "加维斯",
              "假维斯", "甲维斯", "家维斯", "维斯"]

# 安静模式触发词
QUIET_WORDS = ["安静", "别说了", "闭嘴", "不要说话", "听着就行",
               "你先听", "别吵", "静一下", "停", "够了"]

# 恢复说话触发词
RESUME_WORDS = ["说吧", "你说", "继续说", "可以说了", "说话"]

# 智能过渡语气词库（长句等云端时用，按长度分级，随机不重复）
TRANSITION_PHRASES = {
    "short": ["嗯。", "好。", "行。", "嗯嗯。"],
    "medium": ["让我想想。", "这个嘛。", "好问题。", "有点意思。", "稍等。"],
    "long": ["这个方向挺好的。", "我梳理一下思路。", "让我整理下信息。", "这个值得聊聊。"],
}
_last_transitions: list = []  # 避免重复


class JarvisV3:
    def __init__(self):
        self.bridge = AudioBridge()
        self.player = AudioPlayer(sample_rate=24000)

        # 唤醒词检测（音频级）
        self._oww = None
        try:
            import openwakeword
            from openwakeword.model import Model as OWWModel
            self._oww = OWWModel(wakeword_models=["hey_jarvis_v0.1"])
            logger.info("openwakeword 就绪 (hey_jarvis)")
        except Exception as e:
            logger.warning(f"openwakeword 加载失败: {e}，使用文本匹配")

        # 状态
        self._state = "IDLE"
        self._state_lock = threading.Lock()
        self._last_wake_time: float = 0
        self._responding = False
        self._last_play_time: float = 0

        # VAD
        self._speech_frames: list = []
        self._speech_ms: float = 0
        self._silence_ms: float = 0
        self._is_speech = False
        self._resample_buf = np.array([], dtype=np.float32)

        # 处理队列
        self._segment_queue = queue.Queue()

        # 回声过滤
        self._recent_tts: list = []

        # MLX 锁
        self._mlx_lock = threading.Lock()

        # 停止
        self._stop = threading.Event()

    @property
    def state(self):
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, v):
        with self._state_lock:
            old = self._state
            self._state = v
        if old != v:
            logger.info(f"[state] {old} → {v}")

    # ════════════════════════════════════════
    #  启动
    # ════════════════════════════════════════

    def start(self):
        print("=" * 50)
        print("  贾维斯 V3")
        print("  AEC: Swift VP | STT: Qwen3-ASR MLX")
        print("  TTS: Qwen3-TTS MLX | LLM: oMLX + MiniMax")
        print("  打断: 常驻 OutputStream + barge-in")
        print("  Ctrl+C 退出")
        print("=" * 50)

        # 1. 加载模型（顺序，避免 MLX 冲突）
        print("  加载模型...")
        preload_tts()
        preload_stt()
        logger.info("模型就绪")

        # 2. 启动常驻音频输出
        self.player.start()

        # 3. 启动音频桥接
        if not self.bridge.start(on_mic_chunk=self._on_mic_chunk):
            logger.error("音频桥接失败")
            return

        # 4. 处理线程
        threading.Thread(target=self._process_loop, daemon=True).start()

        logger.info("就绪，说「贾维斯」唤醒")
        self._greet()

        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        except KeyboardInterrupt:
            print("\n贾维斯下线")
        finally:
            self.player.stop()
            self.bridge.stop()

    def _greet(self):
        hour = datetime.now().hour
        greetings = {
            range(0, 6): "夜深了。",
            range(6, 9): "早。",
            range(9, 12): "上午好。",
            range(12, 14): "中午好。",
            range(14, 18): "下午好。",
            range(18, 22): "晚上好。",
            range(22, 24): "还在忙？",
        }
        for r, g in greetings.items():
            if hour in r:
                self._speak_single(g)
                break

    # ════════════════════════════════════════
    #  麦克风回调
    # ════════════════════════════════════════

    def _on_mic_chunk(self, chunk: np.ndarray):
        """48kHz → 16kHz → VAD + 唤醒词检测"""
        # 降采样 48kHz → 16kHz
        self._resample_buf = np.concatenate([self._resample_buf, chunk])
        ratio = 3
        usable = (len(self._resample_buf) // ratio) * ratio
        if usable == 0:
            return
        resampled = self._resample_buf[:usable:ratio]
        self._resample_buf = self._resample_buf[usable:]

        # openwakeword 检测（每 1280 样本 = 80ms @ 16kHz）
        if self._oww and not self._responding:
            # 喂入 oww
            prediction = self._oww.predict(resampled)
            for mdl, score in prediction.items():
                if score > 0.5:
                    logger.info(f"[wake] 唤醒词检测: {mdl}={score:.2f}")
                    self._last_wake_time = time.time()
                    self._oww.reset()

        # 按帧处理 VAD
        for i in range(0, len(resampled) - VAD_WINDOW + 1, VAD_WINDOW):
            frame = resampled[i:i + VAD_WINDOW]
            self._process_vad_frame(frame)

    def _process_vad_frame(self, frame: np.ndarray):
        energy = float(np.abs(frame).mean())

        # 回复中：纯能量打断（不跑 STT，避免锁冲突和回声碎片）
        if self._responding:
            playback_energy = self.player.current_playback_energy
            # 动态阈值：播放能量 × 回声系数 + 安全余量
            dynamic_threshold = max(BARGE_IN_ENERGY, playback_energy * 0.3 + 0.03)

            if energy >= dynamic_threshold:
                if not hasattr(self, '_bi_count'):
                    self._bi_count = 0
                self._bi_count += 1
                if self._bi_count >= BARGE_IN_FRAMES:
                    logger.info(f"barge-in! energy={energy:.3f} threshold={dynamic_threshold:.3f}")
                    self.player.interrupt()
                    self._responding = False
                    self._bi_count = 0
                    self.state = "IDLE"
                    self._reset_vad()
            else:
                self._bi_count = 0
            return

        # 回复中 或 刚播完冷却期：不收集普通语音段
        if self._responding:
            return
        if (time.time() - self._last_play_time) < 0.8:
            return

        # 正常 VAD
        frame_ms = VAD_WINDOW / STT_SAMPLE_RATE * 1000
        is_speech = energy >= ENERGY_THRESHOLD

        if not self._is_speech:
            if is_speech:
                self._is_speech = True
                self._speech_frames = [frame.copy()]
                self._speech_ms = frame_ms
                self._silence_ms = 0
                self.state = "LISTENING"
        else:
            self._speech_frames.append(frame.copy())
            if is_speech:
                self._speech_ms += frame_ms
                self._silence_ms = 0
            else:
                self._silence_ms += frame_ms
                if self._silence_ms >= MIN_SILENCE_MS:
                    if self._speech_ms >= MIN_SPEECH_MS:
                        audio = np.concatenate(self._speech_frames)
                        self._segment_queue.put(audio)
                    self._reset_vad()

    # 不打断的语气词/感叹词
    _FILLER_WORDS = {"嗯", "啊", "哦", "呃", "唔", "嗯嗯", "啊啊", "哈", "哈哈",
                     "嘿", "喂", "噢", "哎", "呀", "吧", "了", "的", "吗"}

    def _check_barge_in_content(self, audio: np.ndarray):
        """STT 判断 barge-in 内容：真话打断，语气词忽略"""
        try:
            sf.write("/tmp/cc-barge-in.wav", audio, STT_SAMPLE_RATE)
            with self._mlx_lock:
                result = mlx_transcribe("/tmp/cc-barge-in.wav", len(audio) / STT_SAMPLE_RATE)
            if not result or not result.text:
                return

            text = result.text.strip().strip("。？！，、；：.!? ")
            if not text:
                return

            logger.info(f"[barge-in STT] {text}")

            # 唤醒词 → 立即打断
            for w in WAKE_WORDS:
                if w in text.lower():
                    logger.info(f"barge-in: 唤醒词 '{w}'")
                    self.player.interrupt()
                    self._responding = False
                    self._last_wake_time = time.time()
                    self.state = "IDLE"
                    self._reset_vad()
                    return

            # 语气词/感叹词 → 不打断
            if text in self._FILLER_WORDS or len(text) <= 1:
                return

            # 文本级回声过滤：和最近 TTS 内容对比
            for tts_text in self._recent_tts[-8:]:
                tts_clean = tts_text.strip("。？！，、；：.!? ")
                if not tts_clean or len(tts_clean) < 2:
                    continue
                sim = self._similarity(text, tts_clean)
                if sim >= 0.4:
                    logger.info(f"[barge-in] 回声过滤 sim={sim:.2f}: '{text}' ≈ TTS")
                    return

            # 唤醒词 → 立即打断 + 处理
            has_wake = any(w in text.lower() for w in WAKE_WORDS)
            if has_wake:
                logger.info(f"barge-in: 唤醒词 '{text}'")
                self.player.interrupt()
                self._responding = False
                self._last_wake_time = time.time()
                self.state = "IDLE"
                self._reset_vad()
                self._segment_queue.put(audio)
                return

            # 有实质内容（非语气词/非回声）→ 打断
            if len(text) >= 3:
                logger.info(f"barge-in: 内容打断 '{text}'")
                self.player.interrupt()
                self._responding = False
                self.state = "IDLE"
                self._reset_vad()
                self._segment_queue.put(audio)

        except Exception as e:
            logger.error(f"barge-in STT 异常: {e}")

    def _reset_vad(self):
        self._is_speech = False
        self._speech_frames.clear()
        self._speech_ms = 0
        self._silence_ms = 0

    # ════════════════════════════════════════
    #  处理循环
    # ════════════════════════════════════════

    def _process_loop(self):
        while not self._stop.is_set():
            try:
                audio = self._segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle_segment(audio)
            except Exception as e:
                logger.error(f"处理异常: {e}", exc_info=True)
                self.state = "IDLE"

    def _handle_segment(self, audio: np.ndarray):
        self.state = "PROCESSING"

        # 能量门槛
        energy = float(np.abs(audio).mean())
        if energy < 0.005:
            self.state = "IDLE"
            return

        # STT
        t0 = time.time()
        sf.write(AUDIO_PATH, audio, STT_SAMPLE_RATE)
        with self._mlx_lock:
            segment = mlx_transcribe(AUDIO_PATH, len(audio) / STT_SAMPLE_RATE)
        stt_ms = (time.time() - t0) * 1000

        if not segment or not segment.text or not segment.text.strip():
            self.state = "IDLE"
            return

        text = segment.text.strip()

        # 幻觉过滤
        unique = set(text.replace("。", "").replace("，", "").replace(" ", ""))
        if len(unique) <= 1 and len(text) > 2:
            self.state = "IDLE"
            return

        logger.info(f"[STT] {stt_ms:.0f}ms: {text}")

        # 唤醒词检查
        has_wake, clean_text = self._check_wake(text)
        if not has_wake:
            self.state = "IDLE"
            return

        if not clean_text:
            self.state = "IDLE"
            return

        # 回声过滤
        if self._is_echo(clean_text):
            self.state = "IDLE"
            return

        logger.info(f"[用户] {clean_text}")

        # 安静模式检测
        if self.state == "QUIET":
            # 安静模式下只检查恢复词
            for w in RESUME_WORDS:
                if w in clean_text:
                    self.state = "IDLE"
                    self._speak_single("好，我在。")
                    return
            for w in WAKE_WORDS:
                if w in clean_text.lower():
                    self.state = "IDLE"
                    self._speak_single("在。")
                    return
            # 静默记录，不回复
            logger.info(f"[quiet] 记录: {clean_text}")
            post_event("speech", f"[静默记录] {clean_text}", source="jarvis")
            self.state = "QUIET"
            return

        # 检查是否要进入安静模式
        for w in QUIET_WORDS:
            if w in clean_text:
                logger.info(f"[quiet] 进入安静模式: '{w}'")
                self._speak_single("好的，我听着。")
                self.state = "QUIET"
                return

        # LLM → TTS（预合成管道）
        self._respond(clean_text)

    # 唤醒应答词库（从预缓存中随机选）
    _WAKE_RESPONSES = [
        "在。", "我在。", "说吧。", "在的。", "随时待命。",
        "怎么了。", "好，我在。", "需要我做什么？",
    ]
    _last_wake_response = ""

    def _check_wake(self, text: str) -> tuple:
        import random
        text_lower = text.lower()

        # 文本匹配
        for w in WAKE_WORDS:
            if w in text_lower:
                clean = text_lower.replace(w, "").strip("。？！，、；：. ")
                self._last_wake_time = time.time()
                if not clean:
                    # 随机应答，避免重复
                    available = [r for r in self._WAKE_RESPONSES if r != self._last_wake_response]
                    resp = random.choice(available)
                    self._last_wake_response = resp
                    self._speak_single(resp)
                    return False, ""
                return True, clean

        # 活跃会话
        if (time.time() - self._last_wake_time) < ACTIVE_SESSION_TIMEOUT:
            return True, text

        return False, ""

    def _is_echo(self, text: str) -> bool:
        if not text or len(text) < 2:
            return False
        clean = text.strip("。？！，、；：.!? ")
        if not clean:
            return True

        time_since = time.time() - self._last_play_time
        if time_since < 3:
            threshold = 0.35
        elif time_since < 8:
            threshold = 0.5
        elif time_since < 15:
            threshold = 0.7
        else:
            return False

        for tts in self._recent_tts[-8:]:
            tts_clean = tts.strip("。？！，、；：.!? ")
            if not tts_clean or len(tts_clean) < 3:
                continue  # TTS 太短（如"在"），不做回声匹配
            if len(clean) > len(tts_clean) * 3:
                continue  # 用户说的远比 TTS 长，不可能是回声
            sim = self._similarity(clean, tts_clean)
            if sim >= threshold:
                logger.info(f"[echo] sim={sim:.2f}: '{clean[:20]}' ≈ '{tts_clean[:20]}'")
                return True
        return False

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) <= 3:
            return sum(1 for c in shorter if c in longer) / len(shorter)
        max_match = 0
        for i in range(len(shorter)):
            for j in range(i + 2, min(i + len(longer) + 1, len(shorter) + 1)):
                if shorter[i:j] in longer:
                    max_match = max(max_match, j - i)
        return max_match / len(shorter)

    # ════════════════════════════════════════
    #  回复（预合成管道）
    # ════════════════════════════════════════

    def _respond(self, text: str):
        self._responding = True
        self.state = "SPEAKING"

        synth_queue = queue.Queue()

        # 后台合成线程
        def _synth():
            import re as _re
            import random
            for sentence in think_stream(text):
                if not self._responding:
                    break

                # 过渡标记 → 智能选择语气词
                if sentence == "__TRANSITION__":
                    phrase = self._pick_transition(text)
                    logger.info(f"[过渡] {phrase}")
                    t0 = time.time()
                    with self._mlx_lock:
                        if not self._responding:
                            break
                        pcm, sr = local_tts_to_pcm(phrase)
                    logger.info(f"[TTS] {(time.time()-t0)*1000:.0f}ms | {phrase}")
                    synth_queue.put((pcm, sr, phrase))
                    continue

                # 清理 markdown
                sentence = _re.sub(r'\*\*(.+?)\*\*', r'\1', sentence)
                sentence = _re.sub(r'^[-*•]\s*', '', sentence.strip())
                sentence = _re.sub(r'#{1,3}\s*', '', sentence)
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not self._responding:
                    break  # 合成前再检查一次
                logger.info(f"[回复] {sentence}")
                t0 = time.time()
                with self._mlx_lock:
                    if not self._responding:
                        break  # 锁等待期间可能被打断
                    pcm, sr = local_tts_to_pcm(sentence)
                logger.info(f"[TTS] {(time.time()-t0)*1000:.0f}ms | {len(pcm)/sr:.1f}s")
                synth_queue.put((pcm, sr, sentence))
            synth_queue.put(None)

        threading.Thread(target=_synth, daemon=True).start()

        # 主线程：逐句播放
        while self._responding:
            try:
                item = synth_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            pcm, sr, sentence = item
            if not self._responding:
                break

            self._recent_tts.append(sentence)
            if len(self._recent_tts) > 10:
                self._recent_tts.pop(0)

            self.player.play(pcm, sr)
            while not self.player.wait(timeout=0.05):
                if not self._responding:
                    self.player.interrupt()
                    # 清空合成队列残留
                    while not synth_queue.empty():
                        try: synth_queue.get_nowait()
                        except: break
                    break
            self._last_play_time = time.time()

        self._responding = False
        self._reset_vad()
        time.sleep(0.15)
        self.state = "IDLE"

    def _pick_transition(self, user_text: str) -> str:
        """智能选择过渡语气词：根据用户输入长度分级，避免重复"""
        import random
        # 根据用户输入复杂度选分级
        if len(user_text) < 20:
            pool = TRANSITION_PHRASES["short"]
        elif len(user_text) < 40:
            pool = TRANSITION_PHRASES["medium"]
        else:
            pool = TRANSITION_PHRASES["long"]

        # 避免最近用过的
        available = [p for p in pool if p not in _last_transitions[-3:]]
        if not available:
            available = pool
        choice = random.choice(available)
        _last_transitions.append(choice)
        if len(_last_transitions) > 10:
            _last_transitions.pop(0)
        return choice

    def _speak_single(self, text: str):
        """播放单句（唤醒回复等）"""
        self._recent_tts.append(text)
        if len(self._recent_tts) > 10:
            self._recent_tts.pop(0)
        try:
            with self._mlx_lock:
                pcm, sr = local_tts_to_pcm(text)
        except Exception as e:
            logger.error(f"TTS: {e}")
            return
        self.player.play(pcm, sr)
        self.player.wait(timeout=10)
        self._last_play_time = time.time()
        self._reset_vad()
        time.sleep(0.15)


def main():
    JarvisV3().start()


if __name__ == "__main__":
    main()
