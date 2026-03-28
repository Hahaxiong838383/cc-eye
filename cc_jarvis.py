"""
cc_jarvis.py — 贾维斯 V2 全双工语音交互

架构：
  Swift cc_audio_bridge (VP AEC) ←→ Python AI 大脑

  音频输入：Swift VP → 管道 → VAD → STT → LLM → TTS → 管道 → Swift 播放
  全双工：播放时麦克风不关，VP 消回声，用户可打断

用法：
  cd ~/mycc/2-Projects/cc-eye
  .venv/bin/python cc_jarvis.py
"""

import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 本地模块 ──
from cc_audio_engine import AudioBridge
# VAD: 能量检测
from cc_tts_local import local_tts_to_pcm, preload as preload_tts
from cc_brain import think_stream, think_ollama_fast
from cc_stt_mlx import transcribe as mlx_transcribe, STTResult, preload as preload_stt
from cc_context import build_system_prompt
from cc_events import post_event

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("jarvis")

# ── 配置 ──
VAD_THRESHOLD = 0.5
MIC_SAMPLE_RATE = 48000       # Swift 桥接输出的采样率
STT_SAMPLE_RATE = 16000       # SenseVoice 需要 16kHz
VAD_WINDOW = 512              # Silero VAD 帧大小（32ms @ 16kHz）

MIN_SPEECH_MS = 300           # 最短有效语音段
MIN_SILENCE_MS = 400          # 静音多久算说完
WAKE_WORDS = ["贾维斯", "jarvis", "嘉维斯", "贾维丝",
              "小维斯", "蒋维斯", "夏维斯", "茶维斯", "加维斯",
              "假维斯", "甲维斯", "家维斯", "维斯"]  # 模糊匹配 ASR 常见误识
ACTIVE_SESSION_TIMEOUT = 120  # 唤醒后免唤醒词时间

# barge-in：VP AEC 后回声 ~0.009，人声 ~0.074，阈值 0.03 完美区分
BARGE_IN_ENERGY = 0.03
BARGE_IN_FRAMES = 4           # 连续 4 帧确认（~128ms）

AUDIO_PATH = "/tmp/cc-jarvis-segment.wav"


class JarvisV2:
    """贾维斯 V2 全双工语音引擎"""

    def __init__(self):
        # 音频桥接（Swift VP AEC）
        self.bridge = AudioBridge()

        # VAD: 能量检测（稳定可靠，无框架冲突）
        self._energy_threshold = 0.008  # 说话能量阈值

        # 状态
        self._state = "IDLE"  # IDLE / LISTENING / PROCESSING / SPEAKING
        self._state_lock = threading.Lock()

        # 语音段收集
        self._speech_frames: list = []
        self._speech_ms: float = 0
        self._silence_ms: float = 0
        self._is_speech: bool = False

        # 重采样缓冲（48kHz → 16kHz）
        self._resample_buf = np.array([], dtype=np.float32)

        # 唤醒
        self._last_wake_time: float = 0

        # barge-in
        self._barge_in_count: int = 0
        self._is_playing: bool = False

        # TTS 回声过滤
        self._recent_tts: list = []

        # 处理队列
        self._segment_queue: queue.Queue = queue.Queue()

        # 回复中标志
        self._responding = False

        # MLX GPU 锁（STT 和 TTS 不能同时用 GPU）
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
        print("  贾维斯 V2 — 全双工语音交互")
        print("  音频：Swift VP AEC 硬件回声消除")
        print("  TTS：Qwen3-TTS MLX (M5 GPU)")
        print("  LLM：本地 3b + MiniMax 云端")
        print("  Ctrl+C 退出")
        print("=" * 50)

        # 先加载模型（阻塞，确保 GPU 不并发）
        print("  加载模型中...")
        self._preload()
        logger.info("模型全部就绪")

        # 模型就绪后再启动音频
        if not self.bridge.start(
            on_mic_chunk=self._on_mic_chunk,
            on_play_done=self._on_play_done,
        ):
            logger.error("音频桥接启动失败，降级到 sounddevice")
            return self._start_fallback()

        # 启动处理线程
        threading.Thread(target=self._process_loop, daemon=True, name="process").start()

        logger.info("就绪，说「贾维斯」唤醒")
        self._greet()

        # 主线程等待
        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        except KeyboardInterrupt:
            print("\n贾维斯下线")
        finally:
            self.bridge.stop()

    def _preload(self):
        """后台顺序预加载模型（MLX 不支持并行加载）"""
        try:
            preload_tts()  # TTS 先加载（启动问候需要）
        except Exception as e:
            logger.error(f"TTS 预加载失败: {e}")
        try:
            preload_stt()  # STT 后加载
        except Exception as e:
            logger.error(f"STT 预加载失败: {e}")

    def _greet(self):
        """启动问候"""
        hour = datetime.now().hour
        if hour < 6: g = "夜深了。"
        elif hour < 9: g = "早。"
        elif hour < 12: g = "上午好。"
        elif hour < 14: g = "中午好。"
        elif hour < 18: g = "下午好。"
        elif hour < 22: g = "晚上好。"
        else: g = "还在忙？"

        threading.Thread(target=self._speak, args=(g,), daemon=True).start()

    # ════════════════════════════════════════
    #  麦克风回调（从 Swift VP 管道来）
    # ════════════════════════════════════════

    def _on_mic_chunk(self, chunk: np.ndarray):
        """
        收到 VP 处理后的麦克风数据（48kHz float32）。
        需要降采样到 16kHz 给 VAD。
        """
        # 降采样 48kHz → 16kHz（取每 3 个样本中的 1 个）
        self._resample_buf = np.concatenate([self._resample_buf, chunk])
        ratio = 3  # 48000 / 16000
        usable = (len(self._resample_buf) // ratio) * ratio
        if usable == 0:
            return
        resampled = self._resample_buf[:usable:ratio]
        self._resample_buf = self._resample_buf[usable:]

        # 按 VAD 帧大小处理
        for i in range(0, len(resampled) - VAD_WINDOW + 1, VAD_WINDOW):
            frame = resampled[i:i + VAD_WINDOW]
            self._process_vad_frame(frame)

    def _vad_reset(self):
        """重置 VAD 状态"""
        pass

    def _process_vad_frame(self, frame: np.ndarray):
        """处理一帧 VAD（能量检测）"""
        energy = float(np.abs(frame).mean())

        # ── 回复/播放中：只做 barge-in 检测 ──
        if self._responding or self._is_playing:
            # 每 50 帧打一次 energy 日志（~1.6s）
            if not hasattr(self, '_barge_log_count'):
                self._barge_log_count = 0
            self._barge_log_count += 1
            if self._barge_log_count % 50 == 0:
                logger.debug(f"[barge] energy={energy:.4f} threshold={BARGE_IN_ENERGY}")

            if energy >= BARGE_IN_ENERGY:
                self._barge_in_count += 1
                if self._barge_in_count >= BARGE_IN_FRAMES:
                    logger.info(f"barge-in! (energy={energy:.4f})")
                    sd.stop()  # 立即停止播放
                    self._is_playing = False
                    self._responding = False
                    self._barge_in_count = 0
                    self.state = "IDLE"
                    self._speech_frames.clear()
                    self._speech_ms = 0
                    self._silence_ms = 0
                    self._is_speech = False
            else:
                self._barge_in_count = 0
            return

        # ── 正常模式：语音段切分 ──
        frame_ms = VAD_WINDOW / STT_SAMPLE_RATE * 1000  # 32ms

        is_speech = energy >= self._energy_threshold

        if not self._is_speech:
            if is_speech:
                # 开始说话
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
                    # 说完了
                    if self._speech_ms >= MIN_SPEECH_MS:
                        audio = np.concatenate(self._speech_frames)
                        self._segment_queue.put(audio)
                    self._is_speech = False
                    self._speech_frames.clear()
                    self._speech_ms = 0
                    self._silence_ms = 0

    # ════════════════════════════════════════
    #  播放回调
    # ════════════════════════════════════════

    def _on_play_done(self):
        """单句播完回调。不改状态——状态由 _handle_segment 整体控制。"""
        self._is_playing = False

    # ════════════════════════════════════════
    #  处理循环（STT → LLM → TTS）
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
        """处理语音段：STT → 唤醒词 → LLM → TTS"""
        self.state = "PROCESSING"

        # 能量门槛：过滤噪音/静音段
        energy = float(np.abs(audio).mean())
        if energy < 0.005:
            self.state = "IDLE"
            return

        # STT（Qwen3-ASR MLX，~130ms，GPU 独占）
        t0 = time.time()
        sf.write(AUDIO_PATH, audio, STT_SAMPLE_RATE)
        duration = len(audio) / STT_SAMPLE_RATE
        with self._mlx_lock:
            segment = mlx_transcribe(AUDIO_PATH, duration)
        stt_ms = (time.time() - t0) * 1000

        if not segment or not segment.text or not segment.text.strip():
            self.state = "IDLE"
            return

        text = segment.text.strip()

        # 幻觉过滤：重复字符（"喂喂喂"、"嗯嗯嗯"等）
        unique_chars = set(text.replace("。", "").replace("，", "").replace(" ", ""))
        if len(unique_chars) <= 1 and len(text) > 2:
            self.state = "IDLE"
            return

        logger.info(f"[STT] {stt_ms:.0f}ms: {text}")

        # 先检查唤醒词（确保激活会话）
        has_wake, clean_text = self._check_wake(text)
        if not has_wake:
            self.state = "IDLE"
            return

        if not clean_text:
            self.state = "IDLE"
            return

        # 再做回声过滤（对清洗后的文本）
        if self._is_echo(clean_text):
            logger.info(f"[echo] 过滤: {clean_text}")
            self.state = "IDLE"
            return

        logger.info(f"[用户] {clean_text}")

        # LLM → TTS 预合成管道：播 A 时合成 B，零停顿
        self._responding = True
        self.state = "SPEAKING"

        sentences = []
        synth_queue = queue.Queue()  # (pcm, sr, text)

        # 后台合成线程
        def _synth_worker():
            for sentence in think_stream(clean_text):
                if not self._responding:
                    break
                logger.info(f"[回复] {sentence}")
                t0 = time.time()
                with self._mlx_lock:
                    pcm, sr = local_tts_to_pcm(sentence)
                logger.info(f"[TTS] {(time.time()-t0)*1000:.0f}ms | {len(pcm)/sr:.1f}s")
                synth_queue.put((pcm, sr, sentence))
            synth_queue.put(None)  # 结束标记

        threading.Thread(target=_synth_worker, daemon=True).start()

        # 主线程：逐句播放（合成线程跑在前面）
        while self._responding:
            item = synth_queue.get()
            if item is None:
                break
            pcm, sr, text = item
            if not self._responding:
                break

            self._recent_tts.append(text)
            if len(self._recent_tts) > 10:
                self._recent_tts.pop(0)

            self._is_playing = True
            sd.play(pcm, sr)
            # 非阻塞等待：每 50ms 检查打断
            while sd.get_stream().active:
                if not self._responding:
                    sd.stop()
                    logger.info("播放被打断")
                    break
                time.sleep(0.05)
            self._is_playing = False
            self._last_play_time = time.time()

        self._responding = False
        self._vad_reset()
        self._speech_frames.clear()
        self._is_speech = False
        self._silence_ms = 0
        self._speech_ms = 0
        time.sleep(0.15)
        self.state = "IDLE"

    def _check_wake(self, text: str) -> tuple:
        """检查唤醒词，返回 (has_wake, clean_text)"""
        text_lower = text.lower()

        # 显式唤醒
        for w in WAKE_WORDS:
            if w in text_lower:
                clean = text_lower.replace(w, "").strip("。？！，、；：. ")
                self._last_wake_time = time.time()
                if not clean:
                    # 只说了唤醒词 → 本地秒回
                    self._speak("在。")
                    return False, ""
                return True, clean

        # 活跃会话期间免唤醒
        if (time.time() - self._last_wake_time) < ACTIVE_SESSION_TIMEOUT:
            return True, text

        return False, ""

    def _is_echo(self, text: str) -> bool:
        """
        文本级回声检测：STT 结果和最近 TTS 输出对比。
        相似度高于阈值 → 判定为回声。
        阈值根据上下文动态调整：刚播完时严格，间隔久了宽松。
        """
        if not text or len(text) < 2:
            return False

        clean = text.strip("。？！，、；：.!? ")
        if not clean:
            return True

        # 动态阈值：刚播完严格，长句回声可能延迟很久
        time_since_play = time.time() - getattr(self, '_last_play_time', 0)
        if time_since_play < 3.0:
            threshold = 0.35  # 刚播完，严格
        elif time_since_play < 8.0:
            threshold = 0.5   # 中等（长句回声可能晚到）
        elif time_since_play < 15.0:
            threshold = 0.7   # 宽松（很长的句子）
        else:
            return False

        for tts_text in self._recent_tts[-8:]:
            tts_clean = tts_text.strip("。？！，、；：.!? ")
            if not tts_clean:
                continue

            # 计算相似度：最长公共子序列比率
            similarity = self._text_similarity(clean, tts_clean)
            if similarity >= threshold:
                logger.info(f"[echo] 过滤 sim={similarity:.2f}: '{clean}' ≈ '{tts_clean}'")
                return True

        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """两个文本的相似度（基于公共字符比率，快速近似）"""
        if not a or not b:
            return 0.0
        # 短文本用字符集重叠
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) <= 3:
            common = sum(1 for c in shorter if c in longer)
            return common / len(shorter)
        # 长文本用连续子串匹配
        # 找最长公共子串
        max_match = 0
        for i in range(len(shorter)):
            for j in range(i + 2, len(shorter) + 1):
                sub = shorter[i:j]
                if sub in longer:
                    max_match = max(max_match, len(sub))
        return max_match / len(shorter)

    # ════════════════════════════════════════
    #  TTS 播放
    # ════════════════════════════════════════

    def _speak(self, text: str):
        """合成并播放单句（用于唤醒回复等短句）"""
        self._recent_tts.append(text)
        if len(self._recent_tts) > 10:
            self._recent_tts.pop(0)

        try:
            with self._mlx_lock:
                pcm, sr = local_tts_to_pcm(text)
        except Exception as e:
            logger.error(f"TTS 失败: {e}")
            return

        self._is_playing = True
        sd.play(pcm, sr)
        while sd.get_stream().active:
            time.sleep(0.05)
        self._is_playing = False
        self._last_play_time = time.time()
        self._speech_frames.clear()
        self._is_speech = False
        time.sleep(0.15)

    # ════════════════════════════════════════
    #  降级模式（无 Swift 桥接）
    # ════════════════════════════════════════

    def _start_fallback(self):
        """降级到 sounddevice 半双工"""
        logger.warning("降级到半双工模式")
        # 复用旧的 cc_interact 逻辑
        from cc_interact import main as old_main
        old_main()


def main():
    jarvis = JarvisV2()
    jarvis.start()


if __name__ == "__main__":
    main()
