# cc-eye / 贾维斯 — 系统架构文档

> V3.5 | 2026-03-30 | 基于源码梳理

钢铁侠式多模态 AI 助手 —— **全双工语音 + 摄像头视觉 + 多模型并行 LLM + 音乐控制**，常驻运行在 MacBook Air M5 上。

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        贾维斯 V3.5 架构                               │
│                                                                      │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌───────┐ │
│  │ 耳 (听觉) │  │ 眼 (视觉) │  │  脑 (思考)    │  │嘴 (输出)│  │手(工具)│ │
│  │          │  │          │  │              │  │        │  │       │ │
│  │Swift AEC │  │Qwen2-VL  │  │4B 本地 0.3s  │  │Qwen3   │  │音乐   │ │
│  │能量 VAD  │  │2B 快扫10s│  │Gemini  ~2s   │  │TTS 1.7B│  │ncm-cli│ │
│  │Qwen3-ASR │  │7B 精扫60s│  │MiniMax ~3s   │  │Base+ref│  │飞书API│ │
│  │0.6B 130ms│  │trylock   │  │豆包/GPT 备用 │  │独立进程 │  │osascr.│ │
│  └─────────┘  └──────────┘  └──────────────┘  └────────┘  └───────┘ │
│                                                                      │
│  ── 横向连接 ──                                                       │
│  状态机 (cc_state) · 上下文注入 (cc_context) · 事件流 (cc_events)      │
│  记忆桥接 (cc_memory_bridge) · 工具意图识别 (cc_tools)                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心数据流

```
麦克风 (48kHz)
  │ 降采样 48k→16k（每3采样取1）
  ▼
Swift VP AEC (硬件级回声消除，子进程 cc_audio_bridge)
  │
  ▼
┌─ 是否在播放 TTS? ─┐
│ 是                  │ 否
│ 动态阈值打断检测     │ 正常 VAD
│ energy > max(0.05,  │ energy > 0.008
│  playback×0.3+0.03) │
│ 连续6帧 → barge-in  │ 语音段收集
└─────────────────────┘
  │
  ▼ 语音段完整（≥300ms语音 + ≥400ms静音）
  │
  ▼
openwakeword (音频级) + 拼音匹配 + 文本精确匹配
  │ 未唤醒 + 非活跃会话(>120s) → 丢弃
  ▼
Qwen3-ASR 0.6B MLX STT (~130ms，持 mlx_lock)
  │
  ▼
回声过滤（2s内 TTS 文本相似度 ≥0.7 → 丢弃）
  │
  ▼
安静模式检测（"闭嘴"→进入 / "说吧"→恢复）
  │
  ▼
工具意图检测 (cc_tools.detect_tool_intent)
  ├─ 音乐/飞书 → 直接执行，结果 TTS 播报
  └─ 非工具 → 进入 LLM 路由
      │
      ▼
  think_stream() 并行推理
  ┌────────────────────────────────────┐
  │ L0: __TRANSITION__ 预缓存语气词    │ → TTS 秒播 (0ms)
  │ L1: 4B 本地流式 (max_tokens=30)    │ → 先说（~300ms）
  │ L2/L3: 云端并行请求               │ → 到了接管（去重>50%跳过）
  └────────────────────────────────────┘
      │
      ▼
  TTS 合成管道（cc_tts_local → cc_tts_server UDS）
      │
      ▼
  逐句播放 + 段间 crossfade 15ms
      │
      ▼ barge-in 可随时打断
  回到 IDLE
```

---

## 3. 进程架构

```
cc_jarvis_v3.py (主进程, PID=1)
  │
  ├─ cc_audio_bridge (子进程)
  │   └─ Swift AVAudioEngine
  │       ├─ VP AEC (硬件回声消除)
  │       ├─ 麦克风输入 → 二进制管道 → Python
  │       └─ 协议: MSG_MIC(0x10) / MSG_PLAY(0x01) / MSG_READY(0x12)
  │
  ├─ cc_tts_server.py (子进程, 独立 Metal GPU 上下文)
  │   ├─ UDS 监听: /tmp/cc-tts.sock
  │   ├─ 协议: 长度前缀帧 + msgpack
  │   ├─ Qwen3-TTS 1.7B Base 8bit MLX
  │   ├─ ref_audio 音色锁定 (ICL)
  │   └─ 32+ 条预缓存 (npz 磁盘固化)
  │
  ├─ cc_vision_mlx._monitor_loop (线程)
  │   ├─ 摄像头 640×480 采集
  │   ├─ Qwen2-VL-2B 快扫 (每10s)
  │   ├─ Qwen2.5-VL-7B 精扫 (每60s)
  │   └─ trylock(0.1s)：拿不到 mlx_lock 就跳过
  │
  ├─ _process_loop (线程)
  │   └─ 从 segment_queue 取语音段 → STT → LLM → TTS
  │
  └─ _synth (线程, 每次回复临时创建)
      └─ think_stream() → local_tts_to_pcm() → synth_queue → 主线程播放

oMLX (外部进程, 需手动启动)
  └─ localhost:8000, OpenAI 兼容, Qwen3.5-4B-MLX-4bit

camera_daemon.py (独立进程, 可选, 旧版)
  └─ ollama moondream/minicpm-v 方案
```

### 进程间通信

| 通道 | 协议 | 用途 |
|------|------|------|
| cc_audio_bridge ↔ Python | 二进制管道 (stdin/stdout) | 麦克风/播放音频传输 |
| cc_tts_server ↔ cc_tts_local | UDS `/tmp/cc-tts.sock` + msgpack | TTS 合成请求/响应 |
| cc_vision_mlx → 文件系统 | JSON 文件写入 | 场景描述、事件日志 |
| 主进程 → oMLX | HTTP `localhost:8000` | 4B 本地 LLM 推理 |
| 主进程 → 云端 | HTTPS | Gemini/MiniMax/豆包 API |

---

## 4. 模块清单

### 4.1 主程序与状态

| 文件 | 职责 | 关键细节 |
|------|------|---------|
| `cc_jarvis_v3.py` | 主入口 | 全双工循环、VAD、唤醒、barge-in、回声过滤、安静模式 |
| `cc_state.py` | 语音状态机 | IDLE→LISTENING→PROCESSING→SPEAKING→INTERRUPTED + TOOL_CALLING + QUIET |

### 4.2 耳 — 音频采集与语音识别

| 文件 | 职责 | 关键指标 |
|------|------|---------|
| `cc_audio_engine.py` | AudioBridge 类：管理 Swift 子进程 | 二进制管道通信, 48kHz float32 |
| `cc_audio_bridge.swift` | macOS 原生音频引擎 | AVAudioEngine + VP AEC |
| `cc_stt_mlx.py` | Qwen3-ASR 0.6B MLX 推理 | ~130ms, 持 mlx_lock |
| `cc_vad.py` | Silero VAD (ONNX) | 无 torch 依赖, 备选方案 |
| `cc_aec.py` | NLMS 自适应滤波 (软件层) | 4096 taps, mu=0.05, 备选 |
| `cc_listen.py` | 能量 VAD + SenseVoice STT | 旧版, 已被 V3 替代 |

**V3 实际使用的 VAD**：不是 Silero，而是 `cc_jarvis_v3.py` 内置的**能量阈值 VAD**（`_process_vad_frame`），阈值 0.008，窗口 512 样本 (32ms)。Silero VAD 作为模块存在但 V3 主路径未调用。

### 4.3 眼 — 视觉感知

| 文件 | 职责 | 关键指标 |
|------|------|---------|
| `cc_vision_mlx.py` | VisionEngine 类：双模型 MLX 视觉 | 快扫 2B/10s + 精扫 7B/60s |
| `camera_daemon.py` | 独立守护进程 (旧版 ollama 方案) | moondream + minicpm-v |
| `cc_face.py` | LBPH 人脸识别 (持续学习) | OpenCV, 阈值 120.0 |
| `identity.py` | DeepFace Facenet 身份识别 | 128 维 embedding |
| `expression.py` | DeepFace 7 类表情识别 | EMA α=0.4 |
| `state_fusion.py` | 多信号融合 → 5 种用户状态 | 专注/疲劳/愉悦/困惑/离开 |
| `vision_models.py` | ollama 多模态接口 | moondream/minicpm-v 备选 |
| `look.py` | CLI 快速看一眼 | Claude Code `/cc-eye` skill |

**V3 实际使用的视觉**：`cc_vision_mlx.py` (MLX 原生，嵌入主进程线程)。`camera_daemon.py` 是旧版 ollama 方案，可独立运行但非 V3 默认。

### 4.4 脑 — LLM 路由与上下文

| 文件 | 职责 | 说明 |
|------|------|------|
| `cc_brain.py` | 并行推理主入口 `think_stream()` | 4B+云端并行、去重接管、降级链 |
| `cc_context.py` | 上下文注入器 | CC_IDENTITY + 状态 + 记忆 + 场景 + 事件 |
| `cc_tools.py` | 工具意图识别 + 执行 | 正则 30+ 规则 + Gemini 智能选歌 |
| `cc_events.py` | 统一感知事件流 | deque(200) + 文件双写 |
| `cc_memory_bridge.py` | 日汇总感知记忆 | 本地 LLM 摘要 → RECENT_EVENTS.md |

### 4.5 嘴 — TTS 合成与播放

| 文件 | 职责 | 关键指标 |
|------|------|---------|
| `cc_tts_server.py` | TTS 独立进程 | UDS, Qwen3-TTS 1.7B Base 8bit |
| `cc_tts_local.py` | TTS 客户端 | 缓存 <1ms, miss 走 UDS, 降级本地推理 |
| `cc_voice_profile.py` | 音色配置 | Base 模型 + ref_audio + REF_TEXT |
| `cc_audio_out.py` | AudioPlayer 类 | sounddevice OutputStream, 可中断 |
| `cc_voice.py` | edge-tts 云端合成 | 备用, YunjianNeural |
| `cc_player.py` | 旧版播放器 | V3 用 cc_audio_out 替代 |
| `scripts/gen_jarvis_ref.py` | 生成参考音频 | VoiceDesign → ref_audio.wav |

### 4.6 旧版/兼容模块

| 文件 | 说明 |
|------|------|
| `cc_jarvis.py` | V2 版本入口 (已被 V3 替代) |
| `cc_interact.py` | V2 全双工引擎 (已被 V3 替代) |
| `main.py` | 基础视觉感知入口 (人脸+表情+融合) |
| `interaction.py` | cc 风格互动回应 + 对话气泡 UI |
| `live_view.py` | 实时视图调试工具 |

---

## 5. LLM 并行路由系统

### 5.1 模型清单

| 级别 | 模型 | 接口 | 首 token | 用途 |
|------|------|------|----------|------|
| L0 预缓存 | 32+ 句预合成 PCM | 内存 | 0ms | `__TRANSITION__` 过渡语气词 |
| L1 本地 | **Qwen3.5-4B-MLX-4bit** | oMLX `localhost:8000` | ~300ms | 流式首句 (max_tokens=30) |
| L1 降级 | qwen2.5:3b | ollama `localhost:11434` | ~500ms | oMLX 不可用时兜底 |
| L2 快速 | Gemini 2.5 Flash | 代理 (OpenAI 兼容) | ~2s | 简单问答、闲聊、视觉 |
| L2 备用 | doubao-seed-2.0-lite | 豆包 API | ~2s | Gemini 不可用时备选 |
| L3 深度 | MiniMax M2.7-highspeed | `api.minimaxi.com` | ~3s | 分析、方案、为什么 |
| L3 备用 | GPT 5.4 | 代理 (VPS 不稳) | ~4s | 暂缓使用 |

### 5.2 4B 本地模型 — 核心并行组件

4B **不是**简单的"过渡语填充"，而是**与云端并行执行、先行输出**的核心组件。

**运行配置**：

```python
LOCAL_LLM_API = "http://localhost:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "Qwen3.5-4B-MLX-4bit"

# 请求参数
max_tokens = 30          # 只说 1-2 句
temperature = 0.7
enable_thinking = False  # 纯生成，不走 CoT
```

**Prompt 设计** (`_LOCAL_SYSTEM`)：

```
你是贾维斯，川哥的搭档。你通过摄像头能看到川哥的环境。
[你现在看到的]后面是你的实时画面描述。
规则：1句话，15字以内。不说"作为AI"。

示例：
川哥：你看到什么 → 你在桌前看电脑呢。
川哥：天气怎么样 → 我查查最新的。
川哥：播点音乐 → 马上安排。
```

**上下文注入**（精简版，控制 prefill 时间）：

- 最近 2 轮历史（不是全部 10 轮）
- 视觉场景描述（`get_scene_context()`）
- 最近 30 秒事件（只取最后 2 行）

### 5.3 路由完整流程

```
用户语音文本
  │
  ├─ 工具意图? (cc_tools.detect_tool_intent)
  │   ├─ 音乐/飞书 → 直接执行，不走 LLM
  │   └─ music_smart → __TRANSITION__ + Gemini 智能选歌
  │
  ├─ _needs_cloud() == False
  │   │ (纯问候：你好/谢谢/再见/晚安/好的/嗨)
  │   └─→ 4B 独立回答，不启动云端
  │
  ├─ _needs_deep_think() == False (简单问题)
  │   ├─→ L0: __TRANSITION__ 预缓存语气词 (0ms)
  │   ├─→ L1: 4B 本地流式 ─────────────┐
  │   └─→ L2: Gemini Flash (fast 模式)  │ 并行执行
  │        └─ 失败 → 降级 MiniMax ──────┘
  │
  └─ _needs_deep_think() == True (复杂问题: >30字 或 含深度关键词)
      ├─→ L0: __TRANSITION__ 预缓存语气词 (0ms)
      ├─→ L1: 4B 本地流式 ─────────────────────┐
      └─→ L2: Gemini (deep_intro: 2-3句概述)    │ 并行执行
           └─→ L3: MiniMax (deep_detail: 展开)  │
                传入 Gemini 概述避免重复 ─────────┘

播放逻辑:
  1. 4B 先说 → 用户听到即时回应 (~300ms)
  2. 云端结果到达 → 接管输出
  3. 去重: >50% 字符重叠的句子跳过
  4. 全部失败 → think_local() / think_ollama() 兜底
```

### 5.4 三种 Prompt 模式

| 模式 | 对应模型 | Prompt 要点 |
|------|---------|------------|
| `fast` | Gemini | 1-2 句答案，第一句就是答案，自适应长度 |
| `deep_intro` | Gemini | 核心判断 + 关键结论，2-4 句有干货，不空泛 |
| `deep_detail` | MiniMax | 接续概述深入展开，用衔接词，不重复 |

### 5.5 上下文构建 (`_build_context`)

```
CC_IDENTITY (170 tok)                    ← 贾维斯人格
  + [当前状态] status.md (500 chars)     ← 焦点
  + [记忆] memory-items.md (800 chars)   ← 长期记忆
  + [当前时间] 2026-03-30 14:30 星期日
  + [最近2分钟事件] cc_events 时间线      ← 感知融合
  + [当前视觉场景] scene.json             ← 实时画面
  + [对话摘要] 压缩的旧对话               ← 超过20轮时压缩
  + [语音场景] 模式相关指令               ← fast/deep_intro/deep_detail
```

### 5.6 对话历史管理

```python
MAX_HISTORY = 10                # 发给云端的最近轮数
MAX_HISTORY_BEFORE_SUMMARY = 20 # 超过此数触发压缩

# 压缩策略：
# 超过 20 轮 → 把旧的一半提取摘要（"川哥: xxx; cc: xxx"）
# 摘要追加到 _conversation_summary，注入 prompt
```

---

## 6. 全双工关键技术

### 6.1 回声消除 (AEC)

**硬件级** — Swift cc_audio_bridge：

```
AVAudioEngine → inputNode.setVoiceProcessingEnabled(true)
  → 禁用 ducking（防 VP 压低系统输出）
  → 麦克风音频经 VP 处理后通过二进制管道送 Python
  → 协议: 长度前缀(4B) + float32 PCM 数据
```

**软件级** — cc_aec.py (NLMS, 备选)：

```
滤波器长度: 4096 (256ms@16kHz)
步长 mu: 0.05
尾部消散: 0.5s
```

### 6.2 Barge-In 打断机制

```python
# 播放中的打断检测（_process_vad_frame）
playback_energy = self.player.current_playback_energy
dynamic_threshold = max(BARGE_IN_ENERGY,       # 0.05 底线
                        playback_energy * 0.3 + 0.03)  # 动态

if energy >= dynamic_threshold:
    bi_count += 1
    if bi_count >= BARGE_IN_FRAMES:  # 连续 6 帧 (~192ms)
        player.interrupt()           # 立即停止播放
        state = "IDLE"
        reset_vad()
```

**文本级回声过滤**（播完 2s 内的额外保护）：

- 和最近 8 条 TTS 文本比较
- 相似度 ≥ 0.7 → 判定为回声，丢弃
- 相似度 ≥ 0.4 + barge-in 场景 → 也丢弃

**barge-in 内容判断** (`_check_barge_in_content`)：

1. 唤醒词 → 立即打断
2. 语气词/感叹词 (嗯/啊/哦/哈哈 等 18 个) → 不打断
3. TTS 回声 (sim≥0.4) → 不打断
4. 实质内容 (≥3 字) → 打断 + 送入处理队列

### 6.3 唤醒机制（三层）

| 层级 | 实现 | 说明 |
|------|------|------|
| 音频级 | openwakeword `hey_jarvis_v0.1` | score>0.5 即唤醒 |
| 拼音级 | `_has_weisi()`: pypinyin 匹配连续 wei+si | 覆盖所有"X维斯"变体 |
| 文本级 | 13 个精确匹配词 + `jarvis` 英文 | STT 后匹配 |

**活跃会话**：唤醒后 120 秒内免再喊。

**唤醒应答**：从 8 条预缓存中随机选取（避免连续重复）：

```
"我在的。" / "你说吧。" / "在的呢。" / "随时待命。"
"怎么了？" / "好，我在。" / "需要我做什么？" / "还有别的事吗？"
```

### 6.4 安静模式

```
"闭嘴" / "安静" / "别说了" → 进入 QUIET 状态
  │ TTS: "好的，我听着。"
  │ 语音仍在 STT，但不回复，只记录到事件流
  ↓
"说吧" / "你说" / "继续说" → 恢复 IDLE
  TTS: "好，我在。"
```

### 6.5 音乐播放时降噪

```python
# 检测 mpv 进程（1秒缓存避免频繁 pgrep）
if _is_music_playing():
    audio = _bandpass_voice(audio)  # 带通滤波 300Hz-3kHz
    # 只留人声频段，滤掉背景音乐
    # Butterworth 4阶, scipy.signal.butter + sosfilt
```

---

## 7. 视觉感知系统

### 7.1 MLX 视觉引擎 (cc_vision_mlx.py) — V3 主用

```
VisionEngine (嵌入主进程线程, 共享 mlx_lock)
  │
  ├─ 摄像头: cv2.VideoCapture(0), 640×480
  │
  ├─ 快扫 (每 10s)
  │   ├─ 模型: mlx-community/Qwen2-VL-2B-Instruct-4bit
  │   ├─ Prompt: "用中文简要描述画面：有没有人、在做什么、桌上有什么。2句话。"
  │   ├─ trylock(0.1s): 拿不到锁就跳过
  │   └─ 输出: /tmp/cc-eye-scene.json + events.jsonl
  │
  └─ 精扫 (每 60s)
      ├─ 模型: mlx-community/Qwen2.5-VL-7B-Instruct-4bit
      ├─ Prompt: "详细描述：1)人物动作表情 2)桌面物品 3)环境氛围 4)变化。3-4句。"
      └─ trylock(0.1s): 同上
```

**trylock 策略**：

- 模型加载时：`with mlx_lock`（阻塞等锁，只发生一次）
- 推理时：`mlx_lock.acquire(timeout=0.1)` → 拿不到就跳过
- 目的：STT 和 TTS 优先级高于视觉

### 7.2 摄像头守护进程 (camera_daemon.py) — 旧版独立方案

```
1 秒主循环
  ├─ 运动检测 (160×120) ── 每帧极速
  ├─ 人脸检测 (320×240) ── 正常 3tick/s, 静态 6tick/s
  ├─ 快扫 moondream (10s) ── ollama 异步线程
  └─ 精扫 minicpm-v (60s) ── ollama 异步线程
```

### 7.3 状态融合 (state_fusion.py)

5 种综合状态，基于多信号滑窗融合 (5 秒窗口)：

| 状态 | 判定条件 |
|------|---------|
| 专注 | 注视屏幕, 表情中立, yaw < 15° |
| 疲劳 | 眨眼率 > 25次/分 且 EAR < 0.22 (需 2+ 信号) |
| 愉悦 | happy 表情 > 50% 窗口 |
| 困惑 | 负面表情 > 40% 或 (> 25% 且头部偏转 > 15°) |
| 离开 | 人脸检测率 < 15% |

输入信号：EAR (眼睛睁开度) · MAR (嘴巴张开度) · 眨眼频率 · 头部 yaw/pitch · 7 类表情概率 · 身份标签。

---

## 8. TTS 子系统

### 8.1 独立进程架构

```
cc_jarvis_v3.py
  │ 启动子进程
  ▼
cc_tts_server.py (独立 Metal GPU 上下文)
  │
  ├─ 模型: mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
  ├─ 音色: ref_audio (assets/voice/jarvis_ref.wav) + REF_TEXT
  ├─ 缓存: .venv/cache/tts_cache_base.npz (磁盘固化)
  ├─ UDS: /tmp/cc-tts.sock
  └─ 协议: 长度前缀(4B big-endian) + msgpack
      请求: {"action": "synthesize"|"health"|"shutdown", "text": "..."}
      响应: {"ok": bool, "pcm": bytes, "sample_rate": int, "shape": [N]}
```

### 8.2 合成流水线

```
cc_tts_local.py (客户端)
  │
  ├─ 内存缓存命中? ── 是 → PCM 直返 (<1ms)
  │
  ├─ UDS 可用? ── 是 → 发给 cc_tts_server → 等响应
  │   └─ 短连接（每次新建，避免 barge-in 数据流错位）
  │
  └─ UDS 不可用 → 本地直接推理（降级，不推荐）
```

### 8.3 音色方案

| 方案 | 模型 | 状态 | 说明 |
|------|------|------|------|
| ~~VoiceDesign~~ | instruct 模式 | 已弃用 | 每次推理随机音色 |
| **Base + ref_audio** | Base 1.7B 8bit | **当前使用** | ICL 音色锁定，稳定一致 |

**音色定义** (`cc_voice_profile.py`)：

```python
BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
REF_AUDIO_PATH = "assets/voice/jarvis_ref.wav"  # 3.5-5s
REF_TEXT = "所有系统运行正常，随时待命。"

# VoiceDesign instruct (仅用于重新生成 ref_audio)
JARVIS_VOICE_INSTRUCT = "A deep, warm, mature male voice with ..."
```

**约束**：ref_audio 必须 3.5-5s，超过 5s Base ICL 输出空音频。

### 8.4 预缓存

32 条高频短句预合成（启动时 miss 补全 + 磁盘固化）：

```
基础应答:  "好的呢。" "收到了。" "嗯，你说。" "明白了。" ...
过渡衔接:  "让我想想。" "稍等一下。" "我看看。" "好问题。" ...
安静模式:  "好的，我听着。" "好，我在。"
问候语:    "早上好。" "下午好。" "晚上好。"
系统应答:  "所有系统运行正常。" "一切就绪，随时待命。" ...
音乐控制:  "已停止播放。" "继续播放。" "下一首。" ...
```

### 8.5 播放细节

**智能过渡语** (`_pick_transition`)：

```python
TRANSITION_PHRASES = {
    "short":  ["嗯，你说。", "好的呢。", "收到了。", "我看看。"],
    "medium": ["让我想想。", "这个嘛。", "好问题。", ...],
    "long":   ["这个方向挺好的。", "我梳理一下思路。", ...],
}
# 根据用户输入长度选级别，随机不重复
```

**段间 crossfade** (消除接缝"喘气"声)：

```python
CROSSFADE_MS = 15
# 非首段: 跳过 50ms 静音前缀 + 15ms 淡入 (0.3→1.0 线性)
```

**流式断句** (LLM 输出 → 句子切分)：

| 标点 | 行为 | 原因 |
|------|------|------|
| `。？！` | 直接断 | 语义完整 |
| `，、；：` | 攒满 8 字再断 (`_MIN_CLAUSE_LEN`) | 避免碎片 |
| 无标点 | 缓冲区末尾统一 flush | 兜底 |
| ≤2 字纯标点 | 跳过 | 避免"喘气" |

---

## 9. 工具调用系统

### 9.1 意图识别 — 正则优先级链

`cc_tools.py` 定义 30+ 条正则规则 (`_TOOL_PATTERNS`)，按顺序匹配：

```
查询状态  "现在放的什么歌" → music_state
停止      "暂停/停/别播了/关掉" → music_stop
继续      "继续/接着放" → music_resume
切歌      "下一首/切歌/换一首" → music_next / music_prev
音量      "声音大点/小点" → music_vol_up / music_vol_down
推荐      "每日推荐" → music_random / music_recommend
随机      "放点歌/听音乐" → music_random
搜索播放  "播放xxx" → music_play (提取歌名)
搜索      "搜xxx的歌" → music_search
飞书发消息 "发飞书通知xxx" → feishu_send (提取内容)
飞书查消息 "飞书有什么消息" → feishu_read
```

**兜底**：正则没匹配到但含音乐关键词 (歌/音乐/轻松/嗨/古典/...) → `music_smart` (Gemini 智能选歌)。

### 9.2 音乐播放 — ncm-cli + orpheus

```
音乐指令
  │
  ├─ 正则匹配 → 直接执行 (music_play/stop/next/...)
  │   └─ ncm-cli search → _find_playable → _play_song
  │       ├─ 首曲播放
  │       └─ 后续 1-4 首加入队列 (queue add)
  │
  └─ Gemini 智能选歌 (music_smart)
      ├─ Gemini 同步调用: 用户意图 → JSON {"keyword": "..."}
      └─ ncm-cli search → 播放
```

**播放细节**：

- orpheus 模式：网易云 App 原生播放 (VIP 可用)
- 播放后自动隐藏 App 窗口 (`osascript`)
- 音量控制：直接调系统音量 (`osascript set volume`)
- 代理绕过：`_ncm_env()` 清除 `http_proxy/socks` 等环境变量

### 9.3 飞书集成

```python
class FeishuClient:
    # 凭证: ~/mycc/.env (FEISHU_APP_ID + FEISHU_APP_SECRET)
    # Token: tenant_access_token, 有效期 2h, 过期前 5min 自动刷新
    # Session: trust_env=False (绕过 SOCKS5 代理)

    send_message(text, chat_id)       # ✅ 可用
    get_recent_messages(chat_id, n)   # ✅ 可用
    # read_doc                        # TODO
    # query_bitable                   # TODO
```

默认群聊：三机器人群 (cc + codex + gemini)。

---

## 10. 事件系统与记忆

### 10.1 统一感知事件流 (cc_events.py)

```python
# 内存: deque(maxlen=200), 线程安全
# 文件: /tmp/cc-eye-unified-events.jsonl (进程间共享)

post_event(event_type, detail, source)
# event_type: vision / speech / response / face / scene / system
# source: daemon / interact / brain / jarvis

get_context_window(seconds=120)
# 返回最近 2 分钟的事件，格式化为 LLM 可读文本
```

### 10.2 交互日志 (自学习数据)

```python
# /tmp/cc-eye-interactions.jsonl
# 每次交互记录:
{
    "ts": "2026-03-30T14:30:00",
    "user": "今天天气怎么样",
    "route": "gemini",          # gemini / minimax / local / tool
    "local_reply": "我查查。",
    "cloud_reply": "今天北京晴，15度...",
    "latency": 2.3
}
```

### 10.3 记忆桥接 (cc_memory_bridge.py)

```
每日运行 (手动或定时):
  │
  ├─ 读 interactions.jsonl (今天的交互)
  ├─ 读 events.jsonl (今天有意义的视觉事件)
  │   └─ 只取: person_appeared / person_left / scene_described / detail_scan
  │
  ├─ 本地 LLM (qwen2.5:3b) 生成摘要
  │   └─ 隐私红线: 不存音频/图像路径, 闲聊只存主题, 决策类可存具体内容
  │
  └─ 追加到 mycc/0-System/RECENT_EVENTS.md
```

---

## 11. 与 Claude Code 集成

### 11.1 Skill 注册

项目作为 `/cc-eye` skill 注册到 Claude Code。

触发词：`/look`、`看一眼`、`摄像头`、`环境里有什么`、`cc-eye`

### 11.2 look.py CLI

```bash
python look.py                    # 拍照 + moondream 描述
python look.py "桌上有什么?"       # 自定义问题
python look.py --scene            # 读最新场景缓存 (/tmp/cc-eye-scene.json)
python look.py --events           # 查看最近事件
```

### 11.3 共享文件协议

| 文件 | 写入方 | 消费方 | 格式 |
|------|--------|--------|------|
| `/tmp/cc-eye-latest.jpg` | vision_mlx / camera_daemon | look.py, Claude Code | JPEG |
| `/tmp/cc-eye-scene.json` | vision_mlx | cc_context, cc_brain, Claude Code | `{"ts","description","mode"}` |
| `/tmp/cc-eye-events.jsonl` | vision_mlx, cc_events | cc_context, cc_brain | NDJSON |
| `/tmp/cc-eye-unified-events.jsonl` | cc_events | cc_brain._build_context | NDJSON |
| `/tmp/cc-eye-interactions.jsonl` | cc_brain | cc_memory_bridge | NDJSON |
| `/tmp/cc-tts.sock` | cc_tts_server | cc_tts_local | UDS + msgpack |

---

## 12. 配置参考

### 12.1 语音参数 (cc_jarvis_v3.py)

```python
STT_SAMPLE_RATE      = 16000
VAD_WINDOW           = 512       # 32ms @ 16kHz
ENERGY_THRESHOLD     = 0.008     # 正常 VAD 阈值
MIN_SPEECH_MS        = 300       # 最短语音段
MIN_SILENCE_MS       = 400       # 最短尾部静音
BARGE_IN_ENERGY      = 0.05      # 打断能量底线
BARGE_IN_FRAMES      = 6         # ~192ms 防抖
ACTIVE_SESSION_TIMEOUT = 120     # 活跃会话超时 (秒)
```

### 12.2 视觉参数 (cc_vision_mlx.py)

```python
FAST_MODEL     = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DETAIL_MODEL   = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
FAST_INTERVAL  = 10    # 快扫间隔 (秒)
DETAIL_INTERVAL = 60   # 精扫间隔 (秒)
MAX_EVENTS     = 200   # 事件文件最大行数 (滚动保留)
```

### 12.3 视觉参数 (config.py, 人脸/表情/融合)

```python
FACE_MATCH_TOLERANCE       = 0.45     # 人脸匹配阈值
FACE_REGISTER_COUNT        = 10       # 注册采集数
EXPRESSION_SMOOTHING       = 0.4      # 表情 EMA 系数
STATE_WINDOW_SECONDS       = 5.0      # 融合滑窗
FATIGUE_BLINK_RATE         = 25       # 疲劳：眨眼次/分
FATIGUE_EAR_THRESHOLD      = 0.22     # 疲劳：眼睛睁开度
FOCUS_YAW_THRESHOLD        = 15.0     # 困惑：头部偏转度
INTERACTION_COOLDOWN       = 30.0     # 互动冷却 (秒)
```

### 12.4 环境变量 (.env, gitignore 保护)

```
# Gemini 代理 (主用, OpenAI 兼容)
GEMINI_PROXY_BASE_URL=http://...
GEMINI_PROXY_API_KEY=...
GEMINI_PROXY_MODEL=gemini-2.5-flash

# Gemini 官方 (备用)
GEMINI_API_KEY=...

# MiniMax (深度路径)
MINIMAX_API_KEY=sk-cp-...

# 豆包 (备用快速)
DOUBAO_API_KEY=...

# GPT 代理 (暂缓)
GPT_PROXY_BASE_URL=http://...
GPT_PROXY_MODEL=gpt-5.4

# 飞书
FEISHU_APP_ID=...
FEISHU_APP_SECRET=...
```

---

## 13. 关键工程决策与教训

| # | 问题 | 决策 | 原因 |
|---|------|------|------|
| 1 | MLX Metal 同进程并发崩溃 | TTS 拆独立进程 (UDS) | Metal command buffer 不允许并发推理 |
| 2 | VoiceDesign 音色每次随机 | Base + ref_audio (ICL) | 锁定音色，一致稳定 |
| 3 | ref_audio > 5s 输出空音频 | 限制 3.5-5s | Base ICL 的硬性限制 |
| 4 | 播放中 barge-in 误触发 | 动态阈值 + 6 帧防抖 + 文本回声过滤 | 多层区分回声和真实人声 |
| 5 | 视觉阻塞 STT 管道 | trylock(0.1s) + 超时跳过 | 保证语音响应优先 |
| 6 | ncm-cli + mpv 不稳定 | orpheus 模式 (网易云 App) | App 原生播放更可靠 |
| 7 | HuggingFace 频繁网络请求 | `HF_HUB_OFFLINE=1` | 模型已在本地，禁止版本检查 |
| 8 | 深度回答云端+本地重复 | Gemini 概述注入 MiniMax + >50% 去重 | 两层去重保证不重复 |
| 9 | 4B prefill 太慢 | 历史只给 2 轮 + 精简 prompt | 300ms 首 token 目标 |
| 10 | barge-in 后 UDS 数据错位 | 每次短连接 | 避免残留数据干扰 |
| 11 | 音乐播放时 STT 干扰 | 带通滤波 300Hz-3kHz | 只留人声频段 |

---

## 14. 启动与停止

### 启动

```bash
cd ~/mycc/2-Projects/cc-eye

# 1. 启动本地 LLM 服务 (必须)
omlx serve --model-dir ~/models --port 8000 &

# 2. 启动贾维斯主程序
#    自动 fork: TTS 子进程 + cc_audio_bridge 子进程
#    自动启动: 视觉监控线程 + 处理线程
python cc_jarvis_v3.py

# 启动后自动:
#   - 加载 TTS 模型 + 预缓存补全
#   - 加载 STT 模型 (白噪声预热消除首次 4s 延迟)
#   - 播报问候语 ("早上好。" / "下午好。" 等)
#   - 提示 "就绪，说「贾维斯」唤醒"
```

### 停止

- `Ctrl+C` → graceful shutdown
- 自动关闭：视觉引擎 → 播放器 → 音频桥 → TTS 服务子进程
- 交互日志保存到 `/tmp/cc-eye-interactions.jsonl`

---

## 15. 待优化项

| 优先级 | 项目 | 状态 |
|--------|------|------|
| P1 | TTS 预缓存自进化 (按使用频率增量更新) | TODO |
| P1 | 视觉+STT 彻底解耦 (视觉也拆独立进程) | 当前 trylock 可用 |
| P2 | 意图分类 LLM 替代 30+ 条正则 | 研究中 |
| P2 | GPT 5.4 代理稳定后重新接入深度路径 | VPS 不稳, 暂缓 |
| P2 | 状态融合接入 V3 主循环 (当前 V3 未使用) | 模块就绪, 待集成 |
| P3 | 真人参考音频 voice cloning | 研究中 |
