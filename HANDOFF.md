# 贾维斯 V3.4 — 开发交接

> 给新 session 或另一台电脑的 Claude Code 看的。读完就能续上。

## 当前架构（2026-03-29）

```
┌─ 耳朵 ─────────────────┐  ┌─ 眼睛 ──────────────────┐
│ Swift VP AEC 硬件回声消除 │  │ 摄像头 640×480            │
│ → Qwen3-ASR 0.6B MLX    │  │ → Qwen2-VL-2B 快扫(10s)  │
│   (STT 130ms)            │  │ → Qwen2.5-VL-7B 精扫(60s)│
└──────────┬──────────────┘  └────────────┬─────────────┘
           ↓                              ↓
      ┌─ 大脑（三层路由）──────────────────────────────┐
      │ 1. 预缓存过渡词（0ms）                          │
      │ 2. oMLX Qwen3.5-4B 本地（300ms 首 token）      │
      │ 3. 豆包 seed-2.0-lite（快速）/ MiniMax M2.7     │
      │ + 视觉场景注入 prompt                           │
      └───────────────────┬────────────────────────────┘
                          ↓
      ┌─ 嘴巴 ──────────────────────────────────────────┐
      │ Qwen3-TTS 1.7B Base + ref_audio（音色锁定）      │
      │ → 常驻 OutputStream（可打断，无破音）             │
      │ + 73 条预缓存（磁盘固化，启动秒加载）            │
      └──────────────────────────────────────────────────┘
```

## 启动步骤

```bash
# 1. 启动 oMLX（本地 LLM 服务）
cd ~/mycc/2-Projects/cc-eye
PATH=".venv/bin:$PATH" .venv/bin/omlx serve --model-dir ~/models --port 8000 &

# 2. 启动贾维斯（会自动 fork TTS 服务子进程）
#    必须在终端前台运行（需要麦克风/摄像头权限）
PATH=".venv/bin:$PATH" .venv/bin/python cc_jarvis_v3.py

# 启动后自动创建的进程：
#   - cc_jarvis_v3.py  (主进程：STT + 视觉 + 对话逻辑)
#   - cc_tts_server.py (TTS 独立进程：UDS /tmp/cc-tts.sock)
#   - omlx serve       (LLM 服务：HTTP localhost:8000)
```

## 关键文件

| 文件 | 功能 |
|------|------|
| `cc_jarvis_v3.py` | 主程序：音频引擎+VAD+状态机+打断+对话 |
| `cc_brain.py` | LLM 路由：4B本地 + 豆包(快速) + MiniMax(深度) |
| `cc_tts_local.py` | TTS：Base 1.7B + ref_audio 音色锁定 + 预缓存 |
| `cc_stt_mlx.py` | STT：Qwen3-ASR MLX |
| `cc_vision_mlx.py` | 视觉：Qwen2-VL-2B + Qwen2.5-VL-7B + pause/resume |
| `cc_voice_profile.py` | 音色配置（Base模型 + 参考音频路径） |
| `cc_context.py` | 上下文注入（CC_IDENTITY + status + memory） |
| `cc_audio_bridge.swift` | Swift VP AEC 桥接 |
| `scripts/gen_jarvis_ref.py` | 参考音频生成工具 |

## API Keys（.env，gitignore 保护）

- `DOUBAO_API_KEY` — 豆包 seed-2.0-lite 云端快速路径
- `MINIMAX_API_KEY` — MiniMax M2.7 深度思考 + 联网搜索
- `GEMINI_API_KEY` — Gemini 2.5 Flash-Lite（备用，当前未启用）

## 本地模型（~/models/ + HF 缓存）

| 模型 | 用途 | 位置 |
|------|------|------|
| Qwen3.5-4B-MLX-4bit | 本地 LLM（oMLX） | ~/models/ |
| Qwen3.5-9B-4bit | 备用 LLM（oMLX） | ~/models/ |
| Qwen3-ASR-0.6B-8bit | STT | HF 缓存 |
| Qwen3-TTS-1.7B-Base-8bit | TTS（主力，音色锁定） | HF 缓存 |
| Qwen3-TTS-1.7B-VoiceDesign-8bit | TTS（仅生成参考音频用） | HF 缓存 |
| Qwen2-VL-2B-Instruct-4bit | 视觉快扫 | HF 缓存 |
| Qwen2.5-VL-7B-Instruct-4bit | 视觉精扫 | HF 缓存 |

## Prompt 设计（V3.4）

三模型共享一个人格面具，4B 接球，云端回球：

- **4B `_LOCAL_SYSTEM`**：1-3句话 + 4个 few-shot，做不到的事说"我看看"
- **云端 `CC_IDENTITY`**：核心衔接句"你刚才简短接了一句话，现在把完整回答告诉川哥"
- **统一禁用词**：markdown/列表/"作为AI"/"根据搜索结果"/"我无法"
- **统一称呼**："你"，不用"您"，打招呼时用"川哥"

## TTS 音色锁定

- **原理**：VoiceDesign 生成参考音频 → Base 模型 + ref_audio 固定 speaker embedding
- **资源**：`assets/voice/jarvis_ref.wav`（3.5s）+ `jarvis_embedding.npy`
- **缓存**：`.venv/cache/tts_cache_base.npz`（73条）
- **重新生成**：`python scripts/gen_jarvis_ref.py`
- **⚠️ 注意**：参考音频不能超过 5s，否则 Base ICL 输出为空

## 三层 LLM 路由

```
所有查询：
  → 预缓存过渡词秒播（0ms）
  → 4B 本地流式（300ms 首 token，2轮历史）
  → 需要联网/深度时豆包/MiniMax 并行补充
```

## xray 直连规则

以下域名走国内直连，不经代理：
- `minimaxi.com` — MiniMax API
- `volces.com` — 豆包 API
- `volcengine.com` — 火山引擎

## VP AEC 回声消除

- Swift `cc_audio_bridge` 启用 Voice Processing IO
- ducking 设为 `.min`（最小音量衰减）
- 回声从 0.05 降到 ~0.009（消 80%）
- barge-in 纯能量动态阈值（播放能量 × 0.3 + 0.03）

## 已知问题

1. **MLX 锁竞争**：视觉+TTS 同进程共享锁，对话时 TTS 可能等视觉释放锁（当前临时方案：pause/resume）
2. **回声消除不完美**：VP 消 80%，剩余靠文本级过滤 + 动态阈值
3. **本地和云端衔接**：偶有断档，过渡词填补

## 待优化

- **P1**: TTS 独立进程（方案 A：HTTP 服务，解决 MLX 锁竞争，让视觉和语音真正并行）
- P2: TTS 预缓存自进化（按使用频率增量更新）
- P2: 意图分类路由（替代字数规则）
- P2: 视觉主动交互（人到达/离开触发问候）
- P3: 方案 C 真人参考音频 voice cloning
- P3: 中文 S2S 模型跟踪
