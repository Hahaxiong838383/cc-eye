# 贾维斯 V3.3 — 开发交接

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
      │ 3. Gemini 2.5 FL / MiniMax M2.7 云端            │
      │ + 视觉场景注入 prompt                           │
      └───────────────────┬────────────────────────────┘
                          ↓
      ┌─ 嘴巴 ──────────────────────────────────────────┐
      │ Qwen3-TTS 1.7B VoiceDesign（定制贾维斯声音）     │
      │ → 常驻 OutputStream（可打断，无破音）             │
      │ + 73 条预缓存（磁盘固化，启动秒加载）            │
      └──────────────────────────────────────────────────┘
```

## 启动步骤

```bash
# 1. 启动 oMLX（本地 LLM 服务）
cd ~/mycc/2-Projects/cc-eye
PATH=".venv/bin:$PATH" .venv/bin/omlx serve --model-dir ~/models --port 8000 &

# 2. 启动贾维斯
PATH=".venv/bin:$PATH" .venv/bin/python cc_jarvis_v3.py
```

## 关键文件

| 文件 | 功能 |
|------|------|
| `cc_jarvis_v3.py` | 主程序：音频引擎+VAD+状态机+打断+对话 |
| `cc_brain.py` | LLM 路由：本地4B + Gemini + MiniMax |
| `cc_tts_local.py` | TTS：VoiceDesign 1.7B + 预缓存 |
| `cc_stt_mlx.py` | STT：Qwen3-ASR MLX |
| `cc_vision_mlx.py` | 视觉：Qwen2-VL-2B + Qwen2.5-VL-7B |
| `cc_voice_profile.py` | 贾维斯声音描述（VoiceDesign instruct） |
| `cc_audio_bridge.swift` | Swift VP AEC 桥接 |
| `cc_audio_engine.py` | Python 端桥接封装 |
| `cc_audio_out.py` | 常驻 OutputStream 播放器 |

## API Keys（.env，gitignore 保护）

- `MINIMAX_API_KEY` — MiniMax M2.7 深度思考 + 联网搜索
- `GEMINI_API_KEY` — Gemini 2.5 Flash-Lite 快速响应

## 本地模型（~/models/ + HF 缓存）

| 模型 | 用途 | 位置 |
|------|------|------|
| Qwen3.5-4B-MLX-4bit | 本地 LLM（oMLX） | ~/models/ |
| Qwen3.5-9B-4bit | 备用 LLM（oMLX） | ~/models/ |
| Qwen3-ASR-0.6B-8bit | STT | HF 缓存 |
| Qwen3-TTS-1.7B-VoiceDesign-8bit | TTS | HF 缓存 |
| Qwen3-TTS-1.7B-CustomVoice-8bit | TTS 降级 | HF 缓存 |
| Qwen2-VL-2B-Instruct-4bit | 视觉快扫 | HF 缓存 |
| Qwen2.5-VL-7B-Instruct-4bit | 视觉精扫 | HF 缓存 |

## 三层 LLM 路由

```
所有查询：
  → 预缓存过渡词秒播（0ms）
  → 4B 本地流式（300ms 首 token）
  → 需要联网/深度时 Gemini/MiniMax 并行补充
```

- 4B prompt：精简 few-shot（10字以内，像朋友聊天）
- 云端 prompt：完整贾维斯人格 + 记忆 + 场景 + 能力

## VP AEC 回声消除

- Swift `cc_audio_bridge` 启用 Voice Processing IO
- ducking 设为 `.min`（最小音量衰减）
- 回声从 0.05 降到 ~0.009（消 80%）
- barge-in 纯能量动态阈值（播放能量 × 0.3 + 0.03）

## TTS 预缓存

- 73 条常用短句（≥3 字），VoiceDesign 1.7B 生成
- 固化到 `.venv/cache/tts_cache.npz`
- 首次启动合成（几分钟），之后秒加载
- 换音色需删缓存重新生成

## 已知问题

1. **回声消除不完美**：VP 消 80%，剩余靠文本级过滤 + 动态阈值
2. **本地和云端衔接**：偶有断档，过渡词填补
3. **TTS 音色微差**：VoiceDesign 1.7B 每次合成有轻微随机性
4. **视觉 GPU 争抢**：视觉模型和 STT/TTS 共用 MLX 锁串行

## 待优化（P2）

- TTS 预缓存自进化（按使用频率增量更新）
- 意图分类路由（替代字数规则）
- 视觉主动交互（人到达/离开触发问候）
- 方案 C：中文 S2S 模型跟踪（PersonaPlex 等）
- oMLX KV 缓存优化（system prompt 固定部分缓存）
