# 贾维斯 V3.5 — 开发交接

> 给新 session 或另一台电脑的 Claude Code 看的。读完就能续上。

## 当前架构（2026-03-29）

```
┌─ 耳朵 ─────────────────┐  ┌─ 眼睛 ──────────────────┐
│ Swift VP AEC 回声消除     │  │ 摄像头 640×480            │
│ → Qwen3-ASR 0.6B MLX    │  │ → Qwen2-VL-2B 快扫(10s)  │
│   (STT ~130ms)           │  │ → Qwen2.5-VL-7B 精扫(60s)│
│ + 音乐播放时带通滤波       │  │ + trylock 不阻塞 STT      │
└──────────┬──────────────┘  └────────────┬─────────────┘
           ↓                              ↓
      ┌─ 大脑（多模型协同）────────────────────────────┐
      │ 4B 本地过渡语（0.3s）+ 视觉场景注入             │
      │                                                │
      │ 简单问题 → Gemini 2.5 Flash 快速回答（~2s）     │
      │ 深度问题 → Gemini 概述（~2s）→ MiniMax 展开     │
      │                                                │
      │ 音乐操作 → 正则直接执行 / Gemini 智能调度       │
      │ + 视觉场景注入 prompt                           │
      └───────────────────┬────────────────────────────┘
                          ↓
      ┌─ 嘴巴 ──────────────────────────────────────────┐
      │ TTS 独立进程（UDS /tmp/cc-tts.sock）             │
      │ Qwen3-TTS 1.7B Base + ref_audio 音色锁定        │
      │ → 流式合成（首 chunk 290ms）+ 段间 crossfade     │
      │ + 32 条预缓存（磁盘固化）                        │
      └──────────────────────────────────────────────────┘
      ┌─ 音乐 ──────────────────────────────────────────┐
      │ ncm-cli + 网易云 App（orpheus 模式，VIP 可用）   │
      │ 正则匹配（播放/暂停/音量/切歌）+ Gemini 智能选歌 │
      │ 播放后自动隐藏 App 窗口                          │
      └──────────────────────────────────────────────────┘
```

## 启动步骤

```bash
# 1. 启动 oMLX（本地 LLM 服务）
cd ~/mycc/2-Projects/cc-eye
PATH=".venv/bin:$PATH" .venv/bin/omlx serve --model-dir ~/models --port 8000 &

# 2. 启动贾维斯（自动 fork TTS 服务子进程）
#    必须在终端前台运行（需要麦克风/摄像头权限）
PATH=".venv/bin:$PATH" .venv/bin/python cc_jarvis_v3.py
```

## 关键文件

| 文件 | 功能 |
|------|------|
| `cc_jarvis_v3.py` | 主程序：音频引擎+VAD+状态机+打断+对话+音乐降噪 |
| `cc_brain.py` | LLM 路由：4B本地 + Gemini(快速) + MiniMax(深度) + GPT(备用) |
| `cc_tts_local.py` | TTS 客户端：本地缓存 + UDS 远程合成 |
| `cc_tts_server.py` | TTS 服务端：独立进程，UDS /tmp/cc-tts.sock |
| `cc_stt_mlx.py` | STT：Qwen3-ASR MLX + 白噪声预热 |
| `cc_vision_mlx.py` | 视觉：双模型 + trylock 不阻塞 STT |
| `cc_voice_profile.py` | 音色配置（Base模型 + 参考音频路径） |
| `cc_context.py` | 上下文注入（CC_IDENTITY + status + memory） |
| `cc_tools.py` | 工具桥接：飞书消息 + 音乐播放（ncm-cli） |
| `scripts/gen_jarvis_ref.py` | 参考音频生成工具 |

## API Keys（.env，gitignore 保护）

- `MINIMAX_API_KEY` — MiniMax M2.7 深度思考
- `GEMINI_API_KEY` — Gemini 官方（备用）
- `DOUBAO_API_KEY` — 豆包（备用）
- `GEMINI_PROXY_BASE_URL` + `GEMINI_PROXY_API_KEY` — Gemini 代理（主力快速路径）
- `GPT_PROXY_BASE_URL` + `GPT_PROXY_MODEL` — GPT 5.4 代理（备用，VPS 不稳定）

## LLM 路由架构

```
简单问题（天气/视觉/闲聊）:
  4B 过渡语(0.3s) → Gemini fast(~2s, 1-3句)

深度问题（分析/方案/为什么）:
  4B 过渡语(0.3s) → Gemini deep_intro(~2s, 概述)
                   → MiniMax deep_detail(~3s, 展开，接收 Gemini 概述避免重复)

音乐操作:
  正则匹配 → 工具直接执行（暂停/音量/切歌：0延迟）
  模糊请求 → Gemini 生成搜索关键词 → ncm-cli 搜索播放
```

## Prompt 三模式

| 模式 | 谁用 | 风格 |
|------|------|------|
| fast | Gemini（简单问题） | 简洁 1-3 句，第一句就是答案 |
| deep_intro | Gemini（深度概述） | 2-3 句点明核心，结尾自然过渡 |
| deep_detail | MiniMax（深度展开） | 接着概述深入，不重复，用衔接词 |

## TTS 音色锁定

- Base 模型 + ref_audio（3.5s 参考音频）
- TTS 独立进程（UDS），不与视觉争 MLX 锁
- 流式合成（首 chunk 290ms）+ 段间 15ms crossfade
- 32 条预缓存（含音乐操作回复）
- 重新生成音色：`python scripts/gen_jarvis_ref.py`

## 音乐播放

- ncm-cli + 网易云 App（orpheus 模式）
- 播放后自动隐藏 App 窗口（osascript）
- 正则匹配 20+ 种口语表达（暂停/继续/切歌/音量/查状态）
- 模糊请求走 Gemini 智能选歌
- 音乐播放时 VAD/STT 带通滤波（300Hz-3kHz 人声频段）

## 经验教训

1. **MLX 同进程不能并发** — Metal command buffer 断言失败 crash，必须用锁或拆进程
2. **视觉 trylock** — 视觉扫描用 `acquire(timeout=0.1)`，拿不到跳过，不阻塞 STT
3. **TTS ref_audio ≤5s** — Base 模型 ICL 参考音频过长会输出空
4. **VoiceDesign 音色不一致** — 每次从 instruct 重新推理，改用 Base + ref_audio 锁定
5. **ncm-cli play + mpv 有 bug** — orpheus 模式通过 App 播放更可靠
6. **GPT 代理不稳定** — VPS SSH 常断，timeout 要设 60s，Gemini + MiniMax 更可靠
7. **流式断句** — 句号直接断 + 逗号攒满 8 字再断，避免碎片
8. **HF_HUB_OFFLINE=1** — 禁止模型版本检查，减少网络请求
9. **深度模式前序注入** — Gemini 概述传给 MiniMax prompt，避免重复

## 待优化

- P1: TTS 预缓存自进化（按使用频率增量更新）
- P1: 贾维斯视觉+语音完整打通（当前基础可用，需深化）
- P2: GPT 代理稳定后切回深度路径
- P2: 意图分类路由（替代关键词规则）
- P3: 方案 C 真人参考音频 voice cloning
