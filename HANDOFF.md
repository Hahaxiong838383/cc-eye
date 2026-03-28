# 贾维斯全双工 MVP — 开发交接

> 给另一台电脑上的 Claude Code 看的。读完这个文件就能接着干。

## 刚完成的事（2026-03-28 07:00-09:00）

半双工→全双工语音交互架构升级，4 个新模块 + 1 个核心重构：

| 文件 | 行数 | 功能 | 状态 |
|------|------|------|------|
| `cc_vad.py` | 420 | Silero VAD ONNX（0.07ms/帧，替代能量阈值） | ✅ 导入验证通过 |
| `cc_state.py` | 231 | 6态状态机（12条转移规则） | ✅ 测试通过 |
| `cc_player.py` | 416 | InterruptablePlayer（<50ms stop，替代afplay） | ✅ 语法通过 |
| `cc_interact.py` | 676 | AudioEngine 全双工引擎（重写） | ✅ 导入通过 |
| `cc_memory_bridge.py` | 294 | 每日感知汇总→mycc记忆 | ✅ 已有 |

## 架构变化

```
旧：listen_once() 阻塞录音 → STT → LLM → afplay 阻塞播放（半双工，不可打断）
新：常驻 InputStream callback → Silero VAD → SpeechSegmenter → queue → 处理线程 → InterruptablePlayer（全双工，可打断）
```

核心：**麦克风永不关，播放器可随时停，barge-in 3帧防抖**

## 依赖

venv 里需要额外安装：
```bash
pip install pydub
```
Silero VAD ONNX 模型在首次运行时自动下载到 `.venv/models/silero_vad.onnx`。
其他依赖（onnxruntime, sounddevice, edge-tts, funasr）已在 requirements.txt 中。

## 已知问题（等待实测修复）

1. **未实测过真实对话**：代码全部通过导入验证和 AST 检查，但还没跑过完整的"说话→识别→回复→打断"流程
2. **sounddevice 必须前台终端运行**：后台进程无法获取 macOS 麦克风权限（PortAudio -9986）
3. **AEC 参考信号降采样**：`_on_player_pcm_frame` 里 24kHz→16kHz 的降采样逻辑写了但没实际喂给 AEC（需要补全）
4. **edge-tts 合成方式**：当前 `_play_sentence` 先收集所有 mp3 chunks 再解码播放（非流式），首包时间可能偏高，后续改为真正流式

## 后续待办（按优先级）

### P0：实测修 bug
- [ ] 在终端跑 `python cc_interact.py`，说"贾维斯你好"，观察状态转移日志
- [ ] 确认 VAD 不误触（空房间不触发）
- [ ] 确认 STT 识别正确
- [ ] 确认 TTS 播放正常（sounddevice OutputStream 能出声）
- [ ] 测试 barge-in（播放中说话打断）

### P1：功能增强
- [ ] 摄像头+语音联合理解（"这是什么" → 截帧 + Vision API）
- [ ] 情感语调 TTS（Fish Speech / ChatTTS 替代 edge-tts）
- [ ] 更自然的 turn-taking（语义级 endpointing）

### P2：体验优化
- [ ] 去掉唤醒词（Porcupine 音频级检测）
- [ ] 声纹识别（区分用户）
- [ ] AEC 块级升级（当前逐样本循环太慢）

## 启动命令

```bash
cd ~/mycc/2-Projects/cc-eye
source .venv/bin/activate
python cc_interact.py
```

## 关键代码入口

- `cc_interact.py:88` — `AudioEngine.__init__`
- `cc_interact.py:210` — `_audio_callback`（32ms/帧，VAD + AEC + barge-in）
- `cc_interact.py:295` — `_handle_segment`（STT → 唤醒词 → LLM → TTS）
- `cc_interact.py:383` — `_process_speech`（流式 LLM → 句级 TTS 播放）
- `cc_vad.py:74` — `SileroVAD` 类
- `cc_state.py:57` — `TRANSITIONS` 转移规则表
- `cc_player.py:66` — `InterruptablePlayer` 类
