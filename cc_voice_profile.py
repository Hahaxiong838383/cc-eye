"""
cc_voice_profile.py — 贾维斯声音档案

锁定版本：2026-03-29 v5

音色锁定方案：
    1. VoiceDesign 生成参考音频 → assets/voice/jarvis_ref.wav
    2. Base 模型 speaker encoder 提取 embedding → assets/voice/jarvis_embedding.npy
    3. 运行时用 Base 模型 + ref_audio 合成，音色固定不漂移

重新生成音色：python scripts/gen_jarvis_ref.py
"""

from pathlib import Path

# ── 音色资源路径 ──
_VOICE_DIR = Path(__file__).parent / "assets" / "voice"
REF_AUDIO_PATH = _VOICE_DIR / "jarvis_ref.wav"
EMBEDDING_PATH = _VOICE_DIR / "jarvis_embedding.npy"

# ── 模型配置 ──
# 主力：Base 模型（有 speaker encoder，支持 ref_audio 音色锁定）
BASE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

# 保留 VoiceDesign 配置（仅用于重新生成参考音频）
VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

# VoiceDesign instruct（仅 gen_jarvis_ref.py 使用）
JARVIS_VOICE_INSTRUCT = (
    "A deep, warm, mature male voice with expressive intonation and natural pitch variation. "
    "Slightly faster than normal speaking pace, energetic but controlled. "
    "Clear, clean, calm yet emotionally engaged, not flat or monotone. "
    "Elegant and reliable with a subtle British gentleman quality. "
    "The timbre carries a noticeable electronic and synthetic texture, "
    "like the core AI of an advanced futuristic system, with distinct metallic resonance, "
    "crisp digital edges, and precision-engineered articulation. "
    "More electronic than human, but warm and emotionally present. "
    "The delivery has natural rises and falls in pitch, emphasizing key words with subtle intensity shifts, "
    "creating a sense of rhythm and life in speech. "
    "A hint of dry wit and understated humor, as if quietly amused by its own brilliance. "
    "Emotional range includes: pride and quiet satisfaction when completing tasks, "
    "genuine warmth when greeting, subtle excitement and positive energy when reporting good results, "
    "concern when warning, and confident reassurance when things are under control. "
    "After accomplishing something, the voice carries a sense of fulfillment and understated self-congratulation, "
    "like a professional who takes genuine pleasure in doing excellent work. "
    "The overall feel is a charming, brilliant AI butler who genuinely cares, "
    "takes pride in every task, and radiates positive competent energy. "
    "Speaking Chinese."
)

# 参考音频对应的文本（用于 ICL 模式）
REF_TEXT = "所有系统运行正常，随时待命。"
