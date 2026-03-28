"""
cc_voice_profile.py — 贾维斯声音档案（VoiceDesign 描述 + 配置）

锁定版本：2026-03-28 v4
"""

# 贾维斯声音描述（英文，给 Qwen3-TTS VoiceDesign 用）
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

# TTS 模型配置
VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
CUSTOM_VOICE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

# 降级音色（VoiceDesign 不可用时用 CustomVoice）
FALLBACK_SPEAKER = "aiden"
