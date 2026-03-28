"""
生成贾维斯参考音频 → 提取 speaker embedding → 保存为 .npy

用法：
    python scripts/gen_jarvis_ref.py

流程：
    1. 用 VoiceDesign 模型生成 5 段不同文本的音频
    2. 拼接为一段完整参考音频（~15-20秒）
    3. 用 Base 模型的 speaker encoder 提取 embedding
    4. 保存 embedding (.npy) + 参考音频 (.wav)
"""

import sys
import numpy as np
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from cc_voice_profile import JARVIS_VOICE_INSTRUCT, VOICE_DESIGN_MODEL

OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "voice"
REF_AUDIO_PATH = OUTPUT_DIR / "jarvis_ref.wav"
EMBEDDING_PATH = OUTPUT_DIR / "jarvis_embedding.npy"

# 用于生成参考音频的文本（覆盖不同语气：陈述、建议、问候、警告）
REF_TEXTS = [
    "所有系统运行正常，随时待命。",
    "这个方案整体思路没问题，但第二步风险有点大，换成异步会稳很多。",
    "早上好，今天天气不错，适合出门走走。",
    "检测到异常，建议你看一下。",
    "放心，交给我处理。",
]

SAMPLE_RATE = 24000


def generate_ref_audio():
    """用 VoiceDesign 生成多段参考音频并拼接"""
    from mlx_audio.tts.utils import load_model

    print("[1/3] 加载 VoiceDesign 模型...")
    vd_model = load_model(VOICE_DESIGN_MODEL)

    all_audio = []
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)  # 0.5s 静音间隔

    for i, text in enumerate(REF_TEXTS):
        print(f"  生成 [{i+1}/{len(REF_TEXTS)}]: {text}")
        results = list(vd_model.generate_voice_design(
            text=text,
            language="Chinese",
            instruct=JARVIS_VOICE_INSTRUCT,
        ))
        audio = np.array(results[0].audio, dtype=np.float32)
        # 归一化
        peak = np.abs(audio).max()
        if peak > 0.001:
            audio = audio * (0.95 / peak)
        all_audio.append(audio)
        all_audio.append(silence)

    # 拼接
    ref_audio = np.concatenate(all_audio)
    duration = len(ref_audio) / SAMPLE_RATE
    print(f"  参考音频: {duration:.1f}s, {len(ref_audio)} samples")

    return ref_audio


def extract_and_save(ref_audio: np.ndarray):
    """用 Base 模型提取 speaker embedding 并保存"""
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    print("[2/3] 加载 Base 模型（提取 speaker embedding）...")
    base_model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    print("  提取 embedding...")
    audio_mx = mx.array(ref_audio)
    embedding = base_model.extract_speaker_embedding(audio_mx, sr=SAMPLE_RATE)
    embedding_np = np.array(embedding)
    print(f"  Embedding shape: {embedding_np.shape}")  # 应该是 [1, 1024]

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(EMBEDDING_PATH), embedding_np)
    print(f"  Embedding 已保存: {EMBEDDING_PATH}")

    # 保存参考音频为 wav
    import wave
    pcm_int16 = (ref_audio * 32767).astype(np.int16)
    with wave.open(str(REF_AUDIO_PATH), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    print(f"  参考音频已保存: {REF_AUDIO_PATH}")

    return embedding_np


def verify(embedding_np: np.ndarray):
    """验证：用 base 模型 + ref_audio 合成一句话，确认音色"""
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    print("[3/3] 验证音色一致性...")
    base_model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    # 读取参考音频
    ref_audio_mx = mx.array(np.load(str(REF_AUDIO_PATH).replace(".wav", "_raw.npy")) if False else
                            np.array([0.0]))  # placeholder

    # 用 ref_audio wav 文件验证
    test_text = "你好，我是贾维斯，所有系统运行正常。"
    print(f"  合成测试: {test_text}")

    results = list(base_model.generate(
        text=test_text,
        ref_audio=str(REF_AUDIO_PATH),
        ref_text="所有系统运行正常，随时待命。",
        lang_code="auto",
    ))

    test_audio = np.array(results[0].audio, dtype=np.float32)
    test_path = OUTPUT_DIR / "jarvis_verify.wav"

    import wave
    pcm_int16 = (test_audio * 32767).astype(np.int16)
    with wave.open(str(test_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    print(f"  验证音频: {test_path}")
    print("  请听两个文件对比音色是否一致：")
    print(f"    参考: {REF_AUDIO_PATH}")
    print(f"    验证: {test_path}")


if __name__ == "__main__":
    ref_audio = generate_ref_audio()
    embedding = extract_and_save(ref_audio)
    verify(embedding)
    print("\n完成！如果音色满意，运行 cc-eye 即可自动使用锁定的音色。")
