"""
vision_models.py — 本地多模态视觉模型接口

通过 ollama API 调用本地多模态模型（感知层），
cc 拿到文字描述后做分析和决策（思考层）。

支持模型：moondream（快速）、minicpm-v（均衡）
"""

import base64
import json
import logging
import time
from typing import Optional, Dict
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OLLAMA_API = "http://localhost:11434/api/generate"


class VisionModel:
    """本地多模态视觉模型接口"""

    MODELS = {
        "fast": "moondream",          # 1.6B，最快
        "balanced": "minicpm-v",      # 8B，均衡
    }

    # 模型原生输入尺寸（长边最大值）— 超过此值的像素纯属浪费
    MAX_DIMS = {
        "moondream": 384,             # SigLIP encoder 378×378
        "minicpm-v": 448,             # InternViT 448×448
    }

    # 发给模型的 JPEG 质量 — 低于此值影响识别
    JPEG_QUALITY = {
        "moondream": 65,
        "minicpm-v": 75,
    }

    def __init__(self, mode: str = "fast"):
        self.model = self.MODELS.get(mode, self.MODELS["fast"])
        self._available = self._check_available()
        if self._available:
            logger.info(f"VisionModel 就绪: {self.model}")
        else:
            logger.warning(f"VisionModel 不可用: {self.model}（ollama 未运行或模型未下载）")

    def _check_available(self) -> bool:
        """检查 ollama 和模型是否可用"""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code != 200:
                return False
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            return self.model in models
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def _prepare_image(self, image_path: str) -> Optional[str]:
        """Resize 到模型原生尺寸 + 降 JPEG 质量，返回 base64。

        moondream 内部用 378×378，minicpm-v 用 448×448。
        发 1280×720 的图过去全是浪费 — 编码慢、传输慢、模型还得再缩。
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None

            max_dim = self.MAX_DIMS.get(self.model, 384)
            h, w = img.shape[:2]

            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            quality = self.JPEG_QUALITY.get(self.model, 70)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as e:
            logger.error(f"图片预处理失败: {e}")
            # 降级：原始文件直接读取
            try:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

    def describe_image(self, image_path: str, prompt: str = "Describe what you see in this image in detail.") -> Optional[str]:
        """
        让本地模型描述一张图片。

        Args:
            image_path: 图片文件路径
            prompt: 提示词

        Returns:
            模型返回的文字描述，或 None（不可用时）
        """
        if not self._available:
            self._available = self._check_available()
            if not self._available:
                return None

        try:
            img_b64 = self._prepare_image(image_path)
            if not img_b64:
                return None

            start = time.time()
            resp = requests.post(
                OLLAMA_API,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=60,
            )

            if resp.status_code != 200:
                logger.error(f"ollama API 错误: {resp.status_code}")
                return None

            result = resp.json().get("response", "")
            elapsed = time.time() - start
            logger.info(f"VisionModel [{self.model}] 耗时 {elapsed:.1f}s")
            return result

        except requests.Timeout:
            logger.error("ollama API 超时")
            return None
        except Exception as e:
            logger.error(f"VisionModel 错误: {e}")
            return None

    def describe_camera(self, prompt: str = "Describe what you see. Focus on people, objects, and any changes.") -> Optional[str]:
        """直接描述摄像头 daemon 的最新帧"""
        latest = "/tmp/cc-eye-latest.jpg"
        if not Path(latest).exists():
            return None
        return self.describe_image(latest, prompt)


def quick_look(prompt: str = "What do you see?") -> Optional[str]:
    """快速看一眼摄像头画面（便捷函数）"""
    vm = VisionModel(mode="fast")
    return vm.describe_camera(prompt)


if __name__ == "__main__":
    # 测试：让本地模型看摄像头最新帧
    print("测试 VisionModel...")
    vm = VisionModel(mode="fast")
    if vm.is_available:
        result = vm.describe_camera("Describe this office scene in detail. What objects and furniture can you see?")
        if result:
            print(f"\n本地模型说：\n{result}")
        else:
            print("描述失败")
    else:
        print(f"模型 {vm.model} 不可用，请确认 ollama serve 已运行且模型已下载")
