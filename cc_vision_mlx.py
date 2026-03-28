"""
cc_vision_mlx.py — MLX 视觉模块（贾维斯的眼睛）

双模型策略：
- 快扫：Qwen2-VL-2B（~0.2s，每 10 秒）
- 精扫：Qwen2.5-VL-7B（~1s，每 60 秒或事件触发）

输出：/tmp/cc-eye-scene.json（供 LLM prompt 注入）
"""

import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

SCENE_FILE = Path("/tmp/cc-eye-scene.json")
LATEST_FRAME = Path("/tmp/cc-eye-latest.jpg")
EVENTS_FILE = Path("/tmp/cc-eye-events.jsonl")

FAST_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DETAIL_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

FAST_INTERVAL = 10    # 快扫间隔（秒）
DETAIL_INTERVAL = 60  # 精扫间隔（秒）
MAX_EVENTS = 200


class VisionEngine:
    """MLX 视觉引擎"""

    def __init__(self, mlx_lock: threading.Lock):
        self._mlx_lock = mlx_lock
        self._fast_model = None
        self._detail_model = None
        self._fast_processor = None
        self._detail_processor = None
        self._camera: Optional[cv2.VideoCapture] = None
        self._running = False
        self._last_scene = ""
        self._last_detail = ""

    def start(self):
        """启动视觉监控"""
        self._running = True

        # 打开摄像头
        self._camera = cv2.VideoCapture(0)
        if not self._camera.isOpened():
            print("[vision] 摄像头打开失败")
            return

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[vision] 摄像头就绪")

        # 后台加载模型 + 监控
        threading.Thread(target=self._monitor_loop, daemon=True, name="vision").start()

    def stop(self):
        self._running = False
        if self._camera:
            self._camera.release()

    def _load_fast(self):
        """懒加载快扫模型"""
        if self._fast_model is not None:
            return
        from mlx_vlm import load
        print("[vision] 加载快扫模型 (2B)...")
        with self._mlx_lock:
            self._fast_model, self._fast_processor = load(FAST_MODEL)
        print("[vision] 快扫模型就绪")

    def _load_detail(self):
        """懒加载精扫模型"""
        if self._detail_model is not None:
            return
        from mlx_vlm import load
        print("[vision] 加载精扫模型 (7B)...")
        with self._mlx_lock:
            self._detail_model, self._detail_processor = load(DETAIL_MODEL)
        print("[vision] 精扫模型就绪")

    def _capture(self) -> Optional[str]:
        """拍一帧，保存到 /tmp，返回路径"""
        if not self._camera or not self._camera.isOpened():
            return None
        ret, frame = self._camera.read()
        if not ret:
            return None
        cv2.imwrite(str(LATEST_FRAME), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return str(LATEST_FRAME)

    def _describe(self, image_path: str, mode: str = "fast") -> Optional[str]:
        """用视觉模型描述图片"""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        if mode == "fast":
            self._load_fast()
            model, processor = self._fast_model, self._fast_processor
            model_id = FAST_MODEL
            prompt = "用中文简要描述画面：有没有人、在做什么、桌上有什么。2句话。"
        else:
            self._load_detail()
            model, processor = self._detail_model, self._detail_processor
            model_id = DETAIL_MODEL
            prompt = "用中文详细描述画面：1)人物及其动作表情 2)桌面物品 3)环境氛围 4)任何值得注意的变化。3-4句话。"

        try:
            config = load_config(model_id)
            formatted = apply_chat_template(processor, config, prompt, num_images=1)
            with self._mlx_lock:
                output = generate(model, processor, formatted, [image_path],
                                  max_tokens=150, verbose=False)
            # output 可能是 str 或 GenerationResult
            if hasattr(output, 'text'):
                text = output.text
            elif isinstance(output, str):
                text = output
            else:
                text = str(output)
            return text.strip() if text else None
        except Exception as e:
            print(f"[vision] {mode} 描述失败: {e}")
            return None

    def _post_event(self, event_type: str, detail: str):
        """写入事件日志"""
        entry = {
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "detail": detail[:200],
        }
        try:
            with open(EVENTS_FILE, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # 滚动保留
            lines = EVENTS_FILE.read_text().strip().split("\n")
            if len(lines) > MAX_EVENTS:
                EVENTS_FILE.write_text("\n".join(lines[-MAX_EVENTS:]) + "\n")
        except Exception:
            pass

    def _update_scene(self, description: str, mode: str):
        """更新场景文件"""
        scene = {
            "ts": datetime.now().isoformat(),
            "description": description,
            "mode": mode,
        }
        SCENE_FILE.write_text(json.dumps(scene, ensure_ascii=False))

    def _monitor_loop(self):
        """视觉监控主循环"""
        print("[vision] 监控启动")
        last_fast = 0
        last_detail = 0

        # 启动时先做一次快扫
        time.sleep(2)  # 等摄像头稳定

        while self._running:
            now = time.time()

            # 快扫（每 10 秒）
            if now - last_fast >= FAST_INTERVAL:
                image = self._capture()
                if image:
                    desc = self._describe(image, "fast")
                    if desc and desc != self._last_scene:
                        self._last_scene = desc
                        self._update_scene(desc, "fast")
                        self._post_event("fast_scan", desc)
                        print(f"[vision] 快扫: {desc[:50]}")
                last_fast = now

            # 精扫（每 60 秒）
            if now - last_detail >= DETAIL_INTERVAL:
                image = self._capture()
                if image:
                    desc = self._describe(image, "detail")
                    if desc:
                        self._last_detail = desc
                        self._update_scene(desc, "detail")
                        self._post_event("detail_scan", desc)
                        print(f"[vision] 精扫: {desc[:80]}")
                last_detail = now

            time.sleep(1)

    @property
    def last_scene(self) -> str:
        return self._last_scene
