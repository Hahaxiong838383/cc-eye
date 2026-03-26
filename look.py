#!/usr/bin/env python3
"""
look.py — cc 快速看一眼（命令行工具）

用法：
  python look.py                    # 拍照 + moondream 描述
  python look.py "桌上有什么？"       # 自定义问题
  python look.py --scene             # 读取最新场景缓存（不拍照）
  python look.py --events            # 查看最近事件
"""

import json
import sys
import time
from pathlib import Path

SCENE_PATH = "/tmp/cc-eye-scene.json"
EVENTS_PATH = "/tmp/cc-eye-events.jsonl"
LATEST_FRAME = "/tmp/cc-eye-latest.jpg"


def read_scene() -> None:
    """读取最新场景描述缓存"""
    if not Path(SCENE_PATH).exists():
        print("暂无场景描述（daemon 未运行或模型未就绪）")
        return
    scene = json.loads(Path(SCENE_PATH).read_text())
    print(f"时间: {scene['ts']}")
    print(f"人脸: {scene['face_count']}")
    print(f"场景: {scene['description']}")


def read_events(n: int = 10) -> None:
    """读取最近 N 条事件"""
    if not Path(EVENTS_PATH).exists():
        print("暂无事件记录")
        return
    lines = Path(EVENTS_PATH).read_text().strip().split("\n")
    recent = lines[-n:]
    for line in recent:
        evt = json.loads(line)
        print(f"  [{evt['ts'][-8:]}] {evt['type']}: {evt.get('detail', '')}")


def quick_look(prompt: str) -> None:
    """拍一帧 + moondream 描述"""
    import cv2
    from vision_models import VisionModel

    # 如果 daemon 在跑，直接用它的帧
    if Path(LATEST_FRAME).exists():
        import os
        age = time.time() - os.path.getmtime(LATEST_FRAME)
        if age < 10:
            print("使用 daemon 最新帧...")
        else:
            # 帧太旧，自己拍一张
            print("拍摄中...")
            cap = cv2.VideoCapture(0)
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cv2.imwrite(LATEST_FRAME, frame)
            cap.release()
    else:
        print("拍摄中...")
        cap = cv2.VideoCapture(0)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.imwrite(LATEST_FRAME, frame)
        cap.release()

    vm = VisionModel(mode="fast")
    if not vm.is_available:
        print("moondream 模型不可用")
        return

    result = vm.describe_camera(prompt)
    if result:
        print(f"\n{result}")
    else:
        print("描述失败")


def main() -> None:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--scene":
            read_scene()
        elif arg == "--events":
            read_events()
        elif arg == "--help" or arg == "-h":
            print(__doc__)
        else:
            quick_look(arg)
    else:
        quick_look("Describe what you see concisely: people, objects, and what's happening.")


if __name__ == "__main__":
    main()
