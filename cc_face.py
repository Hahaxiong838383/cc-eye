"""
cc_face.py — cc 人脸识别模块（LBPH 持续学习）

基于 OpenCV LBPH（Local Binary Patterns Histograms），不需要外部模型下载。
支持持续学习：每次识别到已知人脸时自动采集新样本，越看越准。

用法：
    face = FaceRecognizer()
    face.register("chuange", "/tmp/cc-eye-latest.jpg")  # 注册川哥
    name, conf = face.recognize("/tmp/cc-eye-latest.jpg")  # 识别
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# ── 配置 ──
FACE_DB_DIR = Path(__file__).parent / "faces"
FACE_MODEL_PATH = FACE_DB_DIR / "lbph_model.yml"
FACE_REGISTRY_PATH = FACE_DB_DIR / "registry.json"
CONFIDENCE_THRESHOLD = 120.0  # LBPH 距离阈值（越小越像，<120 认为是同一人，样本多了可以调低）
MAX_SAMPLES_PER_PERSON = 50  # 每人最多保存的训练样本数
AUTO_LEARN_INTERVAL = 30     # 自动学习间隔（秒），同一人不重复采集


class FaceRecognizer:
    """LBPH 人脸识别器，支持持续学习"""

    def __init__(self):
        FACE_DB_DIR.mkdir(exist_ok=True)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        self._registry: dict = {}  # label_id -> name
        self._sample_count: dict = {}  # name -> count
        self._last_learn_time: dict = {}  # name -> timestamp
        self._trained = False

        self._load_registry()
        self._load_model()

    def _load_registry(self) -> None:
        """加载人脸注册表"""
        if FACE_REGISTRY_PATH.exists():
            data = json.loads(FACE_REGISTRY_PATH.read_text())
            self._registry = {int(k): v for k, v in data.get("registry", {}).items()}
            self._sample_count = data.get("sample_count", {})

    def _save_registry(self) -> None:
        """保存人脸注册表"""
        data = {
            "registry": self._registry,
            "sample_count": self._sample_count,
            "updated": datetime.now().isoformat(),
        }
        FACE_REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load_model(self) -> None:
        """加载已训练的 LBPH 模型"""
        if FACE_MODEL_PATH.exists() and self._registry:
            try:
                self._recognizer.read(str(FACE_MODEL_PATH))
                self._trained = True
                print(f"[cc-face] 模型已加载，已知 {len(self._registry)} 人")
            except Exception as e:
                print(f"[cc-face] 模型加载失败: {e}")
                self._trained = False

    def _detect_faces(self, image: np.ndarray) -> list:
        """检测图像中的人脸，返回 [(x,y,w,h), ...]"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        return faces, gray

    def _extract_face(self, gray: np.ndarray, rect: tuple) -> np.ndarray:
        """提取并标准化人脸区域"""
        x, y, w, h = rect
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = cv2.equalizeHist(face)
        return face

    def register(self, name: str, image_path: str, count: int = 1) -> bool:
        """
        注册一个人的脸。从图像中检测人脸并添加为训练样本。

        Args:
            name: 人名（如 "chuange"）
            image_path: 图像文件路径
            count: 当前注册的是第几张样本

        Returns:
            是否注册成功
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[cc-face] 读取图像失败: {image_path}")
            return False

        faces, gray = self._detect_faces(image)
        if len(faces) == 0:
            print("[cc-face] 未检测到人脸")
            return False

        # 取最大的脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = self._extract_face(gray, (x, y, w, h))

        # 分配 label_id
        label_id = self._get_or_create_label(name)

        # 保存样本图像
        sample_dir = FACE_DB_DIR / name
        sample_dir.mkdir(exist_ok=True)
        sample_path = sample_dir / f"sample_{count:03d}.jpg"
        cv2.imwrite(str(sample_path), face)

        self._sample_count[name] = self._sample_count.get(name, 0) + 1
        self._save_registry()

        # 重新训练
        self._retrain()

        print(f"[cc-face] 已注册 {name}（样本 #{self._sample_count[name]}）")
        return True

    def _get_or_create_label(self, name: str) -> int:
        """获取或创建人名对应的 label_id"""
        for label_id, n in self._registry.items():
            if n == name:
                return label_id
        new_id = max(self._registry.keys(), default=-1) + 1
        self._registry[new_id] = name
        return new_id

    def _retrain(self) -> None:
        """用所有样本重新训练 LBPH 模型"""
        faces = []
        labels = []

        for label_id, name in self._registry.items():
            sample_dir = FACE_DB_DIR / name
            if not sample_dir.exists():
                continue
            for img_path in sorted(sample_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (160, 160))
                    faces.append(img)
                    labels.append(label_id)

        if not faces:
            return

        self._recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        self._recognizer.train(faces, np.array(labels))
        self._recognizer.save(str(FACE_MODEL_PATH))
        self._trained = True
        print(f"[cc-face] 模型已训练（{len(faces)} 样本，{len(self._registry)} 人）")

    def recognize(self, image_path: str) -> list:
        """
        识别图像中所有人脸。

        Returns:
            [(name, confidence, rect), ...] 列表
            name 为 None 表示陌生人
        """
        if not self._trained:
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        faces, gray = self._detect_faces(image)
        results = []

        for rect in faces:
            face = self._extract_face(gray, tuple(rect))
            label_id, confidence = self._recognizer.predict(face)

            if confidence < CONFIDENCE_THRESHOLD:
                name = self._registry.get(label_id, "unknown")
                results.append((name, confidence, tuple(rect)))
            else:
                results.append((None, confidence, tuple(rect)))

        return results

    def auto_learn(self, image_path: str) -> Optional[str]:
        """
        持续学习：识别到已知人脸时，如果距上次采集超过间隔，自动添加新样本。

        Returns:
            识别到的人名，或 None
        """
        if not self._trained:
            return None

        results = self.recognize(image_path)
        now = datetime.now().timestamp()

        for name, confidence, rect in results:
            if name is None:
                continue

            last_time = self._last_learn_time.get(name, 0)
            if now - last_time < AUTO_LEARN_INTERVAL:
                return name  # 太频繁，跳过采集，但返回识别结果

            # 检查样本数量上限
            count = self._sample_count.get(name, 0)
            if count >= MAX_SAMPLES_PER_PERSON:
                return name

            # 采集新样本
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = self._extract_face(gray, rect)

            sample_dir = FACE_DB_DIR / name
            sample_dir.mkdir(exist_ok=True)
            sample_path = sample_dir / f"sample_{count+1:03d}.jpg"
            cv2.imwrite(str(sample_path), face)

            self._sample_count[name] = count + 1
            self._last_learn_time[name] = now
            self._save_registry()

            # 每 5 个新样本重新训练一次
            if (count + 1) % 5 == 0:
                self._retrain()
                print(f"[cc-face] {name} 持续学习：{count+1} 样本，模型已更新")

            return name

        return None


def register_from_camera(name: str = "chuange", n_samples: int = 5) -> bool:
    """
    从摄像头最新帧注册人脸，自动采集多张样本。

    Args:
        name: 人名
        n_samples: 采集样本数（从不同帧）
    """
    fr = FaceRecognizer()
    success = 0
    for i in range(n_samples):
        if fr.register(name, "/tmp/cc-eye-latest.jpg", count=i+1):
            success += 1
            import time
            time.sleep(1)  # 等摄像头刷新下一帧
    print(f"[cc-face] {name} 注册完成：{success}/{n_samples} 样本")
    return success > 0


if __name__ == "__main__":
    print("=== 人脸注册 ===")
    print("请面对摄像头...")
    register_from_camera("chuange", n_samples=5)
