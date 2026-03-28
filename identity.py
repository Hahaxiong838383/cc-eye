"""
identity.py -- 人脸身份识别模块
基于 DeepFace (Facenet + OpenCV detector) 实现注册 + 识别
"""

import os
import pickle
import logging
import tempfile
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    FACE_DATA_DIR,
    FACE_MATCH_TOLERANCE,
    FACE_REGISTER_COUNT,
)

logger = logging.getLogger(__name__)

# DeepFace 配置
_MODEL_NAME = "Facenet"
_DETECTOR_BACKEND = "opencv"

# face_recognition 的 tolerance 是欧氏距离阈值（越小越严格）
# 转换为余弦相似度阈值：similarity > (1 - tolerance)
_SIMILARITY_THRESHOLD = 1.0 - FACE_MATCH_TOLERANCE


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _extract_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    从单帧中提取人脸 embedding。

    DeepFace.represent() 需要文件路径或 numpy 数组。
    返回 embedding 向量，无人脸或多人脸时返回 None。
    """
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=_MODEL_NAME,
            detector_backend=_DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if len(results) < 1:
            return None
        # 取第一张脸（通常是最大的）
        return np.array(results[0]["embedding"], dtype=np.float64)
    except (ValueError, Exception) as exc:
        logger.debug("提取 embedding 失败: %s", exc)
        return None


def _extract_all_embeddings(frame: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
    """
    提取帧中所有人脸的 embedding 和位置信息。

    Returns:
        [(embedding, facial_area), ...] -- facial_area 是 dict: {x, y, w, h}
    """
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=_MODEL_NAME,
            detector_backend=_DETECTOR_BACKEND,
            enforce_detection=False,
        )
        return [
            (np.array(r["embedding"], dtype=np.float64), r["facial_area"])
            for r in results
            if r.get("facial_area", {}).get("w", 0) > 30  # 过滤噪声检测
        ]
    except (ValueError, Exception):
        return []


class FaceIdentity:
    """人脸身份管理：注册、加载、识别"""

    def __init__(self) -> None:
        # name -> list of embedding vectors
        self._known_faces: Dict[str, List[np.ndarray]] = {}
        os.makedirs(FACE_DATA_DIR, exist_ok=True)
        self.load_known_faces()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load_known_faces(self) -> None:
        """启动时从 face_data/ 加载已注册的人脸编码"""
        self._known_faces.clear()
        for fname in os.listdir(FACE_DATA_DIR):
            if not fname.endswith(".pkl"):
                continue
            name = fname[:-4]
            path = os.path.join(FACE_DATA_DIR, fname)
            try:
                with open(path, "rb") as f:
                    encodings = pickle.load(f)
                if isinstance(encodings, list) and len(encodings) > 0:
                    self._known_faces[name] = encodings
                    logger.info("已加载人脸: %s (%d 个编码)", name, len(encodings))
            except Exception as exc:
                logger.warning("加载 %s 失败: %s", path, exc)
        logger.info("共加载 %d 个已注册身份", len(self._known_faces))

    def is_registered(self, name: str) -> bool:
        return name in self._known_faces

    def has_any_registered(self) -> bool:
        return len(self._known_faces) > 0

    def register(self, name: str, frames: List[np.ndarray]) -> bool:
        """
        从多帧中提取人脸 embedding，保存到 face_data/{name}.pkl

        Args:
            name: 用户名
            frames: BGR 帧列表（至少 FACE_REGISTER_COUNT 张）

        Returns:
            是否注册成功
        """
        encodings: List[np.ndarray] = []
        for frame in frames:
            embedding = _extract_embedding(frame)
            if embedding is not None:
                encodings.append(embedding)

        if len(encodings) < max(1, FACE_REGISTER_COUNT // 2):
            logger.warning(
                "注册失败: 只提取到 %d 个编码 (需要至少 %d)",
                len(encodings),
                FACE_REGISTER_COUNT // 2,
            )
            return False

        # 保存
        path = os.path.join(FACE_DATA_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(encodings, f)

        self._known_faces[name] = encodings
        logger.info("注册成功: %s (%d 个编码)", name, len(encodings))
        return True

    def identify(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        识别帧中的人脸身份

        Returns:
            (name, confidence) -- 未知返回 ("unknown", 0.0)
        """
        if not self._known_faces:
            return ("unknown", 0.0)

        # 提取所有人脸
        face_data = _extract_all_embeddings(frame)
        if not face_data:
            return ("unknown", 0.0)

        # 取最大的脸（距离摄像头最近）
        target_embedding, _ = max(
            face_data,
            key=lambda item: item[1]["w"] * item[1]["h"],
        )

        best_name = "unknown"
        best_confidence = 0.0

        for name, known_encs in self._known_faces.items():
            similarities = [
                _cosine_similarity(target_embedding, known_enc)
                for known_enc in known_encs
            ]
            max_sim = max(similarities)
            if max_sim > _SIMILARITY_THRESHOLD:
                # 将相似度映射为 confidence（与旧接口一致）
                confidence = max_sim
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_name = name

        return (best_name, best_confidence)

    def get_face_locations(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        返回帧中所有人脸位置 (top, right, bottom, left)

        注意：返回格式与 face_recognition 一致，便于 main.py 绘制。
        使用 OpenCV Haar Cascade 做轻量级检测，不走 DeepFace（省性能）。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )
        detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        locations = []
        for (x, y, w, h) in detections:
            # 转换为 (top, right, bottom, left) 格式
            locations.append((y, x + w, y + h, x))
        return locations
