"""本地 AI 分割(isnet-anime ONNX):为剪影生成角色蒙版。

设计(见 .workbuddy/plans/2026-10-10-ai-matting-and-ui.md):
- 完全本地、离线;`onnxruntime` 为可选依赖,缺失或任何异常时静默降级(返回 None)。
- 预处理:等比缩放到长边 1024、居中 pad、/255、减 ImageNet 均值(实测优于官方样例的 /255)。
- 输出:与原图同尺寸的 uint8 soft 蒙版(0~255),交给 imaging 阈值化与清理。
- 模型文件:程序目录 `models/isnetis.onnx`(176MB,不随 git 分发,见 README)。
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from .paths import base_dir

log = logging.getLogger(__name__)

MODEL_NAME = "isnetis.onnx"
INPUT_SIZE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)


def default_model_path() -> Path:
    """模型默认位置:程序目录/models/isnetis.onnx。"""
    return base_dir() / "models" / MODEL_NAME


class Segmenter:
    """isnet-anime 推理封装(懒加载,单进程内复用 session)。"""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._session = None
        self._failed = False

    @property
    def is_ready(self) -> bool:
        return self._ensure_session()

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if self._failed:
            return False
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # 只留错误级日志
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            return True
        except Exception:  # noqa: BLE001 - 依赖缺失/模型损坏都走降级
            log.warning("AI 模型不可用,已降级为传统抠图:%s", self.model_path, exc_info=True)
            self._failed = True
            return False

    def predict(self, rgb: np.ndarray) -> np.ndarray | None:
        """返回与原图同尺寸的 uint8 蒙版(0~255);不可用或异常时返回 None。"""
        if not self._ensure_session():
            return None
        try:
            rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3])
            h0, w0 = rgb.shape[:2]
            x, meta = _letterbox(rgb, INPUT_SIZE)
            name = self._session.get_inputs()[0].name
            pred = self._session.run(None, {name: x})[0][0][0]
            return _restore(np.asarray(pred, np.float32), meta, h0, w0)
        except Exception:  # noqa: BLE001 - 推理异常同样降级
            log.warning("AI 推理失败,已降级为传统抠图", exc_info=True)
            return None


def create_segmenter(config, model_path: Path | None = None) -> Segmenter | None:
    """按配置与模型文件创建 Segmenter;不满足条件时返回 None(调用方走传统路径)。"""
    if not getattr(config, "ai_matting", True):
        return None
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_file():
        return None
    return Segmenter(path)


def _letterbox(rgb: np.ndarray, size: int = INPUT_SIZE):
    """等比缩放 + 居中 pad + 归一化。

    返回 (x, meta):
    - x: float32 (1, 3, size, size),图像区域已 /255 减均值,pad 区域为 0
    - meta: (h, w, ph, pw) 缩放后尺寸与 pad 总量(上下/左右各一半)
    """
    img = rgb.astype(np.float32) / 255.0 - _MEAN
    h0, w0 = img.shape[:2]
    if h0 > w0:
        h, w = size, max(1, int(round(size * w0 / h0)))
    else:
        h, w = max(1, int(round(size * h0 / w0))), size
    ph, pw = size - h, size - w
    canvas = np.zeros((size, size, 3), np.float32)
    top, left = ph // 2, pw // 2
    canvas[top:top + h, left:left + w] = cv2.resize(img, (w, h))
    x = np.transpose(canvas, (2, 0, 1))[None]
    return x, (h, w, ph, pw)


def _restore(pred: np.ndarray, meta, h0: int, w0: int) -> np.ndarray:
    """裁剪 pad、放大回原尺寸、转 uint8 蒙版。"""
    h, w, ph, pw = meta
    top, left = ph // 2, pw // 2
    pred = pred[top:top + h, left:left + w]
    mask = cv2.resize(pred, (w0, h0))
    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
