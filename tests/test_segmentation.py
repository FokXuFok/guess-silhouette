"""AI 分割模块(Segmenter)测试。

不依赖真模型的部分始终运行;真模型用例在 models/isnetis.onnx 存在时才跑。
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.config import Config
from app.segmentation import (
    Segmenter,
    _letterbox,
    _restore,
    create_segmenter,
    default_model_path,
)

MODEL = default_model_path()


# ---------------- 创建与降级 ----------------

def test_create_segmenter_disabled_by_config():
    assert create_segmenter(Config(ai_matting=False), model_path=MODEL) is None


def test_create_segmenter_missing_model_returns_none(tmp_path):
    assert create_segmenter(Config(), model_path=tmp_path / "不存在.onnx") is None


def test_create_segmenter_with_existing_model(tmp_path):
    m = tmp_path / "m.onnx"
    m.write_bytes(b"dummy")
    seg = create_segmenter(Config(), model_path=m)
    assert isinstance(seg, Segmenter)


def test_predict_without_onnxruntime_degrades(tmp_path):
    """onnxruntime 不可用时,predict 返回 None(is_ready False),不抛异常。"""
    m = tmp_path / "m.onnx"
    m.write_bytes(b"dummy")
    seg = Segmenter(m)
    rgb = np.zeros((32, 32, 3), np.uint8)
    with patch.dict(sys.modules, {"onnxruntime": None}):
        assert seg.predict(rgb) is None
    assert seg.is_ready is False


def test_predict_with_broken_model_file_degrades(tmp_path):
    """模型文件损坏(不是有效 onnx)时同样降级,不抛异常。"""
    m = tmp_path / "bad.onnx"
    m.write_bytes(b"this is not an onnx file")
    seg = Segmenter(m)
    rgb = np.zeros((32, 32, 3), np.uint8)
    assert seg.predict(rgb) is None
    assert seg.is_ready is False


# ---------------- letterbox 纯函数 ----------------

def test_letterbox_width_dominant_shape_and_pad():
    rgb = np.full((100, 200, 3), 128, np.uint8)
    x, meta = _letterbox(rgb, size=64)
    assert x.shape == (1, 3, 64, 64)
    h, w, ph, pw = meta
    assert (h, w) == (32, 64)          # 长边缩到 64,短边等比
    assert (ph, pw) == (32, 0)         # 上下各 16
    # pad 区域为 0(未归一化的填充)
    assert float(np.abs(x[0, :, 0, :]).max()) == 0.0
    # 图像区域已归一化(减过均值,B 通道 = 128/255-0.406 ≈ 0.096)
    center = x[0, :, 32, 32]
    assert np.all(np.abs(center) < 0.15)


def test_letterbox_height_dominant_center():
    rgb = np.full((200, 100, 3), 128, np.uint8)
    x, meta = _letterbox(rgb, size=64)
    h, w, ph, pw = meta
    assert (h, w) == (64, 32)
    assert (ph, pw) == (0, 32)
    assert float(np.abs(x[0, :, :, 0]).max()) == 0.0
    assert float(np.abs(x[0, :, 5, 32]).max()) > 0.0  # 图像列非零


def test_restore_resizes_and_clips():
    meta = (64, 32, 0, 32)
    pred = np.zeros((64, 32), np.float32)
    pred[32:, :] = 0.8
    out = _restore(pred, meta, h0=200, w0=100)
    assert out.shape == (200, 100)
    assert out.dtype == np.uint8
    assert out.max() <= 255
    assert out[:95].max() == 0            # 上半远离边界区仍是 0
    assert out[150:].min() >= int(0.79 * 255) - 2  # 下半纯 0.8 区


# ---------------- 真模型(模型存在时) ----------------

ROLE_ASSET = Path(__file__).resolve().parent / "assets" / "anime_role.png"


@pytest.mark.skipif(not MODEL.is_file(), reason="models/isnetis.onnx 不存在")
def test_predict_real_model_plausible_mask():
    """真实动漫素材上:能识别主体,覆盖率与中心命中合理。"""
    from PIL import Image

    rgb = np.asarray(Image.open(ROLE_ASSET).convert("RGB"))
    seg = Segmenter(MODEL)
    mask = seg.predict(rgb)
    assert mask is not None, "真模型应能返回蒙版"
    assert seg.is_ready
    assert mask.shape == rgb.shape[:2]
    assert mask.dtype == np.uint8
    cover = float((mask >= 128).mean())
    assert 0.2 < cover < 0.95, f"覆盖率异常:{cover}"
    h, w = mask.shape[:2]
    assert mask[h // 2, w // 2] >= 128, "图片中心应命中主体"
