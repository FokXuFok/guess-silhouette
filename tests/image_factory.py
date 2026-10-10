"""合成测试图生成工具：把"角色"形状（头 + 身体）画到不同背景上，并给出真实蒙版。"""
import numpy as np

SUBJECT_COLOR = (200, 60, 90)


def make_blob(bg="solid", size=(640, 480), seed=0):
    """生成合成图。

    返回 (rgb uint8 HxWx3, gt bool HxW)；bg 可选 solid / gradient / noise。
    """
    rng = np.random.default_rng(seed)
    w, h = size
    if bg == "solid":
        rgb = np.full((h, w, 3), (245, 245, 240), np.uint8)
    elif bg == "gradient":
        top = np.array([40, 70, 130], np.float32)
        bottom = np.array([10, 18, 45], np.float32)
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        col = top[None, None, :] * (1.0 - t) + bottom[None, None, :] * t
        rgb = np.repeat(col.astype(np.uint8), w, axis=1)
    elif bg == "noise":
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        raise ValueError(bg)

    yy, xx = np.mgrid[0:h, 0:w]
    body = (((xx - w * 0.5) / (w * 0.16)) ** 2 + ((yy - h * 0.62) / (h * 0.30)) ** 2) <= 1
    head = (((xx - w * 0.5) / (w * 0.10)) ** 2 + ((yy - h * 0.24) / (h * 0.16)) ** 2) <= 1
    gt = body | head
    rgb[gt] = SUBJECT_COLOR
    return rgb, gt


def iou(a, b):
    """两个布尔蒙版的交并比。"""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    return float((a & b).sum()) / max(int((a | b).sum()), 1)