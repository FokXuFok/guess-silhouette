"""图像处理核心:分档抠图(AI/透明/纯色/渐变/GrabCut)+ 剪影、揭晓图生成。

完全本地运行,不使用联网 AI;AI 档使用本地 ONNX 模型(可选,缺失时自动降级)。
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageOps

from .config import Config

TIER_AI = 0        # AI 模型(isnet):可用时优先,失败静默降级到下面四档
TIER_ALPHA = 1     # 透明背景 PNG:直接读 alpha
TIER_SOLID = 2     # 纯色背景:边缘估计 + Lab 色距 + 只去边缘连通背景
TIER_GRADIENT = 3  # 渐变背景:边缘拟合背景模型后逐像素比较
TIER_COMPLEX = 4   # 复杂背景:GrabCut

WORK_MAX_DIM = 2400          # 处理时最长边上限（保证渲染分辨率 ≥ 1080p）
GRABCUT_MAX_DIM = 512        # 跑 GrabCut 的最长边上限（耗时随分辨率急剧上升）
SOLID_RESIDUAL_TOL = 5.0     # 纯色判定：边界颜色中位残差（Lab ΔE）
GRADIENT_RESIDUAL_TOL = 8.0  # 渐变判定：拟合后残差阈值
MIN_ALPHA_FRACTION = 0.01    # alpha 图判定为"透明背景"的最小透明像素占比
DARK_SUBJECT_L = 55.0        # 主体亮度低于该值视为深色，揭晓时加描边
AI_THRESHOLD = 0.4           # AI 蒙版二值化阈值（实测优于 0.5:浅色区域不被切碎）
AI_MIN_COVER = 0.03          # AI 结果覆盖率低于此值视为失败,回落传统路径
AI_MAX_COVER = 0.98          # 高于此值(几乎整图都是主体)视为失败
AI_SMOOTH_SIGMA = 2.5        # AI 蒙版边缘平滑强度(基准,按分辨率缩放)


@dataclass
class MattingResult:
    mask: np.ndarray          # uint8 0/255，与输入同尺寸
    tier: int
    confidence: float         # 0~1
    needs_fix: bool           # 是否需要人工修正
    reasons: list[str] = field(default_factory=list)


# ---------------- 读取与预处理 ----------------

def load_image(path: Path) -> np.ndarray:
    """读取图片为 RGBA（自动按 EXIF 转正）。"""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")
        return np.asarray(img).copy()


def downscale(rgba: np.ndarray, max_dim: int = WORK_MAX_DIM) -> np.ndarray:
    """最长边超过 max_dim 时等比缩小（INTER_AREA），避免大图处理过慢。"""
    h, w = rgba.shape[:2]
    if max(h, w) <= max_dim:
        return rgba
    scale = max_dim / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgba, size, interpolation=cv2.INTER_AREA)


# ---------------- 分档抠图 ----------------

def auto_mask(rgba: np.ndarray, tolerance: float = 20.0, segmenter=None) -> MattingResult:
    """自动抠图：AI 可用时优先(第 0 档)，失败静默回落按背景复杂度分档处理。"""
    rgba = np.asarray(rgba)

    # 第 0 档：AI 模型(isnet)——可用且结果合理时直接采用
    if segmenter is not None and getattr(segmenter, "is_ready", False):
        ai_result = _ai_mask(rgba, segmenter)
        if ai_result is not None:
            return ai_result

    # 第 1 档：透明背景 PNG，alpha 即蒙版
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[..., 3]
        transparent = float((alpha < 250).mean())
        if transparent >= MIN_ALPHA_FRACTION and int(alpha.max()) > 0:
            mask = (alpha >= 128).astype(np.uint8) * 255
            return MattingResult(mask=mask, tier=TIER_ALPHA, confidence=1.0,
                                 needs_fix=False, reasons=[])

    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)

    kind, model, _ = _classify_background(lab)

    # 第 4 档：复杂背景走 GrabCut，默认标记"建议修正"
    if kind == "complex":
        mask = cleanup_mask(_grabcut_mask(rgb), fill_hole_ratio=1.0).astype(np.uint8) * 255
        return MattingResult(mask=mask, tier=TIER_COMPLEX, confidence=0.35,
                             needs_fix=True, reasons=["复杂背景，自动效果可能不佳，建议人工检查"])

    # 第 2/3 档：与背景模型比较色距
    if kind == "solid":
        diff = np.linalg.norm(lab - model.reshape(1, 1, 3), axis=2)
        tier = TIER_SOLID
    else:
        diff = _poly_diff(lab, model)
        tier = TIER_GRADIENT

    bg_like = diff <= float(tolerance)
    mask_bool = cleanup_mask(_remove_edge_connected_bg(bg_like))
    confidence, needs_fix, reasons = _quality(mask_bool, tier)

    # 效果明显不行（没识别到主体 / 背景几乎没去掉 / 主体贴边）：换 GrabCut 兜底重试一次
    if confidence <= 0.4:
        retry = cleanup_mask(_grabcut_mask(rgb), fill_hole_ratio=1.0)
        if _mask_preferable(retry, mask_bool):
            return MattingResult(mask=retry.astype(np.uint8) * 255, tier=TIER_COMPLEX,
                                 confidence=0.4, needs_fix=True,
                                 reasons=["自动效果一般，已换 GrabCut 重试，建议检查"])

    return MattingResult(mask=mask_bool.astype(np.uint8) * 255, tier=tier,
                         confidence=confidence, needs_fix=needs_fix, reasons=reasons)


def _ai_mask(rgba: np.ndarray, segmenter) -> MattingResult | None:
    """AI 档:调用本地分割模型生成蒙版;结果不合理时返回 None(调用方回落)。"""
    rgb = np.asarray(rgba)[..., :3]
    try:
        soft = segmenter.predict(np.ascontiguousarray(rgb))
    except Exception:  # noqa: BLE001 - AI 失败一律降级,不打断处理流程
        return None
    if soft is None:
        return None
    soft = np.asarray(soft)
    if soft.shape[:2] != rgb.shape[:2]:
        soft = cv2.resize(soft, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    hard = soft.astype(np.uint8) >= int(AI_THRESHOLD * 255 + 0.5)
    mask_bool = _polish_ai_mask(hard)

    cover = float(mask_bool.mean())
    if cover < AI_MIN_COVER or cover > AI_MAX_COVER:
        return None  # 覆盖率荒谬,视为失败
    confidence, needs_fix, reasons = _quality(mask_bool, TIER_AI)
    return MattingResult(mask=mask_bool.astype(np.uint8) * 255, tier=TIER_AI,
                         confidence=confidence, needs_fix=needs_fix, reasons=reasons)


def _polish_ai_mask(mask_bool: np.ndarray) -> np.ndarray:
    """AI 蒙版后处理:去远处碎片 → 填洞 → 去毛刺 → 平滑轮廓(剪影更干净)。

    规则说明:
    - 碎片:面积小于"主域 3%"且小于"整图 0.3%"的连通块视为背景误判,删除;
    - 填洞:被主体包围的内部空洞(如脸部浅色区)全部填实,剪影保持实心;
    - 平滑:高斯模糊 + 阈值,把 AI 上采样留下的台阶/毛刺抹平(强度随分辨率缩放)。
    """
    m = mask_bool.astype(np.uint8)
    area_img = m.size
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]
        main_area = int(areas[order[0]])
        keep = [int(1 + order[0])]
        for idx in order[1:]:
            a = int(areas[idx])
            if a < max(main_area * 0.03, area_img * 0.003):
                continue  # 小碎块:背景误判
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            if a / max(w * h, 1) < 0.22:
                continue  # 大而低填充的弧状/条带状块:背景边缘误判(如家具描边)
            keep.append(int(1 + idx))
        m = np.isin(labels, keep).astype(np.uint8)

    inv = (1 - m).astype(np.uint8)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if nn > 1:
        border = set(int(i) for i in np.unique(np.concatenate(
            [ll[0, :], ll[-1, :], ll[:, 0], ll[:, -1]])))
        fill = [i for i in range(1, nn)
                if i not in border and int(ss[i, cv2.CC_STAT_AREA]) < area_img * 0.5]
        if fill:
            m[np.isin(ll, fill)] = 1

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    h, w = m.shape[:2]
    sigma = max(2.0, min(5.0, min(h, w) / 1080.0 * AI_SMOOTH_SIGMA))
    soft = cv2.GaussianBlur(m * 255, (0, 0), sigma)
    return (soft >= 128).astype(bool)


def mask_from_rect(rgba: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """用矩形框选主体后重跑 GrabCut（编辑器用），返回 uint8 0/255 蒙版。"""
    rgb = np.asarray(rgba)[..., :3]
    return cleanup_mask(_grabcut_mask(rgb, rect=rect), fill_hole_ratio=1.0).astype(np.uint8) * 255


def _grabcut_mask(rgb: np.ndarray, rect=None, iterations: int = 5) -> np.ndarray:
    """GrabCut 前景提取。rect 为 (x, y, w, h)，缺省时取内缩 2% 的整图。

    GrabCut 耗时随分辨率急剧上升（1600px 要 ~28 秒），所以在长图/大图上先降采样分割，
    再把蒙版放大回原尺寸；剪影用途下这点边缘细节损失可以忽略。
    """
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > GRABCUT_MAX_DIM:
        scale = GRABCUT_MAX_DIM / max(h, w)
    work_w = max(1, round(w * scale))
    work_h = max(1, round(h * scale))
    work = rgb if scale == 1.0 else cv2.resize(
        rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)

    if rect is None:
        band = max(3, int(round(min(work_h, work_w) * 0.02)))
        x, y, rw, rh = band, band, max(1, work_w - 2 * band), max(1, work_h - 2 * band)
    else:
        x, y, rw, rh = (int(round(v * scale)) for v in rect)
    x = max(0, min(x, work_w - 1))
    y = max(0, min(y, work_h - 1))
    rw = max(1, min(rw, work_w - x))
    rh = max(1, min(rh, work_h - y))

    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    mask = np.full((work_h, work_w), cv2.GC_BGD, np.uint8)  # 矩形外=确定背景
    mask[y:y + rh, x:x + rw] = cv2.GC_PR_FGD                # 矩形内=可能前景
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(7)                                       # 结果可复现
    cv2.grabCut(bgr, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

    if scale != 1.0:  # 蒙版放大回原尺寸（注意先乘 255，否则插值结果只有 0/1）
        up = cv2.resize(fg.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        fg = up >= 128
    return fg


def _border_coords(h: int, w: int, band_frac: float = 0.02):
    """外圈边缘带采样点坐标（用于估计背景），控制采样规模保证速度。"""
    band = max(3, int(round(min(h, w) * band_frac)))
    band = min(band, max(1, h // 3), max(1, w // 3))
    ys, xs = [], []
    for y0, y1 in ((0, band), (h - band, h)):
        yt, xt = np.meshgrid(np.arange(y0, y1), np.arange(w), indexing="ij")
        ys.append(yt.ravel())
        xs.append(xt.ravel())
    for x0, x1 in ((0, band), (w - band, w)):
        yl, xl = np.meshgrid(np.arange(h), np.arange(x0, x1), indexing="ij")
        ys.append(yl.ravel())
        xs.append(xl.ravel())
    ys = np.concatenate(ys)
    xs = np.concatenate(xs)
    if ys.size > 2000:
        idx = np.linspace(0, ys.size - 1, 2000).astype(int)
        ys, xs = ys[idx], xs[idx]
    return ys, xs


def _poly_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """二次多项式特征：1, u, v, u², v², uv。"""
    ones = np.ones_like(u)
    return np.stack([ones, u, v, u * u, v * v, u * v], axis=1)


def _classify_background(lab: np.ndarray):
    """判断背景档位。

    返回 (kind, model, residual)：
    - solid    → model 为背景中位色（3 维）
    - gradient → model 为二次多项式系数（6x3）
    - complex  → model 为 None
    """
    h, w = lab.shape[:2]
    ys, xs = _border_coords(h, w)
    cols = lab[ys, xs]

    med = np.median(cols, axis=0)
    res0 = np.linalg.norm(cols - med, axis=1)
    r0 = float(np.median(res0))
    if r0 <= SOLID_RESIDUAL_TOL:
        return "solid", med, r0

    u = xs.astype(np.float32) / max(w - 1, 1) * 2.0 - 1.0
    v = ys.astype(np.float32) / max(h - 1, 1) * 2.0 - 1.0
    f = _poly_features(u, v)
    keep = np.ones(cols.shape[0], bool)
    coef = None
    for _ in range(3):  # 剔除外点（贴边主体等）后重新拟合
        if int(keep.sum()) < 12:
            break
        coef, *_ = np.linalg.lstsq(f[keep], cols[keep], rcond=None)
        res = np.linalg.norm(f @ coef - cols, axis=1)
        thr = float(np.percentile(res[keep], 80))
        keep = res <= max(thr, 1e-6)
    if coef is None:
        return "complex", None, float("inf")
    res = np.linalg.norm(f @ coef - cols, axis=1)
    r2 = float(np.median(res[keep])) if keep.any() else float("inf")
    if r2 <= GRADIENT_RESIDUAL_TOL:
        return "gradient", coef, r2
    return "complex", None, r2


def _poly_diff(lab: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """逐像素计算与渐变背景模型（二次多项式）的 Lab 色距，分块计算省内存。"""
    h, w = lab.shape[:2]
    out = np.empty((h, w), np.float32)
    step = 256
    u_cache = np.arange(w, dtype=np.float32) / max(w - 1, 1) * 2.0 - 1.0
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        v = np.arange(y0, y1, dtype=np.float32) / max(h - 1, 1) * 2.0 - 1.0
        uu, vv = np.meshgrid(u_cache, v)
        f = _poly_features(uu.ravel(), vv.ravel())
        est = (f @ coef).reshape(y1 - y0, w, 3)
        out[y0:y1] = np.linalg.norm(lab[y0:y1] - est, axis=2)
    return out


def _remove_edge_connected_bg(bg_like: np.ndarray) -> np.ndarray:
    """只把"与图像边缘连通的背景"去掉；被主体包围的背景空洞保留为主体。"""
    bg = bg_like.astype(np.uint8)
    n, labels = cv2.connectedComponents(bg, connectivity=8)
    if n <= 1:
        return np.ones_like(bg_like, bool)  # 没有背景像素，全为主体
    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ]))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.ones_like(bg_like, bool)
    return ~np.isin(labels, border_labels)


def cleanup_mask(mask_bool: np.ndarray, fill_hole_ratio: float = 0.04) -> np.ndarray:
    """蒙版清理：闭运算补缝 → 填内部小洞 → 去孤立小碎块。

    fill_hole_ratio：内部空洞的填充阈值（占整图比例）；1.0 表示全部填实，
    用于复杂背景档（如头发轮廓内的脸、浅色衣服等被误判成背景的大空洞）。
    """
    m = mask_bool.astype(np.uint8)
    area_img = m.size
    if m.sum() == 0 or float(m.mean()) >= 0.995:
        return mask_bool.astype(bool).copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # 填掉被主体包围的空洞（纯色 logo、镂空斑点等）
    inv = (1 - m).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n > 2:
        border = np.unique(np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
        ]))
        border_set = set(int(i) for i in border)
        fill_ids = [
            i for i in range(1, n)
            if i not in border_set and int(stats[i, cv2.CC_STAT_AREA]) < area_img * fill_hole_ratio
        ]
        if fill_ids:
            m[np.isin(labels, fill_ids)] = 1

    # 去掉零碎小噪点（保留最大块及其它较大块）
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n2 > 2:
        min_area = max(area_img * 0.0004, 64)
        areas = stats2[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        drop = [
            i for i in range(1, n2)
            if i != largest and int(stats2[i, cv2.CC_STAT_AREA]) < min_area
        ]
        if drop:
            m[np.isin(labels2, drop)] = 0

    return m.astype(bool)


def _border_ratio(mask_bool: np.ndarray) -> float:
    """主体贴在图像边缘上的比例（越大约束越差）。"""
    border = np.concatenate([mask_bool[0, :], mask_bool[-1, :],
                             mask_bool[:, 0], mask_bool[:, -1]])
    return float(border.mean())


def _mask_preferable(candidate: np.ndarray, current: np.ndarray) -> bool:
    """GrabCut 兜底结果是否比原结果更合理：主体占比别太离谱，且贴边更少。"""
    area = float(candidate.mean())
    if area < 0.03 or area > 0.95:
        return False
    return _border_ratio(candidate) <= _border_ratio(current)


_BASE_CONF = {TIER_AI: 0.95, TIER_SOLID: 0.9, TIER_GRADIENT: 0.75}


def _quality(mask_bool: np.ndarray, tier: int):
    """蒙版质量评估：置信度、是否需要人工修正、原因说明。"""
    area = float(mask_bool.mean())
    n_comp = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)[0] - 1
    border_ratio = _border_ratio(mask_bool)

    conf = _BASE_CONF.get(tier, 0.75)
    reasons: list[str] = []
    if area < 0.02:
        conf = min(conf, 0.15)
        reasons.append("几乎没识别到主体")
    elif area > 0.97:
        conf = min(conf, 0.15)
        reasons.append("背景几乎未去除")
    if border_ratio > 0.3:
        conf = min(conf, 0.4)
        reasons.append("主体贴边较多，建议检查")
    if n_comp > 8:
        conf = min(conf, 0.55)
        reasons.append("边缘碎块较多")
    return conf, conf < 0.55, reasons


# ---------------- 蒙版后处理 ----------------

def feather_alpha(mask_uint8: np.ndarray, feather_px: float) -> np.ndarray:
    """边缘羽化：高斯模糊蒙版作为 alpha，让剪影边缘不生硬。"""
    mask_uint8 = mask_uint8.astype(np.uint8)
    if feather_px is None or feather_px <= 0:
        return mask_uint8.copy()
    k = float(feather_px)
    return cv2.GaussianBlur(mask_uint8, (0, 0), sigmaX=k, sigmaY=k)


# ---------------- 渲染 ----------------

def _mask_bbox(mask_uint8: np.ndarray, pad: int = 4):
    ys, xs = np.nonzero(mask_uint8 >= 128)
    if ys.size == 0:
        return None
    h, w = mask_uint8.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def _fit_size(w: int, h: int, canvas_w: int, canvas_h: int, ratio: float):
    """角色缩放到画布的尺寸（按高度归一化，宽角色按宽度约束）。"""
    if w <= 0 or h <= 0:
        return 1, 1
    scale = min(canvas_h * ratio / h, canvas_w * ratio / w)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = arr.shape[:2]
    interp = cv2.INTER_AREA if (tw * th) < (w * h) else cv2.INTER_LINEAR
    return cv2.resize(arr, (tw, th), interpolation=interp)


def _canvas(cfg: Config, bg: str) -> np.ndarray:
    out = np.empty((cfg.canvas_h, cfg.canvas_w, 3), np.uint8)
    out[:, :] = ImageColor.getrgb(bg)
    return out


def _composite(dst_rgb: np.ndarray, fg_rgb: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    """把 fg 以 alpha 叠到 dst 的 (x, y) 处（直 alpha 混合）。"""
    th, tw = alpha.shape[:2]
    region = dst_rgb[y:y + th, x:x + tw].astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    region = region * (1.0 - a) + fg_rgb.astype(np.float32) * a
    dst_rgb[y:y + th, x:x + tw] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


def _mean_luminance(rgb: np.ndarray, alpha: np.ndarray) -> float:
    sel = alpha >= 128
    if not sel.any():
        return 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    return float(lab[..., 0][sel].mean())


def render_guess(rgba, mask, config: Config | None = None, feather_px: float | None = None):
    """生成猜谜图：白底 + 黑色剪影，角色居中、高度归一化到统一画布。"""
    cfg = config or Config()
    feather = cfg.feather_px if feather_px is None else feather_px
    mask = np.asarray(mask)
    alpha = feather_alpha(mask, feather)
    out = _canvas(cfg, cfg.guess_bg)

    bbox = _mask_bbox(mask)
    if bbox is None:
        return Image.fromarray(out, "RGB")
    x0, y0, x1, y1 = bbox
    sub_alpha = _resize(alpha[y0:y1, x0:x1],
                        _fit_size(x1 - x0, y1 - y0, cfg.canvas_w, cfg.canvas_h, cfg.subject_ratio))
    th, tw = sub_alpha.shape[:2]
    fg = np.empty((th, tw, 3), np.uint8)
    fg[:, :] = ImageColor.getrgb(cfg.silhouette_color)
    _composite(out, fg, sub_alpha, (cfg.canvas_w - tw) // 2, (cfg.canvas_h - th) // 2)
    return Image.fromarray(out, "RGB")


def render_reveal(rgba, mask, config: Config | None = None, feather_px: float | None = None):
    """生成揭晓图：黑底（或深灰底）原图；深色角色加浅色描边防被吞掉。"""
    cfg = config or Config()
    if cfg.reveal_mode == "original":
        return render_original(rgba, cfg)

    feather = cfg.feather_px if feather_px is None else feather_px
    mask = np.asarray(mask)
    rgb_all = np.asarray(rgba)[..., :3]
    alpha = feather_alpha(mask, feather)
    bg = "#2B2B2B" if cfg.reveal_mode == "darkgray" else "#000000"
    out = _canvas(cfg, bg)

    bbox = _mask_bbox(mask)
    if bbox is None:
        return Image.fromarray(out, "RGB")
    x0, y0, x1, y1 = bbox
    size = _fit_size(x1 - x0, y1 - y0, cfg.canvas_w, cfg.canvas_h, cfg.subject_ratio)
    sub_alpha = _resize(alpha[y0:y1, x0:x1], size)
    sub_rgb = _resize(rgb_all[y0:y1, x0:x1], size)
    th, tw = sub_alpha.shape[:2]
    x = (cfg.canvas_w - tw) // 2
    y = (cfg.canvas_h - th) // 2

    _composite(out, sub_rgb, sub_alpha, x, y)

    # 深色角色保护：沿轮廓补一圈细浅色描边，避免黑底吞掉角色
    if cfg.dark_protect and _mean_luminance(sub_rgb, sub_alpha) < DARK_SUBJECT_L:
        ring = cv2.subtract(cv2.dilate(sub_alpha, np.ones((3, 3), np.uint8)), sub_alpha)
        ring = (ring > 8).astype(np.uint8) * 230
        ring_rgb = np.empty((th, tw, 3), np.uint8)
        ring_rgb[:, :] = (225, 225, 225)
        _composite(out, ring_rgb, ring, x, y)

    return Image.fromarray(out, "RGB")


def render_original(rgba, config: Config | None = None):
    """揭晓图的"原图背景"模式：整图等比放入画布，上下留黑边。"""
    cfg = config or Config()
    out = _canvas(cfg, "#000000")
    rgb = np.asarray(rgba)[..., :3]
    h, w = rgb.shape[:2]
    tw, th = _fit_size(w, h, cfg.canvas_w, cfg.canvas_h, 1.0)
    out[(cfg.canvas_h - th) // 2:(cfg.canvas_h - th) // 2 + th,
        (cfg.canvas_w - tw) // 2:(cfg.canvas_w - tw) // 2 + tw] = _resize(rgb, (tw, th))
    return Image.fromarray(out, "RGB")