"""图像处理核心测试：分档抠图、蒙版清理、羽化、剪影/揭晓图生成。"""
import numpy as np

from app import imaging as im
from app.config import Config
from tests.image_factory import SUBJECT_COLOR, iou, make_blob


# ---------- 分档抠图 ----------

def test_solid_background_is_tier2_and_matches_gt():
    rgb, gt = make_blob("solid")
    res = im.auto_mask(rgb, tolerance=20)
    assert res.tier == im.TIER_SOLID
    assert iou(res.mask > 0, gt) > 0.9
    assert res.needs_fix is False


def test_gradient_background_is_tier3_and_matches_gt():
    rgb, gt = make_blob("gradient")
    res = im.auto_mask(rgb, tolerance=20)
    assert res.tier == im.TIER_GRADIENT
    assert iou(res.mask > 0, gt) > 0.85


def test_noise_background_is_tier4_and_flagged():
    rgb, gt = make_blob("noise")
    res = im.auto_mask(rgb, tolerance=20)
    assert res.tier == im.TIER_COMPLEX
    assert res.needs_fix is True
    assert res.reasons


def test_grabcut_runs_on_reduced_resolution():
    """大图必须先降采样再跑 GrabCut，否则复杂背景图会慢到不可用。"""
    import cv2

    sizes = []
    original = cv2.grabCut

    def spy(img, *args, **kwargs):
        sizes.append(img.shape[:2])
        return original(img, *args, **kwargs)

    cv2.grabCut = spy
    try:
        rgb, _ = make_blob("noise", size=(1600, 1200))
        im.auto_mask(rgb, tolerance=20)
    finally:
        cv2.grabCut = original

    assert sizes, "应调用过 GrabCut"
    assert max(sizes[0]) <= im.GRABCUT_MAX_DIM


def test_grabcut_mask_quality_holds_at_full_size():
    """降采样 GrabCut 后蒙版仍要放大回原图尺寸，且分割质量保持可用。"""
    rgb, gt = make_blob("noise", size=(1600, 1200))
    res = im.auto_mask(rgb, tolerance=20)
    assert res.tier == im.TIER_COMPLEX
    assert res.mask.shape[:2] == gt.shape
    assert iou(res.mask > 0, gt) > 0.85


def test_cleanup_can_fill_large_enclosed_holes():
    """复杂背景档：主体内部被包围的大空洞也要填实（如头发轮廓里的脸）。"""
    m = np.zeros((400, 400), bool)
    m[50:350, 50:350] = True
    m[150:250, 150:250] = False  # 占整图 6% 的大洞

    assert not im.cleanup_mask(m)[200, 200]                     # 默认只填小洞
    assert im.cleanup_mask(m, fill_hole_ratio=1.0)[200, 200]    # 指定后全部填实


def test_bad_gradient_result_falls_back_to_grabcut():
    """渐变档效果差（容差过低导致背景几乎没去除）时，应自动换 GrabCut 重试并标记建议修正。"""
    rgb, gt = make_blob("gradient")
    res = im.auto_mask(rgb, tolerance=0.0)  # 容差过低：背景几乎没被去掉
    assert res.tier == im.TIER_COMPLEX
    assert res.needs_fix is True
    assert res.reasons


def test_transparent_png_uses_alpha():
    rgb, gt = make_blob("solid")
    rgba = np.dstack([rgb, (gt * 255).astype(np.uint8)])
    res = im.auto_mask(rgba, tolerance=20)
    assert res.tier == im.TIER_ALPHA
    assert iou(res.mask > 0, gt) > 0.95
    assert res.needs_fix is False


def test_fully_opaque_rgba_is_not_alpha_tier():
    rgb, gt = make_blob("solid")
    rgba = np.dstack([rgb, np.full(gt.shape, 255, np.uint8)])
    res = im.auto_mask(rgba, tolerance=20)
    assert res.tier == im.TIER_SOLID


def test_mask_from_rect_segments_subject():
    rgb, gt = make_blob("solid", size=(400, 300))
    h, w = gt.shape
    rect = (int(w * 0.2), int(h * 0.05), int(w * 0.6), int(h * 0.9))
    mask = im.mask_from_rect(rgb, rect)
    assert iou(mask > 0, gt) > 0.85


# ---------- 蒙版清理 ----------

def test_cleanup_removes_specks_and_fills_holes():
    m = np.zeros((200, 200), bool)
    m[50:150, 50:150] = True   # 主体
    m[60:64, 60:64] = False    # 内部小洞：应被填上
    m[5:7, 5:7] = True         # 孤立小噪点：应被去掉
    out = im.cleanup_mask(m)
    assert bool(out[100, 100])
    assert bool(out[61, 61])
    assert not bool(out[5, 5])
    assert not bool(out[6, 6])


def test_feather_produces_soft_edge():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 255
    a = im.feather_alpha(m, 2.0)
    assert a[50, 50] == 255
    assert 0 < int(a[70, 50]) < 255


# ---------- 剪影 / 揭晓图 ----------

def test_render_guess_white_bg_black_subject():
    rgb, gt = make_blob("solid", size=(800, 600))
    mask = (gt * 255).astype(np.uint8)
    img = im.render_guess(rgb, mask, feather_px=0.0)
    assert img.size == (1920, 1080)
    px = np.asarray(img)
    assert tuple(px[5, 5][:3]) == (255, 255, 255)
    assert int(px[1080 // 2, 1920 // 2][0]) < 40


def test_render_reveal_black_bg_original_color():
    rgb, gt = make_blob("solid", size=(800, 600))
    mask = (gt * 255).astype(np.uint8)
    img = im.render_reveal(rgb, mask, feather_px=0.0)
    px = np.asarray(img)
    assert tuple(px[5, 5][:3]) == (0, 0, 0)
    center = px[1080 // 2, 1920 // 2]
    assert abs(int(center[0]) - SUBJECT_COLOR[0]) < 30
    assert abs(int(center[2]) - SUBJECT_COLOR[2]) < 30


def test_render_guess_normalizes_subject_height():
    rgb, gt = make_blob("solid", size=(800, 600))
    mask = (gt * 255).astype(np.uint8)
    img = im.render_guess(rgb, mask, feather_px=0.0)
    px = np.asarray(img)
    dark_rows = np.where((px[..., 0] < 60).any(axis=1))[0]
    height = int(dark_rows.max() - dark_rows.min() + 1)
    assert 0.75 * 1080 <= height <= 0.92 * 1080


def test_render_guess_empty_mask_returns_plain_canvas():
    rgb, _ = make_blob("solid", size=(200, 200))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    img = im.render_guess(rgb, mask, feather_px=0.0)
    px = np.asarray(img)
    assert img.size == (1920, 1080)
    assert tuple(px[540, 960][:3]) == (255, 255, 255)


def test_reveal_adds_outline_for_dark_subject():
    rgb = np.full((300, 300, 3), (250, 250, 250), np.uint8)
    mask = np.zeros((300, 300), np.uint8)
    mask[80:220, 80:220] = 255
    rgb[mask > 0] = (12, 10, 15)

    with_protect = np.asarray(im.render_reveal(rgb, mask, feather_px=0.0))
    assert int(with_protect[1080 // 2, 1920 // 2][0]) < 40            # 主体深色
    assert bool((with_protect[540, 400:600, 0] > 120).any())          # 边缘有浅色描边

    without = np.asarray(
        im.render_reveal(rgb, mask, feather_px=0.0, config=Config(dark_protect=False))
    )
    assert not bool((without[540, 400:600, 0] > 120).any())


def test_render_original_mode_shows_whole_image():
    rgb, _ = make_blob("solid", size=(800, 400))
    cfg = Config(reveal_mode="original")
    img = im.render_reveal(rgb, (rgb[..., 0] > 0).astype(np.uint8) * 255, feather_px=0.0, config=cfg)
    px = np.asarray(img)
    assert img.size == (1920, 1080)
    # 上下应有黑边；画面里能看到原图背景（约 245）和主体原色
    assert int(px[5, 960][0]) < 20
    assert int(px[540, 100][0]) > 230          # 左侧背景区
    assert 170 <= int(px[655, 960][0]) <= 230  # 主体区显示原色 (200,60,90)


def test_downscale_caps_max_dimension():
    rgb, gt = make_blob("solid", size=(4000, 1000))
    rgba = np.dstack([rgb, np.full(gt.shape, 255, np.uint8)])
    out = im.downscale(rgba, max_dim=2000)
    assert max(out.shape[:2]) == 2000
    assert out.shape[2] == 4


def test_load_image_applies_exif_orientation(tmp_path):
    from PIL import Image

    rgb, _ = make_blob("solid", size=(120, 60))
    path = tmp_path / "rotated.jpg"
    img = Image.fromarray(rgb)
    exif = img.getexif()
    exif[274] = 6  # Orientation: 顺时针旋转 90°（手机竖拍常见）
    img.save(path, exif=exif)

    loaded = im.load_image(path)
    assert loaded.shape[:2] == (120, 60)  # 转正后应为竖图：高 120 × 宽 60
    assert loaded.shape[2] == 4


# ---------- AI 第 0 档 ----------

class _FakeSegmenter:
    """模拟 Segmenter:返回固定蒙版(None 表示推理失败)。"""

    def __init__(self, mask=None):
        self._mask = mask
        self.is_ready = True
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        return self._mask


def test_auto_mask_prefers_ai_when_available():
    """AI 可用时优先使用,结果按 TIER_AI 返回,不再走传统分档。"""
    rgb, gt = make_blob("noise")  # 传统路径这里是复杂背景 GrabCut
    seg = _FakeSegmenter(mask=gt.astype(np.uint8) * 255)
    res = im.auto_mask(rgb, segmenter=seg)
    assert seg.calls == 1
    assert res.tier == im.TIER_AI
    assert res.mask.shape == rgb.shape[:2]
    assert iou(res.mask > 0, gt) > 0.9
    assert res.needs_fix is False


def test_auto_mask_falls_back_when_ai_fails():
    rgb, _ = make_blob("solid")
    res = im.auto_mask(rgb, segmenter=_FakeSegmenter(None))
    assert res.tier == im.TIER_SOLID  # 静默回落传统路径


def test_auto_mask_falls_back_when_ai_coverage_absurd():
    rgb, _ = make_blob("solid")
    h, w = rgb.shape[:2]
    seg = _FakeSegmenter(mask=np.full((h, w), 255, np.uint8))  # 整图都是主体 → 不合理
    res = im.auto_mask(rgb, segmenter=seg)
    assert res.tier != im.TIER_AI
    assert res.tier == im.TIER_SOLID


def test_auto_mask_skips_not_ready_segmenter():
    rgb, _ = make_blob("solid")
    seg = _FakeSegmenter(None)
    seg.is_ready = False
    res = im.auto_mask(rgb, segmenter=seg)
    assert seg.calls == 0  # 未就绪不应被调用
    assert res.tier == im.TIER_SOLID


def test_ai_mask_polish_removes_far_specks_keeps_subject():
    """AI 蒙版后处理:远离主体的背景碎片应被清除,主体保持完整。"""
    rgb, gt = make_blob("solid")
    h, w = gt.shape
    mask = np.zeros((h, w), np.uint8)
    mask[gt] = 255
    mask[5:12, 5:12] = 255  # 角落 7x7 碎片(面积远小于阈值)
    seg = _FakeSegmenter(mask)
    res = im.auto_mask(rgb, segmenter=seg)
    assert res.tier == im.TIER_AI
    assert int((res.mask[5:12, 5:12] > 0).sum()) == 0  # 碎片被清除
    assert iou(res.mask > 0, gt) > 0.9                 # 主体完好


def test_ai_mask_polish_removes_lowfill_band():
    """大而低填充的弧状条带(背景家具描边类误判)也应被清除。"""
    import cv2

    h, w = 480, 640
    rgb = np.full((h, w, 3), 245, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (320, 300), 80, 255, -1)  # 主体:中心圆
    band = np.array([[10, 60], [160, 140], [320, 40], [480, 140], [630, 60]], np.int32)
    cv2.polylines(mask, [band], False, 255, 6)  # 顶部一条长波浪条带(远离主体)

    seg = _FakeSegmenter(mask)
    res = im.auto_mask(rgb, segmenter=seg)
    assert res.tier == im.TIER_AI
    assert int(res.mask[35:50, 310:330].max()) == 0  # 波浪峰值区被清除
    circ = np.zeros((h, w), np.uint8)
    cv2.circle(circ, (320, 300), 80, 255, -1)
    assert iou(res.mask > 0, circ > 0) > 0.9  # 圆形主体保持完整