"""缓存与状态管理：扫描图片文件夹、处理并缓存剪影、维护 state.json。"""
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

from . import imaging, segmentation
from .config import Config
from .paths import cache_dir_for

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
STATE_VERSION = 1
ALGO_VERSION = 5  # 抠图算法版本：升级后旧缓存整体重算，避免一直用旧效果


@dataclass
class Entry:
    path: str            # 原图绝对路径
    name: str            # 文件名（不含扩展名），即答案
    mtime: float = 0.0
    size: int = 0
    ok: bool = False
    tier: int = 0
    needs_fix: bool = False
    reasons: list[str] = field(default_factory=list)
    verified: bool = False
    guess_file: str = ""
    reveal_file: str = ""
    mask_file: str = ""
    error: str = ""


@dataclass
class SyncSummary:
    entries: list[Entry]
    processed: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def needs_fix_count(self) -> int:
        return sum(1 for e in self.entries if e.ok and e.needs_fix and not e.verified)


class Store:
    """图片文件夹 ↔ 缓存目录 的管理器。"""

    def __init__(self, image_dir: Path, config: Config):
        self.image_dir = Path(image_dir)
        self.config = config
        self.cache_dir = cache_dir_for(self.image_dir)
        self.state_path = self.cache_dir / "state.json"
        # AI 分割器(模型缺失/未装 onnxruntime 时为 None,全流程走传统抠图)
        self.segmenter = segmentation.create_segmenter(config)

    # ---------- 扫描与状态文件 ----------

    def scan_images(self) -> list[Path]:
        """列出图片文件夹里的图片文件（过滤非图片 / 隐藏文件）。"""
        if not self.image_dir.is_dir():
            return []
        files = [
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
        ]
        return sorted(files, key=lambda p: p.name)

    def _load_state(self) -> dict[str, dict]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == STATE_VERSION:
                if data.get("algo") != ALGO_VERSION:
                    return {}  # 算法升级：忽略旧状态，全部重算
                images = data.get("images", {})
                if isinstance(images, dict):
                    return images
        except (OSError, ValueError):
            pass
        return {}

    def _write_state(self, images: dict[str, dict]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"version": STATE_VERSION, "algo": ALGO_VERSION, "images": images},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @staticmethod
    def _entry_from_dict(data: dict, path: Path) -> Entry:
        known = set(Entry.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["path"] = str(path.resolve())
        kwargs["name"] = path.stem
        return Entry(**kwargs)

    def _cache_files_ok(self, entry: Entry) -> bool:
        """缓存文件存在且能正常打开（损坏则触发重新生成）。"""
        for name in (entry.guess_file, entry.reveal_file, entry.mask_file):
            if not name:
                return False
            p = self.cache_dir / name
            if not p.is_file():
                return False
            try:
                with Image.open(p) as im:
                    im.load()
            except Exception:
                return False
        return True

    @staticmethod
    def _changed(path: Path, entry: Entry) -> bool:
        try:
            st = path.stat()
        except OSError:
            return True
        return abs(st.st_mtime - entry.mtime) > 1e-6 or int(st.st_size) != entry.size

    # ---------- 处理 ----------

    def sync(self, force: bool = False, progress=None) -> SyncSummary:
        """扫描并增量处理：新增 / 改动 / 缓存损坏的图重新生成剪影。"""
        state = self._load_state()
        files = self.scan_images()
        entries: list[Entry] = []
        processed = skipped = failed = 0
        total = len(files)

        for i, path in enumerate(files, 1):
            old = state.get(path.name)
            entry = (self._entry_from_dict(old, path) if old
                     else Entry(path=str(path.resolve()), name=path.stem))
            need = (force or old is None or not entry.ok
                    or self._changed(path, entry) or not self._cache_files_ok(entry))
            if need:
                try:
                    self._process(path, entry)
                    processed += 1
                except Exception as exc:  # 图片打不开等：跳过并记录
                    log.warning("处理失败 %s: %s", path.name, exc)
                    entry.ok = False
                    entry.error = str(exc)
                    failed += 1
            else:
                skipped += 1
            entries.append(entry)
            if progress:
                progress(i, total, path.name)

        self._write_state({Path(e.path).name: asdict(e) for e in entries})
        return SyncSummary(entries=entries, processed=processed,
                           skipped=skipped, failed=failed)

    def _process(self, path: Path, entry: Entry) -> None:
        st = path.stat()
        rgba = imaging.load_image(path)
        work = imaging.downscale(rgba)
        res = imaging.auto_mask(work, self.config.tolerance, segmenter=self.segmenter)

        cfg = self.config
        guess = imaging.render_guess(work, res.mask, cfg)
        if cfg.reveal_mode == "original":
            reveal = imaging.render_original(work, cfg)
        else:
            reveal = imaging.render_reveal(work, res.mask, cfg)

        stem = _safe_name(path.stem)
        h6 = _short_hash(str(path.resolve()))
        entry.guess_file = f"{stem}_{h6}_guess.png"
        entry.reveal_file = f"{stem}_{h6}_reveal.png"
        entry.mask_file = f"{stem}_{h6}_mask.png"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        guess.save(self.cache_dir / entry.guess_file)
        reveal.save(self.cache_dir / entry.reveal_file)
        Image.fromarray(res.mask).save(self.cache_dir / entry.mask_file)

        entry.mtime = st.st_mtime
        entry.size = int(st.st_size)
        entry.ok = True
        entry.tier = res.tier
        entry.needs_fix = res.needs_fix
        entry.reasons = list(res.reasons)
        entry.verified = False
        entry.error = ""

    # ---------- 读取缓存（界面 / 编辑器用） ----------

    def load_mask(self, entry: Entry) -> np.ndarray:
        with Image.open(self.cache_dir / entry.mask_file) as im:
            return np.asarray(im.convert("L")).copy()

    def load_render(self, entry: Entry, reveal: bool = False):
        name = entry.reveal_file if reveal else entry.guess_file
        with Image.open(self.cache_dir / name) as im:
            return im.convert("RGB").copy()

    def load_work_image(self, entry: Entry) -> np.ndarray:
        """编辑器用的工作图（与原图同比例，最长边 ≤ WORK_MAX_DIM）。"""
        return imaging.downscale(imaging.load_image(Path(entry.path)))

    def save_edit(self, entry: Entry, mask: np.ndarray, feather_px: float | None = None) -> None:
        """手动修正保存：重渲染剪影/揭晓图，标记为已确认。"""
        mask = np.asarray(mask)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        work = self.load_work_image(entry)
        if mask.shape[:2] != work.shape[:2]:
            resized = Image.fromarray(mask).resize((work.shape[1], work.shape[0]), Image.NEAREST)
            mask = np.asarray(resized).copy()

        cfg = self.config
        if feather_px is not None:
            cfg = replace(cfg, feather_px=float(feather_px))

        guess = imaging.render_guess(work, mask, cfg)
        if cfg.reveal_mode == "original":
            reveal = imaging.render_original(work, cfg)
        else:
            reveal = imaging.render_reveal(work, mask, cfg)

        Image.fromarray(mask).save(self.cache_dir / entry.mask_file)
        guess.save(self.cache_dir / entry.guess_file)
        reveal.save(self.cache_dir / entry.reveal_file)

        entry.verified = True
        entry.needs_fix = False
        entry.error = ""

        state = self._load_state()
        state[Path(entry.path).name] = asdict(entry)
        self._write_state(state)


def _safe_name(stem: str) -> str:
    s = "".join(c for c in stem if c not in '\\/:*?"<>|').strip()
    return s[:60] or "image"


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:6]