"""缓存与状态管理（Store）测试：扫描过滤、处理、跳过、缓存损坏自愈、坏图跳过。"""
import json
import os

import numpy as np
from PIL import Image

from app.config import Config
from app.store import Store
from tests.image_factory import make_blob


def _write_png(path, kind="solid", size=(640, 480)):
    rgb, _ = make_blob(kind, size=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def test_sync_processes_images_and_saves_state(tmp_path):
    img_dir = tmp_path / "图片"
    img_dir.mkdir()
    _write_png(img_dir / "漩涡鸣人.png")
    (img_dir / "说明.txt").write_text("not an image", encoding="utf-8")

    store = Store(img_dir, Config())
    summary = store.sync()

    assert summary.processed == 1
    assert len(summary.entries) == 1
    entry = summary.entries[0]
    assert entry.name == "漩涡鸣人"
    assert entry.ok
    assert (store.cache_dir / entry.guess_file).exists()
    assert (store.cache_dir / entry.reveal_file).exists()
    assert (store.cache_dir / entry.mask_file).exists()

    state = json.loads((store.cache_dir / "state.json").read_text(encoding="utf-8"))
    assert state["version"] == 1
    assert "漩涡鸣人.png" in state["images"]


def test_sync_skips_unchanged_images(tmp_path):
    img_dir = tmp_path / "图片"
    _write_png(img_dir / "a.png")
    store = Store(img_dir, Config())
    store.sync()
    second = store.sync()
    assert second.processed == 0
    assert second.skipped == 1


def test_changed_image_is_reprocessed(tmp_path):
    img_dir = tmp_path / "图片"
    path = _write_png(img_dir / "a.png")
    store = Store(img_dir, Config())
    store.sync()

    rgb, _ = make_blob("gradient")
    Image.fromarray(rgb).save(path)
    future = os.path.getmtime(path) + 10
    os.utime(path, (future, future))

    assert store.sync().processed == 1


def test_algo_version_change_forces_reprocess(tmp_path):
    """抠图算法升级后，旧缓存应整体重算，避免一直用旧效果。"""
    img_dir = tmp_path / "图片"
    _write_png(img_dir / "a.png")
    store = Store(img_dir, Config())
    store.sync()

    state_path = store.cache_dir / "state.json"
    data = json.loads(state_path.read_text(encoding="utf-8"))
    data["algo"] = 1  # 旧算法版本
    state_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert store.sync().processed == 1


def test_corrupt_cache_is_regenerated(tmp_path):
    img_dir = tmp_path / "图片"
    _write_png(img_dir / "a.png")
    store = Store(img_dir, Config())
    first = store.sync()
    guess = store.cache_dir / first.entries[0].guess_file
    guess.write_bytes(b"not a real png")

    assert store.sync().processed == 1
    with Image.open(guess) as im:
        im.load()


def test_unreadable_image_marked_failed(tmp_path):
    img_dir = tmp_path / "图片"
    img_dir.mkdir()
    (img_dir / "坏图.png").write_bytes(b"this is not an image")
    summary = Store(img_dir, Config()).sync()
    assert len(summary.entries) == 1
    assert summary.entries[0].ok is False
    assert summary.entries[0].error


def test_empty_folder_sync(tmp_path):
    img_dir = tmp_path / "图片"
    img_dir.mkdir()
    summary = Store(img_dir, Config()).sync()
    assert summary.entries == []
    assert summary.processed == 0


def test_removed_file_drops_state_entry(tmp_path):
    img_dir = tmp_path / "图片"
    p = _write_png(img_dir / "a.png")
    _write_png(img_dir / "b.png")
    store = Store(img_dir, Config())
    store.sync()
    p.unlink()
    summary = store.sync()
    assert [e.name for e in summary.entries] == ["b"]


def test_save_edit_marks_verified_and_rerenders(tmp_path):
    img_dir = tmp_path / "图片"
    _write_png(img_dir / "a.png")
    store = Store(img_dir, Config())
    summary = store.sync()
    entry = summary.entries[0]
    assert entry.ok

    mask = store.load_mask(entry)
    assert isinstance(mask, np.ndarray)
    edited = mask.copy()
    edited[:] = 0
    h, w = edited.shape
    edited[h // 4: h * 3 // 4, w // 4: w * 3 // 4] = 255

    store.save_edit(entry, edited)
    assert entry.needs_fix is False
    assert entry.verified is True
    again = store.sync()
    assert again.processed == 0  # 编辑后不会因为状态变化被重跑