"""图片入库（添加图片）测试：复制、重名处理、重复内容跳过。"""
from PIL import Image

from app.library import add_images_to
from tests.image_factory import make_blob


def _make_png(path, kind="solid", size=(120, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb, _ = make_blob(kind, size=size)
    Image.fromarray(rgb).save(path)
    return path


def test_add_new_images_copies_them(tmp_path):
    src_dir = tmp_path / "下载"
    lib_dir = tmp_path / "图片"
    lib_dir.mkdir()
    a = _make_png(src_dir / "鸣人.png")
    b = _make_png(src_dir / "路飞.jpg")

    result = add_images_to(lib_dir, [a, b])
    assert result.added == 2
    assert (lib_dir / "鸣人.png").exists()
    assert (lib_dir / "路飞.jpg").exists()


def test_identical_duplicate_is_skipped(tmp_path):
    src = _make_png(tmp_path / "来源" / "鸣人.png")
    lib_dir = tmp_path / "图片"
    lib_dir.mkdir()
    _make_png(lib_dir / "鸣人.png")  # 同名同内容

    result = add_images_to(lib_dir, [src])
    assert result.added == 0
    assert result.skipped == 1


def test_name_conflict_gets_numbered_suffix(tmp_path):
    src = _make_png(tmp_path / "来源" / "鸣人.png", kind="gradient")
    lib_dir = tmp_path / "图片"
    lib_dir.mkdir()
    _make_png(lib_dir / "鸣人.png", kind="solid")  # 同名但内容不同

    result = add_images_to(lib_dir, [src])
    assert result.added == 1
    assert result.renamed == 1
    assert (lib_dir / "鸣人_2.png").exists()
    assert (lib_dir / "鸣人.png").exists()  # 原有文件不被覆盖


def test_non_image_files_are_skipped(tmp_path):
    src = tmp_path / "说明.txt"
    src.write_text("not an image", encoding="utf-8")
    lib_dir = tmp_path / "图片"
    lib_dir.mkdir()

    result = add_images_to(lib_dir, [src])
    assert result.added == 0
    assert result.skipped == 1


def test_source_already_inside_library_is_skipped(tmp_path):
    lib_dir = tmp_path / "图片"
    lib_dir.mkdir()
    inside = _make_png(lib_dir / "鸣人.png")

    result = add_images_to(lib_dir, [inside])
    assert result.added == 0
    assert result.skipped == 1
    assert len(list(lib_dir.iterdir())) == 1