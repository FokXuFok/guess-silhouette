"""配置与路径工具测试。"""
from app.config import Config
from app.paths import cache_dir_for


def test_config_roundtrip(tmp_path):
    p = tmp_path / "config.json"
    cfg = Config()
    cfg.tolerance = 33
    cfg.reveal_mode = "darkgray"
    cfg.save(p)
    loaded = Config.load(p)
    assert loaded.tolerance == 33
    assert loaded.reveal_mode == "darkgray"
    assert loaded.feather_px == cfg.feather_px


def test_config_missing_file_returns_defaults(tmp_path):
    loaded = Config.load(tmp_path / "nope.json")
    assert loaded == Config()


def test_config_corrupt_file_returns_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{broken json", encoding="utf-8")
    assert Config.load(p) == Config()


def test_cache_dir_next_to_image_folder(tmp_path):
    image_dir = tmp_path / "猜剪影" / "图片"
    image_dir.mkdir(parents=True)
    cache = cache_dir_for(image_dir)
    assert cache == tmp_path / "猜剪影" / "缓存" / "图片"
    assert cache.is_dir()