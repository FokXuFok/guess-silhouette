"""运行路径工具：程序根目录、图片目录、缓存目录。"""
import sys
from pathlib import Path


def base_dir() -> Path:
    """程序根目录：打包后为 exe 所在目录，开发时为仓库根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_image_dir() -> Path:
    """默认图片目录：程序目录下的 图片/。"""
    return base_dir() / "图片"


def cache_dir_for(image_dir: Path) -> Path:
    """缓存目录：图片文件夹同级的 缓存/<文件夹名>/。

    同级目录不可写时（如选到盘符根目录），退回程序目录下。
    """
    image_dir = Path(image_dir).resolve()
    target = image_dir.parent / "缓存" / image_dir.name
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = base_dir() / "缓存" / image_dir.name
        target.mkdir(parents=True, exist_ok=True)
    return target