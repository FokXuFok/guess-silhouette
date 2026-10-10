"""图片入库：把外部图片复制进 图片 文件夹（主页面【添加图片】用）。

规则：
- 非图片文件：跳过
- 已经在图库里的文件：跳过
- 同名且内容相同：跳过（避免重复添加）
- 同名但内容不同：自动加序号（如 鸣人_2.png），不覆盖已有文件
"""
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .store import IMAGE_EXTS


@dataclass
class AddResult:
    added: int = 0      # 成功加入
    skipped: int = 0    # 跳过（非图片 / 内容重复 / 已在图库）
    renamed: int = 0    # 因重名自动加了序号


def add_images_to(image_dir: Path, sources) -> AddResult:
    """把 sources 中的图片复制到 image_dir，返回添加结果统计。"""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    result = AddResult()

    for src in sources:
        src = Path(src)
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            result.skipped += 1
            continue
        try:
            already_inside = src.resolve().parent == image_dir.resolve()
        except OSError:
            already_inside = False
        if already_inside:
            result.skipped += 1
            continue

        dst = image_dir / src.name
        if dst.exists():
            if _same_content(src, dst):
                result.skipped += 1
                continue
            index = 2
            while True:
                dst = image_dir / f"{src.stem}_{index}{src.suffix}"
                if not dst.exists():
                    break
                index += 1
            result.renamed += 1

        shutil.copy2(src, dst)
        result.added += 1

    return result


def _same_content(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        return hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
    except OSError:
        return False