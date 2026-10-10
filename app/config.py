"""配置：投影样式与抠图参数，保存在程序目录 config.json。"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import base_dir


@dataclass
class Config:
    tolerance: float = 20.0        # 抠图容差（Lab 色距，越大去除越多）
    feather_px: float = 1.5        # 边缘羽化像素
    reveal_mode: str = "black"     # 揭晓背景：black / darkgray / original
    dark_protect: bool = True      # 深色角色揭晓时加浅色描边
    canvas_w: int = 1920
    canvas_h: int = 1080
    subject_ratio: float = 0.82    # 角色占画布高度比例
    guess_bg: str = "#FFFFFF"      # 猜谜图背景
    silhouette_color: str = "#000000"  # 剪影颜色
    ai_matting: bool = True        # AI 抠图优先(模型缺失/onnxruntime 未装时自动降级)

    @staticmethod
    def default_path() -> Path:
        return base_dir() / "config.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        p = Path(path) if path is not None else cls.default_path()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        p = Path(path) if path is not None else self.default_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass