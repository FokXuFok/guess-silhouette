"""让 pytest 能从仓库根目录导入 app 包,并提供测试默认配置。"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _disable_ai_matting(monkeypatch):
    """测试默认禁用 AI 抠图:避免依赖 176MB 模型与推理耗时,保证确定性。

    仅 patch 模块属性(store 走属性访问),test_segmentation 中直接调用的
    create_segmenter 是 import 早期绑定,不受影响;AI 专项测试自行构造 fake。
    """
    import app.segmentation as segmentation

    monkeypatch.setattr(segmentation, "create_segmenter",
                        lambda config, model_path=None: None)