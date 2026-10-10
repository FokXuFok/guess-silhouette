"""界面冒烟测试：在独立子进程里跑真实启动路径。

覆盖：主页面 →【添加图片】→【开始】→ 空格循环（揭晓 / 下一题）→ 展示完毕页 → 回主页面 → 手动修正。

说明：Windows 商店版 Python 在同一进程内反复创建 Tcl/Tk 解释器会偶发失败，
所以这里用子进程跑，同时也更接近真实启动路径。
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CHILD_SCRIPT = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, r"__ROOT__")

from PIL import Image
from app.config import Config
from app.editor import EditorDialog
from app.ui import App
from tests.image_factory import make_blob

img_dir = Path(r"__IMG_DIR__")
extra_dir = Path(r"__EXTRA_DIR__")

rgb, _ = make_blob("solid")
Image.fromarray(rgb).save(img_dir / "测试角色.png")
rgb, _ = make_blob("noise")
Image.fromarray(rgb).save(img_dir / "复杂背景角色.png")

# 库外的一张图，用于测试【添加图片】
rgb, _ = make_blob("gradient")
Image.fromarray(rgb).save(extra_dir / "新角色.png")

app = App(img_dir, Config())
app.withdraw()
app.start()


def pump(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.01)


def wait_until(cond, timeout=40):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        pump(0.05)


# 启动后应停在主页面，并显示图片数量
wait_until(lambda: bool(app.entries) and not app._syncing)
assert app.view == "home", "启动后应停在主页面"
assert app._can_start is True
assert "共 2 张" in app.home_status.cget("text"), app.home_status.cget("text")

# 【添加图片】：模拟文件选择器选一张外部图片，自动复制进库并处理
import app.ui as ui_mod
ui_mod.filedialog.askopenfilenames = lambda **kw: (str(extra_dir / "新角色.png"),)
app.add_images()
wait_until(lambda: not app._syncing)
assert "已添加 1 张图片" in app.home_note.cget("text"), app.home_note.cget("text")
assert "共 3 张" in app.home_status.cget("text"), app.home_status.cget("text")
assert (img_dir / "新角色.png").exists(), "选中的图片应被复制进图片文件夹"

# 【开始】：空格循环 —— 揭晓 → 再按空格下一题黑影
app.start_game()
assert app.view == "game"
assert app.answer_label.cget("text") == "？？？"

shown = []
guard = 0
while app.view == "game" and guard < 20:
    guard += 1
    if not app.revealed:
        shown.append(app.deck.current.name)
        assert app.answer_label.cget("text") == "？？？"
    else:
        assert app.answer_label.cget("text") == app.deck.current.name
    app.handle_space()
    pump(0.02)

assert app.view == "done", "展示完一轮后应进入完毕页"
assert len(shown) == 3 and len(set(shown)) == 3, f"三张图片应不重复展示：{shown}"
assert "3" in app.done_sub.cget("text")

# 完毕页：空格回主页面
app.handle_space()
assert app.view == "home"

# 手动修正：复杂背景图应被标记"建议修正"
needs = [e for e in app.entries if e.needs_fix and not e.verified]
assert needs, "复杂背景图应被标记建议修正"
target = needs[0]
dlg = EditorDialog(app, app.store, target, on_saved=app._on_editor_saved)
pump(1.2)

h, w = dlg.mask.shape[:2]
dlg._push_undo()
dlg._stroke((w // 2, h // 2), (w // 2 + 20, h // 2), 0)
dlg._undo_op()
dlg._redo_op()
dlg._undo_op()

dlg._push_undo()
dlg.mask[:] = 0
dlg.mask[h // 4:h * 3 // 4, w // 4:w * 3 // 4] = 255
dlg._save()
pump(0.5)

assert target.verified is True, "保存后应标记为已确认"
assert target.needs_fix is False
assert (app.store.cache_dir / target.guess_file).exists()

# ESC：从游戏返回主页面
app.start_game()
assert app.view == "game"
app._on_escape()
assert app.view == "home"

app.destroy()
print("SMOKE_OK")
'''


def test_app_smoke(tmp_path):
    img_dir = tmp_path / "图片"
    img_dir.mkdir()
    extra_dir = tmp_path / "外部图片"
    extra_dir.mkdir()
    code = (CHILD_SCRIPT
            .replace("__ROOT__", str(ROOT))
            .replace("__IMG_DIR__", str(img_dir))
            .replace("__EXTRA_DIR__", str(extra_dir)))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, cwd=str(ROOT),
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "TclError" in output:
        pytest.skip(f"无图形环境：{output.strip()[-300:]}")
    assert proc.returncode == 0, output
    assert "SMOKE_OK" in proc.stdout