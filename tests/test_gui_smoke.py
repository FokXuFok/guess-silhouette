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
import numpy as np
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
# 编辑器窗口必须全屏(用户要求)；tk 返回值可能是 int 1 或 str '1'
assert str(dlg.attributes("-fullscreen")) == "1", "编辑器应全屏打开"
# 画笔：擦一下再撤销、重做
h, w = dlg.mask.shape[:2]
dlg._push_undo()
dlg._stroke((w // 2, h // 2), (w // 2 + 20, h // 2), 0)
dlg._undo_op()
dlg._redo_op()
dlg._undo_op()

# 回归:恢复模式左键必须写 255(曾与擦除等价,恢复功能形同虚设)
from types import SimpleNamespace
dlg.mask[:, :] = 0
cx = dlg.offset[0] + int(w * dlg.scale / 2)
cy = dlg.offset[1] + int(h * dlg.scale / 2)

dlg.mode.set("restore")
dlg._on_press(SimpleNamespace(x=cx, y=cy), restore=False)
assert int(dlg.mask[h // 2, w // 2]) == 255, "恢复模式左键应恢复角色(255)"

dlg.mode.set("erase")
dlg._on_press(SimpleNamespace(x=cx, y=cy), restore=False)
assert int(dlg.mask[h // 2, w // 2]) == 0, "擦除模式左键应去除背景(0)"

dlg.mode.set("restore")
dlg._on_press(SimpleNamespace(x=cx, y=cy), restore=True)
assert int(dlg.mask[h // 2, w // 2]) == 0, "恢复模式右键应临时反向为擦除(0)"

dlg.mode.set("erase")
dlg._on_press(SimpleNamespace(x=cx, y=cy), restore=True)
assert int(dlg.mask[h // 2, w // 2]) == 255, "擦除模式右键应临时反向为恢复(255)"

# 快速选择(魔棒):点击按颜色容差选中连通区域,恢复模式整片设为角色/去除模式整片去掉
lx = lambda fx: dlg.offset[0] + int(w * dlg.scale * fx)
ly = lambda fy: dlg.offset[1] + int(h * dlg.scale * fy)
dlg.rgb[:, : w // 2] = (200, 60, 90)   # 左半:角色色
dlg.rgb[:, w // 2:] = (40, 90, 200)    # 右半:背景色

dlg.mask[:, :] = 0
dlg.mode.set("magic_restore")
dlg._on_press(SimpleNamespace(x=lx(0.2), y=ly(0.5)), restore=False)
assert int(dlg.mask[h // 2, int(w * 0.2)]) == 255, "魔棒恢复:点击的相似颜色区域应整片恢复(255)"
assert int(dlg.mask[5, int(w * 0.45)]) == 255, "魔棒应选中远处连通同色区域(笔刷圆够不到)"
assert int(dlg.mask[h // 2, int(w * 0.8)]) == 0, "魔棒不应影响另一侧不同颜色区域"
dlg._undo_op()
assert int(dlg.mask[h // 2, int(w * 0.2)]) == 0, "魔棒应可撤销"

dlg.mask[:, :] = 255
dlg.mode.set("magic_erase")
dlg._on_press(SimpleNamespace(x=lx(0.2), y=ly(0.5)), restore=True)
assert int(dlg.mask[5, int(w * 0.45)]) == 0, "魔棒去除:远处连通同色区域应整片去掉(0)"
assert int(dlg.mask[h // 2, int(w * 0.8)]) == 255, "魔棒去除不应影响另一侧不同颜色区域"

# 渐变背景:容差逐像素传播会让 floodFill 沿渐变蔓延覆盖整片;
# FIXED_RANGE + 颜色模型分类后,只应选中与种子颜色分布相近的左端
dlg.mask[:, :] = 255
grad = np.zeros((h, w, 3), np.uint8)
ramp = np.linspace(0, 255, w, dtype=np.uint8)   # 左黑渐变到右白
grad[:] = ramp[:, None]
dlg.rgb[:] = grad                               # 整图左黑→右白渐变
dlg.mode.set("magic_erase")
dlg.magic_tol.set(20)                           # 容差 20:左端 (0,0,0), 右端差距远超 20
dlg._on_press(SimpleNamespace(x=lx(0.1), y=ly(0.5)), restore=False)
left_hit = int((dlg.mask[:, : w // 4] == 0).sum())
right_hit = int((dlg.mask[:, 3 * w // 4:] == 0).sum())
assert right_hit == 0, "渐变右端颜色与种子分布差异大,不应被选中"
assert left_hit > 0, "渐变左端颜色与种子接近,应被选中"

# 相近色区分(PS 快速选择核心能力):残留块与角色块颜色都接近灰色背景,
# 点击残留块应精确选中残留,不吞颜色相近的角色块
dlg.rgb[:] = (128, 128, 128)                    # 背景:灰
dlg.rgb[h // 4: 3 * h // 4, w // 4: w // 2] = (140, 130, 120)   # 角色块(左)
dlg.rgb[h // 4: 3 * h // 4, w // 2 + 20: 3 * w // 4] = (160, 150, 130)  # 残留块(右,颜色接近角色)
dlg.mask[:, :] = 255
dlg.magic_tol.set(16)
dlg.mode.set("magic_erase")
dlg._on_press(SimpleNamespace(x=lx(0.63), y=ly(0.5)), restore=False)
residual_center = int(dlg.mask[h // 2, w // 2 + 20 + 10])
subject_center = int(dlg.mask[h // 2, w // 4 + 10])
bg_center = int(dlg.mask[2, 2])
assert residual_center == 0, "应精确去除点击的残留块"
assert subject_center == 255, "颜色相近的角色块不应被吞掉"
assert bg_center == 255, "背景灰不应被顺带去除"

# 点击已是目标态的区域:无操作(去除模式点在纯背景上,mask 已是 0)
dlg.mask[:, :] = 0
before = dlg.mask.copy()
dlg._on_press(SimpleNamespace(x=lx(0.95), y=ly(0.95)), restore=False)
assert int((dlg.mask != before).sum()) == 0, "去除模式点在已是背景的区域应无操作"

# 极大容差 + 整图近似同色:面积上限保护,应拒绝操作而非全图生效
dlg.rgb[:] = (128, 128, 128)                    # 全图灰,任何点都是同色
dlg.mask[:, :] = 255
dlg.magic_tol.set(64)                           # 全图同色 → floodFill 会选满 100%
dlg._on_press(SimpleNamespace(x=lx(0.5), y=ly(0.5)), restore=False)
cover = float((dlg.mask == 0).mean())
assert cover < 0.7, f"面积上限保护:全图近似同色时应拒绝(防止整图被误操作),实际覆盖 {cover:.0%}"

# 视图缩放:滚轮缩放以鼠标为锚(锚点下图像坐标不变),复位回适应窗口
dlg._reset_view()
pump(0.2)
fit_scale = dlg.scale
wx, wy = dlg.offset[0] + int(w * dlg.scale * 0.3), dlg.offset[1] + int(h * dlg.scale * 0.4)
before_img = dlg._to_image_xy(wx, wy)
dlg._zoom(2.0, wx, wy)
pump(0.2)
assert abs(dlg.scale - fit_scale * 2) < 1e-6, "放大 2 倍应生效"
after_img = dlg._to_image_xy(wx, wy)
assert abs(after_img[0] - before_img[0]) < 0.5 and abs(after_img[1] - before_img[1]) < 0.5, \
    "缩放锚点下的图像坐标应保持不动"
assert dlg._user_view is True
# 放大视图下画笔/魔棒坐标换算仍正确
dlg.mode.set("erase")
dlg._on_press(SimpleNamespace(x=wx, y=wy), restore=False)
ix, iy = int(before_img[0]), int(before_img[1])
assert dlg.mask[iy, ix] == 0, "放大视图下画笔应落在正确的图像位置"
dlg._reset_view()
pump(0.2)
assert abs(dlg.scale - fit_scale) < 1e-6 and dlg._user_view is False, "复位应回到适应窗口"

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
app.update()
# 全屏游戏中右上角必须有【修正这张】悬浮按钮(否则开始后没法修正剪影)
assert app.fix_btn.place_info(), "全屏游戏中应显示【修正这张】"
app._on_escape()
assert app.view == "home"
app.update()
assert not app.fix_btn.place_info(), "非游戏视图不应显示悬浮【修正这张】"

app.destroy()
print("SMOKE_OK")
# 与真实程序退出方式一致(guess.py 同样用 os._exit 硬退):
# 规避 Tk + onnxruntime 混合环境下偶发的解释器退出挂起
import os as _os
import sys as _sys
_sys.stdout.flush()
_os._exit(0)
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