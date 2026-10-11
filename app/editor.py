"""手动修正编辑器：矩形框选 / 擦除 / 恢复 / 撤销重做 / 容差 / 羽化。

现场不用修，活动前把效果差的图修一遍即可。
"""
import logging
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import imaging

log = logging.getLogger(__name__)

BG = "#0E0F14"
PANEL_BG = "#15171E"
FG = "#F2F3F7"
FG_DIM = "#A7ABB8"
ACCENT = "#5B8DFF"
REMOVED_TINT = np.array([214, 76, 76], np.float32)  # 被去掉的背景：红色半透明
FONT = "Microsoft YaHei UI"

TIER_LABEL = {
    imaging.TIER_AI: "AI 智能抠图(isnet)",
    imaging.TIER_ALPHA: "透明背景（alpha）",
    imaging.TIER_SOLID: "纯色背景",
    imaging.TIER_GRADIENT: "渐变背景",
    imaging.TIER_COMPLEX: "复杂背景（GrabCut）",
}


class EditorDialog(tk.Toplevel):
    """显示原图 + 蒙版（红色标出去掉的背景），支持画笔与矩形重抠。"""

    def __init__(self, parent, store, entry, on_saved=None):
        super().__init__(parent)
        self.title(f"手动修正 - {entry.name}")
        self.configure(bg=BG)
        self.transient(parent)
        try:
            self.attributes("-fullscreen", True)  # 修正窗口全屏（用户要求）
        except tk.TclError:
            try:
                self.state("zoomed")
            except tk.TclError:
                pass
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.store = store
        self.entry = entry
        self.on_saved = on_saved

        self.rgba = store.load_work_image(entry)
        mask = store.load_mask(entry)
        if mask.shape[:2] != self.rgba.shape[:2]:
            mask = np.asarray(
                Image.fromarray(mask).resize((self.rgba.shape[1], self.rgba.shape[0]), Image.NEAREST)
            )
        self.mask = mask.astype(np.uint8)
        self.rgb = np.ascontiguousarray(self.rgba[..., :3])
        self.h, self.w = self.mask.shape[:2]

        # 状态
        self.mode = tk.StringVar(value="erase")
        self.brush = tk.IntVar(value=48)
        self.magic_tol = tk.IntVar(value=16)
        self.tol = tk.DoubleVar(value=float(store.config.tolerance))
        self.feather = tk.DoubleVar(value=float(store.config.feather_px))
        self.scale = 1.0
        self.offset = (0, 0)
        self._photo = None
        self._last_pt = None
        self._rect_start = None
        self._rect_item = None
        self._brush_cursor = None       # 笔刷大小预览圈
        self._undo: list = []
        self._redo: list = []
        self._tol_job = None
        self._dirty = True
        self._busy = False
        self._ready = False
        self._task_queue: queue.Queue = queue.Queue()
        self._closing = False

        self._build_widgets()
        self._bind_events()
        self.after(60, self._tick)
        self.after(600, lambda: setattr(self, "_ready", True))

    # ---------- 界面 ----------

    def _build_widgets(self):
        self.canvas = tk.Canvas(self, bg="#0A0B0F", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(self, bg=PANEL_BG, width=290)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        tk.Label(panel, text=self.entry.name, bg=PANEL_BG, fg=FG, font=(FONT, 14, "bold"),
                 wraplength=260, justify="left").pack(anchor="w", padx=16, pady=(16, 2))
        info = TIER_LABEL.get(self.entry.tier, "未知")
        tk.Label(panel, text=f"自动档位：{info}", bg=PANEL_BG, fg=FG_DIM,
                 font=(FONT, 10)).pack(anchor="w", padx=16)
        if self.entry.reasons:
            tk.Label(panel, text="；".join(self.entry.reasons), bg=PANEL_BG, fg="#E0A84C",
                     font=(FONT, 10), wraplength=260, justify="left").pack(anchor="w", padx=16, pady=(2, 0))

        self.hint = tk.Label(panel, text="红色 = 被去掉的背景", bg=PANEL_BG, fg=FG_DIM, font=(FONT, 10))
        self.hint.pack(anchor="w", padx=16, pady=(10, 0))

        tk.Label(panel, text="工具", bg=PANEL_BG, fg=FG, font=(FONT, 11, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        for text, val in (("矩形框选主体（重新抠图）", "rect"),
                          ("擦除背景（去掉角色）", "erase"),
                          ("恢复角色（找回被误删部分）", "restore"),
                          ("快速选择去除（点背景整片去掉）", "magic_erase"),
                          ("快速选择恢复（点角色整片找回）", "magic_restore")):
            tk.Radiobutton(panel, text=text, variable=self.mode, value=val, bg=PANEL_BG,
                           fg=FG, selectcolor=BG, activebackground=PANEL_BG, activeforeground=FG,
                           font=(FONT, 10), anchor="w").pack(anchor="w", padx=10)

        tk.Label(panel, text="笔刷大小", bg=PANEL_BG, fg=FG, font=(FONT, 10)).pack(anchor="w", padx=16, pady=(12, 0))
        tk.Scale(panel, from_=8, to=200, orient="horizontal", variable=self.brush, bg=PANEL_BG,
                 fg=FG, troughcolor=BG, highlightthickness=0, length=240).pack(padx=12)

        tk.Label(panel, text="快速选择容差（越大选得越多）", bg=PANEL_BG, fg=FG, font=(FONT, 10)).pack(anchor="w", padx=16, pady=(8, 0))
        tk.Scale(panel, from_=4, to=64, orient="horizontal", variable=self.magic_tol, bg=PANEL_BG,
                 fg=FG, troughcolor=BG, highlightthickness=0, length=240).pack(padx=12)

        tk.Label(panel, text="容差（重跑自动抠图）", bg=PANEL_BG, fg=FG, font=(FONT, 10)).pack(anchor="w", padx=16, pady=(8, 0))
        if self.entry.tier in (imaging.TIER_SOLID, imaging.TIER_GRADIENT):
            tk.Scale(panel, from_=5, to=60, orient="horizontal", variable=self.tol, bg=PANEL_BG,
                     fg=FG, troughcolor=BG, highlightthickness=0, length=240,
                     command=self._on_tol).pack(padx=12)
        else:
            tk.Scale(panel, from_=5, to=60, orient="horizontal", variable=self.tol, bg=PANEL_BG,
                     fg=FG_DIM, troughcolor=BG, highlightthickness=0, length=240,
                     state="disabled").pack(padx=12)
            tk.Label(panel, text="当前图不适用容差，请用矩形/画笔修正", bg=PANEL_BG, fg=FG_DIM,
                     font=(FONT, 9), wraplength=250, justify="left").pack(anchor="w", padx=16)

        tk.Label(panel, text="边缘羽化（保存后生效）", bg=PANEL_BG, fg=FG, font=(FONT, 10)).pack(anchor="w", padx=16, pady=(8, 0))
        tk.Scale(panel, from_=0, to=4, resolution=0.5, orient="horizontal", variable=self.feather,
                 bg=PANEL_BG, fg=FG, troughcolor=BG, highlightthickness=0, length=240).pack(padx=12)

        row = tk.Frame(panel, bg=PANEL_BG)
        row.pack(fill="x", padx=16, pady=(16, 4))
        for text, cmd in (("撤销", self._undo_op), ("重做", self._redo_op)):
            tk.Button(row, text=text, command=cmd, takefocus=0, bg="#1E212B", fg=FG, relief="flat",
                      activebackground="#262A36", activeforeground=FG, font=(FONT, 10), width=8).pack(
                side="left", padx=(0, 8))

        tk.Button(panel, text="保存并继续修正", command=self._save, takefocus=0, bg=ACCENT, fg="white",
                  relief="flat", activebackground="#4478E0", activeforeground="white",
                  font=(FONT, 12, "bold"), pady=6).pack(fill="x", padx=16, pady=(14, 4))
        tk.Button(panel, text="取消", command=self._cancel, takefocus=0, bg="#1E212B", fg=FG,
                  relief="flat", activebackground="#262A36", activeforeground=FG,
                  font=(FONT, 10)).pack(fill="x", padx=16, pady=(0, 14))
        tk.Label(panel, text="左键 = 当前工具\n右键 = 临时反向擦/恢复\n快速选择:点一下整片生效\nCtrl+Z 撤销 / Ctrl+Y 重做",
                 bg=PANEL_BG, fg=FG_DIM, font=(FONT, 9), justify="left").pack(anchor="w", padx=16, pady=(6, 0))

    def _bind_events(self):
        c = self.canvas
        c.bind("<Button-1>", lambda e: self._on_press(e, restore=False))
        c.bind("<B1-Motion>", lambda e: self._on_motion(e, restore=False))
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Button-3>", lambda e: self._on_press(e, restore=True))
        c.bind("<B3-Motion>", lambda e: self._on_motion(e, restore=True))
        c.bind("<ButtonRelease-3>", self._on_release)
        c.bind("<Motion>", self._on_hover, add="+")
        c.bind("<Leave>", lambda e: self._hide_brush_cursor())
        c.bind("<Configure>", lambda e: setattr(self, "_dirty", True))
        self.bind("<Escape>", lambda e: self._cancel())
        self.bind("<Control-z>", lambda e: self._undo_op())
        self.bind("<Control-y>", lambda e: self._redo_op())

    # ---------- 渲染 ----------

    def _tick(self):
        if self._closing:
            return
        self._poll_task()
        if self._dirty:
            self._dirty = False
            self._redraw()
        self.after(40, self._tick)

    def _make_overlay(self) -> Image.Image:
        rgb = self.rgb.copy()
        removed = self.mask < 128
        rgb[removed] = (rgb[removed].astype(np.float32) * 0.45 + REMOVED_TINT * 0.55).astype(np.uint8)
        return Image.fromarray(rgb)

    def _redraw(self):
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        scale = min(cw / self.w, ch / self.h)
        self.scale = max(scale, 1e-6)
        tw, th = max(1, int(self.w * self.scale)), max(1, int(self.h * self.scale))
        img = self._make_overlay().resize((tw, th), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.offset = ((cw - tw) // 2, (ch - th) // 2)
        self.canvas.delete("all")
        self._rect_item = None
        self._brush_cursor = None
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self._photo)

    # ---------- 坐标与绘制 ----------

    def _to_image_xy(self, x, y):
        return ((x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale)

    def _on_press(self, event, restore: bool):
        if self._busy:
            return
        if self.mode.get() in ("magic_erase", "magic_restore"):
            self._last_pt = None  # 防止切模式后拖动误出笔刷线
            self._push_undo()
            self._magic_select(self._to_image_xy(event.x, event.y),
                               erase=(self.mode.get() == "magic_erase"))
            self._dirty = True
            return
        if self.mode.get() == "rect":
            self._rect_start = (event.x, event.y)
            if self._rect_item:
                self.canvas.delete(self._rect_item)
            self._rect_item = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#FFD24C", dash=(6, 4), width=2)
            return
        self._push_undo()
        self._last_pt = self._to_image_xy(event.x, event.y)
        self._stroke(self._last_pt, self._last_pt, self._stroke_value(restore))
        self._dirty = True

    def _stroke_value(self, right_button: bool) -> int:
        """笔刷落笔值：擦除=0（去背景），恢复=255（找回角色）。

        左键 = 当前工具；右键 = 临时反向。恢复模式下左键也必须写 255。
        """
        erase = (self.mode.get() == "erase") != bool(right_button)
        return 0 if erase else 255

    def _on_motion(self, event, restore: bool):
        if self._busy:
            return
        if self.mode.get() == "rect":
            if self._rect_start and self._rect_item:
                self.canvas.coords(self._rect_item, self._rect_start[0], self._rect_start[1], event.x, event.y)
            return
        if self._last_pt is None:
            return
        pt = self._to_image_xy(event.x, event.y)
        self._stroke(self._last_pt, pt, self._stroke_value(restore))
        self._last_pt = pt
        self._dirty = True
        self._update_brush_cursor(event.x, event.y)

    def _on_release(self, event):
        if self.mode.get() == "rect":
            if self._rect_start and self._rect_item:
                x0 = min(self._rect_start[0], event.x)
                y0 = min(self._rect_start[1], event.y)
                x1 = max(self._rect_start[0], event.x)
                y1 = max(self._rect_start[1], event.y)
                self.canvas.delete(self._rect_item)
                self._rect_item = None
                self._rect_start = None
                ix0, iy0 = self._to_image_xy(x0, y0)
                ix1, iy1 = self._to_image_xy(x1, y1)
                ix0 = max(0, int(ix0))
                iy0 = max(0, int(iy0))
                ix1 = min(self.w, int(ix1))
                iy1 = min(self.h, int(iy1))
                rw, rh = ix1 - ix0, iy1 - iy0
                if rw >= 8 and rh >= 8:
                    self._push_undo()
                    self._run_task(lambda: imaging.mask_from_rect(self.rgba, (ix0, iy0, rw, rh)),
                                   self._apply_mask)
            return
        self._last_pt = None
        self._dirty = True

    def _stroke(self, p0, p1, val: int):
        r = max(1, int(round(self.brush.get() / 2.0 / self.scale)))
        a = (int(round(p0[0])), int(round(p0[1])))
        b = (int(round(p1[0])), int(round(p1[1])))
        if a == b:
            cv2.circle(self.mask, a, r, val, -1)
        else:
            cv2.line(self.mask, a, b, val, thickness=max(1, r * 2))

    def _magic_select(self, pt, erase: bool):
        """快速选择:PS 快速选择工具同款思路(颜色模型分类 + 连通限定)。

        参考 Paint Selection (SIGGRAPH 2009) / Adobe US8121407 的简化版:
        1. 作用域限定:去除只能改当前前景、恢复只能改当前背景——点击处已是
           目标态则提示并返回,天然杜绝"全图被改"。
        2. floodFill(FIXED_RANGE, 容差减半)取种子区域——与种子颜色相近的
           严格核心(不沿渐变蔓延)。
        3. 双直方图贝叶斯分类:种子区域 vs 其余像素的颜色分布(BGR 各 16 级),
           逐像素 log 似然比打分,>0 判为目标色;高斯模糊平滑边界。
           —— 物体内部的渐变、阴影也能被吃全,而颜色不同的邻近区域不会被吞。
        4. 只保留与点击点 8 连通的部分。
        5. 面积 >65% 拒绝(兜底防整图误操作)。
        """
        x, y = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        tol = max(1, int(self.magic_tol.get()))

        # 1. 作用域限定
        scope = self.mask > 128 if erase else self.mask <= 128
        if not scope[y, x]:
            self._hint_msg("这里已经是" + ("背景" if erase else "角色") + ",无需操作")
            return

        # 2. 种子区域
        x0, x1 = max(0, x - 1), min(self.w, x + 2)
        y0, y1 = max(0, y - 1), min(self.h, y + 2)
        seed = self.rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0).astype(np.uint8)
        work = self.rgb.copy()
        work[y, x] = int(seed[0]), int(seed[1]), int(seed[2])
        tol_eff = max(4, tol // 2)
        ff = np.zeros((self.h + 2, self.w + 2), np.uint8)
        cv2.floodFill(work, ff, (x, y), 0, (tol_eff,) * 3, (tol_eff,) * 3,
                      cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8) | 8)
        raw = (ff[1:-1, 1:-1] > 0) & scope
        if not raw.any():
            self._hint_msg("容差太小没选中东西,调大试试", warn=True)
            return

        # 3. 双直方图贝叶斯分类
        quant = self.rgb.astype(np.int32) >> 4                    # BGR 各 16 级
        idx = (quant[..., 0] << 8) | (quant[..., 1] << 4) | quant[..., 2]
        flat = idx.ravel()
        n_bins = 4096
        eps = 0.5
        obj_h = np.bincount(flat[raw.ravel()], minlength=n_bins).astype(np.float64)
        ref_h = np.bincount(flat[(~raw).ravel()], minlength=n_bins).astype(np.float64)
        obj_p = (obj_h + eps) / (obj_h.sum() + eps * n_bins)
        ref_p = (ref_h + eps) / (ref_h.sum() + eps * n_bins)
        score = (np.log(obj_p)[flat] - np.log(ref_p)[flat]).reshape(self.h, self.w)
        score = cv2.GaussianBlur(score.astype(np.float32), (5, 5), 1.5)
        cand = (score > 0) & scope
        cand |= raw                                               # 点击处必须在结果里

        # 4. 连通限定:只保留含点击点的连通域
        n, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        lid = int(labels[y, x])
        region = labels == lid if lid > 0 else raw

        # 5. 面积上限
        cover = float(region.mean())
        if cover > 0.65:
            self._hint_msg(f"⚠ 选区过大({cover:.0%}),调小容差或换个点", warn=True)
            return

        self.mask[region] = 0 if erase else 255
        self._hint_msg(f"红色 = 被去掉的背景  ·  上次选区 {cover:.0%}")

    def _hint_msg(self, text: str, warn: bool = False):
        if self.hint.winfo_exists():
            self.hint.configure(text=text, fg="#E0A84C" if warn else FG_DIM)

    # ---------- 笔刷光标预览 ----------

    def _on_hover(self, event):
        if self._busy or self.mode.get() != "erase" and self.mode.get() != "restore":
            self._hide_brush_cursor()
            return
        self._update_brush_cursor(event.x, event.y)

    def _update_brush_cursor(self, cx, cy):
        """在画布上画一个跟随鼠标的圆圈，提示实际笔刷覆盖范围（提升涂抹精准度）。"""
        if self._brush_cursor:
            self.canvas.delete(self._brush_cursor)
        r = max(2.0, self.brush.get() / 2.0 * self.scale)
        self._brush_cursor = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#FFFFFF", dash=(3, 3))

    def _hide_brush_cursor(self):
        if self._brush_cursor:
            self.canvas.delete(self._brush_cursor)
            self._brush_cursor = None

    # ---------- 撤销 / 重做 ----------

    def _snapshot(self):
        return np.packbits(self.mask >= 128), self.mask.shape

    def _push_undo(self):
        self._undo.append(self._snapshot())
        if len(self._undo) > 30:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snap):
        packed, shape = snap
        self.mask = (np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape) * 255).astype(np.uint8)

    def _undo_op(self):
        if not self._undo or self._busy:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self._dirty = True

    def _redo_op(self):
        if not self._redo or self._busy:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self._dirty = True

    # ---------- 容差重算（后台线程） ----------

    def _on_tol(self, _value=None):
        if not self._ready or self._busy:
            return
        if self._tol_job:
            self.after_cancel(self._tol_job)
        self._tol_job = self.after(500, self._reapply_auto)

    def _reapply_auto(self):
        self._tol_job = None
        if self._busy or not self.winfo_exists():
            return
        tol = float(self.tol.get())
        self._push_undo()
        self._run_task(lambda: imaging.auto_mask(self.rgba, tolerance=tol).mask, self._apply_mask)

    def _apply_mask(self, mask):
        if mask is not None and mask.shape[:2] == (self.h, self.w):
            self.mask = mask.astype(np.uint8)
        self._dirty = True

    # ---------- 后台任务 ----------

    def _run_task(self, fn, done):
        if self._busy:
            return
        self._busy = True
        self.hint.configure(text="处理中，请稍候…", fg="#E0A84C")

        def work():
            try:
                self._task_queue.put(("ok", fn(), done))
            except Exception as exc:  # noqa: BLE001 - 后台任务需要兜底
                log.exception("后台处理失败")
                self._task_queue.put(("err", exc, done))

        threading.Thread(target=work, daemon=True).start()

    def _poll_task(self):
        try:
            kind, payload, done = self._task_queue.get_nowait()
        except queue.Empty:
            return
        self._busy = False
        self.hint.configure(text="红色 = 被去掉的背景", fg=FG_DIM)
        if kind == "ok":
            done(payload)
        else:
            messagebox.showerror("处理失败", str(payload), parent=self)

    # ---------- 保存 / 取消 ----------

    def _save(self):
        if self._busy:
            return
        try:
            self.store.save_edit(self.entry, self.mask, feather_px=float(self.feather.get()))
        except Exception as exc:  # noqa: BLE001 - 保存失败要提示而不是崩
            log.exception("保存修正失败")
            messagebox.showerror("保存失败", str(exc), parent=self)
            return
        self._closing = True
        if self.on_saved:
            self.on_saved(self.entry)
        self.destroy()

    def _cancel(self):
        if self._undo and not messagebox.askyesno("取消修正", "放弃本次修改吗？", parent=self):
            return
        self._closing = True
        self.destroy()