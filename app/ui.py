"""Tkinter 主界面。

视图：
- 主页面：标题 + 【开始】【添加图片】两个选项 + 图片就绪状态
- 游戏页：黑影/原图 + 答案；空格揭晓 → 再按空格下一题；ESC 或【结束本轮】回主页面
- 完毕页：一轮全部展示完后提示，空格/ESC/按钮回主页面

视觉：深色舞台感、单一强调色、精确间距；按钮为自绘圆角（悬停提亮 / 按下变暗）。
"""
import logging
import os
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .config import Config
from .deck import Deck
from .editor import EditorDialog
from .library import add_images_to
from .store import Store

log = logging.getLogger(__name__)

# ---------- 设计令牌(深色舞台:单一强调色 + 分层灰阶 + 发丝线) ----------
BG = "#0E0F14"            # 窗口底
SURFACE = "#15171E"       # 面板/顶栏
BORDER = "#242833"        # 发丝线
FG = "#F2F3F7"            # 主文字
FG_DIM = "#A7ABB8"        # 次级文字
FG_FAINT = "#6C7080"      # 弱文字
ACCENT = "#5B8DFF"        # 强调色(唯一)
ACCENT_HOVER = "#6E9BFF"
ACCENT_PRESS = "#4478E0"
WARN = "#E0A84C"          # 提醒态
ERROR = "#E0705F"         # 错误态
BTN_BG = "#1E212B"        # 次级按钮
BTN_HOVER = "#262A36"
BTN_PRESS = "#181B23"
DISABLED_BG = "#191B22"
FONT = "Microsoft YaHei UI"

HOME = "home"
GAME = "game"
DONE = "done"

_round_cache: dict = {}


def _shorten_path(path: str, max_chars: int = 30) -> str:
    """顶栏路径显示:过长时保留尾部(文件夹名在末尾最重要)。"""
    if len(path) <= max_chars:
        return path
    return "…" + path[-(max_chars - 1):]


def _round_image(w: int, h: int, radius: int, fill: str) -> ImageTk.PhotoImage:
    """生成抗锯齿圆角矩形图片(4x 超采样),按参数缓存。"""
    key = (w, h, radius, fill)
    img = _round_cache.get(key)
    if img is not None:
        return img
    scale = 4
    big = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle(
        (0, 0, w * scale - 1, h * scale - 1), radius * scale, fill=fill)
    img = ImageTk.PhotoImage(big.resize((w, h), Image.LANCZOS))
    _round_cache[key] = img
    return img


class FancyButton(tk.Canvas):
    """自绘圆角按钮:悬停提亮、按下变暗、禁用降饱和。

    兼容 `configure(state=...)` 与 `configure(text=...)`(与 tk.Button 用法一致)。
    """

    def __init__(self, master, text, command=None, *, primary=False, font_size=15,
                 pad_x=30, pad_y=11, radius=None, parent_bg=BG, fixed_width=None):
        super().__init__(master, highlightthickness=0, bd=0, bg=parent_bg, takefocus=0)
        self._font = tkfont.Font(family=FONT, size=font_size,
                                 weight="bold" if primary else "normal")
        self._text = text
        self._command = command
        self._primary = primary
        self._state = "normal"
        self._hover = False
        self._pressed = False
        tw = self._font.measure(text)
        th = self._font.metrics("linespace")
        # 注意:不能叫 self._w/self._h——Tkinter 内部用 _w 存 widget 路径名
        self._bw = fixed_width or (tw + pad_x * 2)
        self._bh = th + pad_y * 2
        self._radius = radius if radius is not None else min(14, self._bh // 3)
        self.configure(width=self._bw, height=self._bh, cursor="hand2")
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._render()

    # ---- 状态色 ----
    def _colors(self):
        if self._state == "disabled":
            return DISABLED_BG, FG_FAINT
        if self._primary:
            if self._pressed:
                return ACCENT_PRESS, "#FFFFFF"
            if self._hover:
                return ACCENT_HOVER, "#FFFFFF"
            return ACCENT, "#FFFFFF"
        if self._pressed:
            return BTN_PRESS, FG
        if self._hover:
            return BTN_HOVER, FG
        return BTN_BG, FG

    def _render(self):
        fill, fg = self._colors()
        self.delete("all")
        self.create_image(0, 0, anchor="nw",
                          image=_round_image(self._bw, self._bh, self._radius, fill))
        self.create_text(self._bw // 2, self._bh // 2 + 1, text=self._text,
                         fill=fg, font=self._font)

    # ---- 事件 ----
    def _on_enter(self, _e=None):
        if self._state == "disabled":
            return
        self._hover = True
        self._render()

    def _on_leave(self, _e=None):
        self._hover = False
        self._pressed = False
        self._render()

    def _on_press(self, _e=None):
        if self._state == "disabled":
            return
        self._pressed = True
        self._render()

    def _on_release(self, _e=None):
        if self._state == "disabled":
            return
        was = self._pressed
        self._pressed = False
        self._render()
        if was and self._command is not None:
            self._command()

    # ---- 兼容 tk.Button 的 configure ----
    def configure(self, cnf=None, **kw):
        state = kw.pop("state", None)
        if state is not None:
            self.set_state(state)
        text = kw.pop("text", None)
        if text is not None:
            self._text = text
            self._render()
        if cnf or kw:
            super().configure(cnf, **kw)

    config = configure

    def set_state(self, state: str):
        self._state = state
        self.configure(cursor="hand2" if state != "disabled" else "")
        self._render()

    def set_text(self, text: str):
        self._text = text
        self._render()

    def invoke(self):
        """模拟点击(便于测试/脚本)。"""
        if self._state != "disabled" and self._command is not None:
            self._command()


class App(tk.Tk):
    def __init__(self, image_dir: Path, config: Config):
        super().__init__()
        self.title("猜剪影")
        self.configure(bg=BG)
        self.geometry("1280x760")
        self.minsize(960, 600)
        self.option_add("*Font", (FONT, 10))

        self.config_obj = config
        self.image_dir = Path(image_dir)
        self.store = Store(self.image_dir, config)
        self.entries: list = []
        self.deck: Deck | None = None
        self.revealed = False
        self.fullscreen = False
        self.view = HOME
        self._cur_pil: Image.Image | None = None
        self._photo = None
        self._img_cache: dict = {}
        self._photo_cache: dict = {}
        self._fit_job = None
        self._note_job = None
        self._editor = None
        self._syncing = False
        self._can_start = False

        self._build_home()
        self._build_game()
        self._build_done()
        self._build_menu()
        self._relayout()
        self._bind_keys()

    # ================= 主页面 =================

    def _build_home(self):
        self.home_frame = tk.Frame(self, bg=BG)
        center = tk.Frame(self.home_frame, bg=BG)
        center.place(relx=0.5, rely=0.46, anchor="center")

        tk.Label(center, text="猜 剪 影", bg=BG, fg=FG, font=(FONT, 64, "bold")).pack()
        tk.Frame(center, bg=ACCENT, width=52, height=3).pack(pady=(16, 16))
        tk.Label(center, text="动漫社活动 · 投影猜角色", bg=BG, fg=FG_DIM,
                 font=(FONT, 15)).pack(pady=(0, 30))

        status_row = tk.Frame(center, bg=BG)
        status_row.pack(pady=(0, 26))
        self.status_dot = tk.Canvas(status_row, width=8, height=8, bg=BG,
                                    highlightthickness=0)
        self._dot = self.status_dot.create_oval(0, 0, 8, 8, fill=FG_FAINT, outline="")
        self.status_dot.pack(side="left", padx=(0, 9))
        self.home_status = tk.Label(status_row, text="正在检查图片…", bg=BG, fg=FG_DIM,
                                    font=(FONT, 15))
        self.home_status.pack(side="left")

        row = tk.Frame(center, bg=BG)
        row.pack()
        self.start_btn = FancyButton(row, text="开 始", command=self._btn(self.start_game),
                                     primary=True, font_size=23, pad_x=52, pad_y=15)
        self.start_btn.pack(side="left", padx=10)
        self.add_btn = FancyButton(row, text="添加图片", command=self._btn(self.add_images),
                                   font_size=23, pad_x=40, pad_y=15)
        self.add_btn.pack(side="left", padx=10)

        self.home_note = tk.Label(center, text="", bg=BG, fg=WARN, font=(FONT, 12))
        self.home_note.pack(pady=(20, 0))

        tk.Label(self.home_frame, text="空格 / 回车：开始 · F11：全屏开关 · Alt+F4：退出程序",
                 bg=BG, fg=FG_FAINT, font=(FONT, 10)).place(relx=0.5, rely=0.97, anchor="s")

    # ================= 游戏页 =================

    def _build_game(self):
        # 顶部：文件夹 / 进度 / 状态 + 辅助按钮（全屏时隐藏），底下一根发丝线
        self.top = tk.Frame(self, bg=SURFACE)
        inner = tk.Frame(self.top, bg=SURFACE)
        inner.pack(fill="x")
        self.folder_label = tk.Label(inner, text=_shorten_path(str(self.image_dir)),
                                     bg=SURFACE, fg=FG_FAINT, font=(FONT, 10))
        self.folder_label.pack(side="left", padx=(16, 8), pady=10)
        self.progress_label = tk.Label(inner, text="", bg=SURFACE, fg=FG, font=(FONT, 12, "bold"))
        self.progress_label.pack(side="left", padx=8)
        self.status_label = tk.Label(inner, text="", bg=SURFACE, fg=FG_DIM, font=(FONT, 10))
        self.status_label.pack(side="left", padx=8)
        for i, (text, cmd) in enumerate((("选择文件夹", self.choose_folder),
                                         ("打开文件夹", self.open_folder),
                                         ("重新生成", self.regenerate))):
            # 最右按钮额外留白,避免被右上角悬浮的【结束本轮】盖住
            pad_right = 168 if i == 0 else 8
            FancyButton(inner, text=text, command=self._btn(cmd), font_size=10,
                        pad_x=14, pad_y=6, parent_bg=SURFACE).pack(
                side="right", padx=(0, pad_right), pady=8)
        tk.Frame(self.top, bg=BORDER, height=1).pack(fill="x")

        # 中间：大图区域
        self.image_label = tk.Label(self, bg=BG, bd=0)
        self.image_label.bind("<Configure>", self._on_area_resize)

        # 底部：答案 + 按钮
        self.bottom = tk.Frame(self, bg=BG)
        self.answer_label = tk.Label(self.bottom, text="", bg=BG, fg=FG, font=(FONT, 42, "bold"))
        self.answer_label.pack(side="top", pady=(8, 6))
        self.buttons = tk.Frame(self.bottom, bg=BG)
        for text, cmd, accent in (("显示答案", self.reveal, True),
                                  ("下一题", self.next_question, False),
                                  ("跳过", self.skip, False),
                                  ("修正这张", lambda: self._open_editor(), False)):
            FancyButton(self.buttons, text=text, command=self._btn(cmd), primary=accent,
                        font_size=13, pad_x=22, pad_y=9).pack(side="left", padx=6)

        # 右上角：结束本轮（全屏时也可见，主持人也能用鼠标点）
        self.finish_btn = FancyButton(self, text="结束本轮 (Esc)",
                                      command=self._btn(self.end_round),
                                      font_size=10, pad_x=16, pad_y=7, parent_bg=BG)
        # 修正这张：全屏游戏中悬浮在【结束本轮】左侧（窗口模式用底部按钮行，不重复显示）
        self.fix_btn = FancyButton(self, text="修正这张",
                                   command=self._btn(lambda: self._open_editor()),
                                   font_size=10, pad_x=16, pad_y=7, parent_bg=BG)

    # ================= 展示完毕页 =================

    def _build_done(self):
        self.done_frame = tk.Frame(self, bg=BG)
        center = tk.Frame(self.done_frame, bg=BG)
        center.place(relx=0.5, rely=0.45, anchor="center")

        tk.Label(center, text="本轮已全部展示完毕", bg=BG, fg=FG,
                 font=(FONT, 44, "bold")).pack()
        self.done_sub = tk.Label(center, text="", bg=BG, fg=FG_DIM, font=(FONT, 16))
        self.done_sub.pack(pady=(14, 34))
        FancyButton(center, text="返回主页面", command=self._btn(self.end_round),
                    primary=True, font_size=18, pad_x=40, pad_y=12).pack()

        tk.Label(self.done_frame, text="按空格 / ESC 也可以返回主页面", bg=BG, fg=FG_FAINT,
                 font=(FONT, 11)).place(relx=0.5, rely=0.97, anchor="s")

    # ================= 菜单 =================

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="添加图片…", command=self.add_images)
        file_menu.add_command(label="选择图片文件夹…", command=self.choose_folder)
        file_menu.add_command(label="打开图片所在文件夹", command=self.open_folder)
        file_menu.add_command(label="重新生成全部剪影", command=self.regenerate)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.destroy)
        menubar.add_cascade(label="文件", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="全屏开关（F11）", command=self._toggle_fullscreen)
        menubar.add_cascade(label="视图", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="快捷键说明", command=self.show_help)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.configure(menu=menubar)

    def _btn(self, fn):
        """按钮点击后把焦点交还主窗口，保证键盘快捷键继续可用。"""
        def command():
            self.focus_set()
            fn()
        return command

    # ================= 视图切换 =================

    def _show_view(self, view: str):
        self.view = view
        self._relayout()
        self._apply_cursor()
        if view == HOME:
            self._refresh_home()

    def _relayout(self):
        """按当前视图排布窗口内容。"""
        self.home_frame.pack_forget()
        self.done_frame.pack_forget()
        self.top.pack_forget()
        self.bottom.pack_forget()
        self.image_label.pack_forget()
        self.finish_btn.place_forget()
        self.fix_btn.place_forget()

        if self.view == HOME:
            self.home_frame.pack(side="top", fill="both", expand=True)
            return
        if self.view == DONE:
            self.done_frame.pack(side="top", fill="both", expand=True)
            return

        # 游戏视图
        if not self.fullscreen:
            self.top.pack(side="top", fill="x")
        self.bottom.pack(side="bottom", fill="x")
        self.image_label.pack(side="top", fill="both", expand=True)
        if self.fullscreen:
            self.buttons.pack_forget()
            # 全屏时右上角保留【修正这张】+【结束本轮】，随时能修正当前图
            self.fix_btn.place(relx=1.0, rely=0.0, anchor="ne",
                               x=-14 - self.finish_btn._bw - 10, y=12)
        else:
            self.buttons.pack(side="top", pady=(0, 14))  # 底部按钮行已含【修正这张】
        self.finish_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-14, y=12)

    def _apply_cursor(self):
        """只有全屏游戏时才隐藏鼠标光标；主页面要能看到光标点按钮。"""
        hide = self.fullscreen and self.view == GAME
        cursor = "none" if hide else ""
        for w in (self, self.top, self.image_label, self.bottom, self.answer_label,
                  self.home_frame, self.done_frame):
            try:
                w.configure(cursor=cursor)
            except tk.TclError:
                pass

    # ================= 启动与处理 =================

    def start(self):
        """启动流程（不含 mainloop，便于测试 / 嵌入）。"""
        self.after(100, self._startup)

    def run(self):
        self.start()
        self.mainloop()

    def _startup(self):
        try:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._show_view(HOME)
        self.home_status.configure(text="正在检查图片…")
        self._enter_fullscreen()
        self._scan_and_process()

    def _scan_and_process(self, force: bool = False):
        if self._syncing:
            messagebox.showinfo("提示", "正在处理图片，请稍候…", parent=self)
            return
        files = self.store.scan_images()
        if not files:
            self.entries = []
            self._refresh_home()
            self._refresh_status()
            return
        self._syncing = True
        self._refresh_home()
        self._refresh_status()
        _SyncRunner(self, self.store, force, self._on_processed, total=len(files))

    def _on_processed(self, summary, error):
        self._syncing = False
        if error is not None:
            log.error("处理图片失败：%s", error)
            messagebox.showerror("处理失败", f"处理图片时出错：\n{error}", parent=self)
            self._refresh_home()
            self.home_status.configure(text="处理图片时出错，可查看 缓存/运行日志.log")
            return
        self.entries = summary.entries
        self._img_cache.clear()      # 重新生成后避免显示旧剪影
        self._photo_cache.clear()
        self._refresh_status()
        ok_entries = [e for e in summary.entries if e.ok]
        if self.view == GAME:
            if not ok_entries:
                self.end_round()
            else:
                self.deck = Deck(ok_entries, auto_restart=False)
                self.revealed = False
                self._show_current()
        self._refresh_home()

    # ================= 题面与图片 =================

    def _get_render(self, entry, reveal: bool) -> Image.Image:
        key = (entry.name, reveal)
        img = self._img_cache.get(key)
        if img is None:
            img = self.store.load_render(entry, reveal)
            if len(self._img_cache) >= 6:
                self._img_cache = {}
            self._img_cache[key] = img
        return img

    def _set_answer(self, text: str):
        """答案文字:名字过长时自动降字号,避免撑满屏幕。"""
        if len(text) <= 8:
            size = 42
        elif len(text) <= 14:
            size = 34
        else:
            size = 26
        self.answer_label.configure(text=text, font=(FONT, size, "bold"))

    def _show_current(self, reset_reveal: bool = True):
        if self.deck is None or self.deck.current is None:
            return
        entry = self.deck.current
        if reset_reveal:
            self.revealed = False
        try:
            self._cur_pil = self._get_render(entry, self.revealed)
        except Exception:  # noqa: BLE001 - 缓存读失败不该崩
            log.exception("读取缓存失败：%s", entry.name)
            self._cur_pil = None
        if self._cur_pil is None:
            self.image_label.configure(image="", text=f"无法读取：{entry.name}",
                                       fg=FG_DIM, font=(FONT, 14))
        else:
            self._fit_and_show()
        self._set_answer(entry.name if self.revealed else "？？？")
        self._update_progress()

    def _available_size(self):
        w = self.image_label.winfo_width()
        h = self.image_label.winfo_height()
        if w <= 10 or h <= 10:
            w = max(640, self.winfo_width() - 40)
            h = max(360, self.winfo_height() - 180)
        return w, h

    def _fit_and_show(self):
        if self._cur_pil is None:
            return
        w, h = self._available_size()
        iw, ih = self._cur_pil.size
        scale = min(w / iw, h / ih, 2.0)
        tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
        key = (id(self._cur_pil), tw, th)
        photo = self._photo_cache.get(key)
        if photo is None:
            photo = ImageTk.PhotoImage(self._cur_pil.resize((tw, th), Image.LANCZOS))
            self._photo_cache = {key: photo}
        self._photo = photo  # 保持引用，防止被 GC 掉
        self.image_label.configure(image=photo, text="")

    def _on_area_resize(self, _event=None):
        if self._fit_job:
            self.after_cancel(self._fit_job)
        self._fit_job = self.after(80, self._refit)

    def _refit(self):
        self._fit_job = None
        if self.view == GAME:
            self._fit_and_show()

    # ================= 游戏操作 =================

    def start_game(self):
        """【开始】/ 空格：进入游戏，本轮不重复展示，展示完即止。"""
        if self._syncing:
            messagebox.showinfo("提示", "正在处理图片，请稍候…", parent=self)
            return
        ok_entries = [e for e in self.entries if e.ok]
        if not ok_entries:
            messagebox.showinfo("提示", "还没有可用图片，先点【添加图片】。", parent=self)
            return
        self.deck = Deck(ok_entries, auto_restart=False)
        self.revealed = False
        if not self.fullscreen:
            self._enter_fullscreen()  # 进游戏自动全屏（投影用）
        self._show_view(GAME)
        self._show_current()

    def end_round(self):
        """ESC /【结束本轮】：结束本轮回主页面。"""
        self.deck = None
        self.revealed = False
        self._cur_pil = None
        self._show_view(HOME)

    def handle_space(self):
        """空格：主页面=开始；游戏中=揭晓 → 下一题黑影循环；完毕页=回主页面。"""
        if self.view == HOME:
            if self._can_start:
                self.start_game()
        elif self.view == DONE:
            self.end_round()
        elif self.view == GAME:
            if not self.revealed:
                self.reveal()
            else:
                self.next_question()

    def reveal(self):
        """揭晓当前题：白底黑影 → 黑底原图 + 角色名。"""
        if self.deck is None or self.deck.current is None or self.revealed:
            return
        self.revealed = True
        self._show_current(reset_reveal=False)

    def next_question(self):
        if self.deck is None:
            return
        self.deck.next()
        if self.deck.exhausted:
            self._show_finished()
            return
        self._show_current()

    def skip(self):
        if self.deck is None:
            return
        self.deck.skip()
        if self.deck.exhausted:
            self._show_finished()
            return
        self._show_current()

    def _show_finished(self):
        self.done_sub.configure(text=f"本轮共展示 {self.deck.total} 张图片，没有重复")
        self._show_view(DONE)

    # ================= 辅助功能 =================

    def add_images(self):
        """【添加图片】：多选图片复制进 图片 文件夹，然后自动扫描生成剪影。"""
        if self._syncing:
            messagebox.showinfo("提示", "正在处理图片，请稍候…", parent=self)
            return
        paths = filedialog.askopenfilenames(
            parent=self,
            title="选择要添加的图片（可多选，文件名就是角色名）",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")],
        )
        if not paths:
            return
        result = add_images_to(self.image_dir, [Path(p) for p in paths])
        parts = [f"已添加 {result.added} 张图片"]
        if result.renamed:
            parts.append(f"{result.renamed} 张重名已自动加序号")
        if result.skipped:
            parts.append(f"跳过 {result.skipped} 张（重复或非图片）")
        self._show_note("，".join(parts))
        self._scan_and_process()

    def _show_note(self, text: str, ms: int = 8000):
        self.home_note.configure(text=text)
        if self._note_job:
            self.after_cancel(self._note_job)
        self._note_job = self.after(ms, lambda: self.home_note.configure(text=""))

    def choose_folder(self):
        if self._syncing:
            messagebox.showinfo("提示", "正在处理图片，请稍候…", parent=self)
            return
        path = filedialog.askdirectory(parent=self, initialdir=str(self.image_dir),
                                       title="选择图片文件夹（文件名 = 角色名）")
        if not path:
            return
        self.image_dir = Path(path)
        self.folder_label.configure(text=_shorten_path(str(self.image_dir)))
        self.store = Store(self.image_dir, self.config_obj)
        self.entries = []
        self._img_cache.clear()
        self._photo_cache.clear()
        self._scan_and_process()

    def open_folder(self):
        try:
            os.startfile(str(self.image_dir))  # Windows
        except OSError:
            messagebox.showinfo("图片文件夹", str(self.image_dir), parent=self)

    def regenerate(self):
        if self._syncing:
            messagebox.showinfo("提示", "正在处理图片，请稍候…", parent=self)
            return
        if not messagebox.askyesno("重新生成", "将重新处理所有图片并覆盖已有剪影，确定吗？", parent=self):
            return
        self._scan_and_process(force=True)

    def show_help(self):
        messagebox.showinfo(
            "快捷键",
            "主页面：\n"
            "  空格 / 回车：开始\n"
            "游戏中：\n"
            "  空格：揭晓答案 → 再按一次：下一题黑影（循环）\n"
            "  →（右方向键）：直接下一题\n"
            "  Esc /【结束本轮】按钮：结束本轮回主页面\n"
            "通用：\n"
            "  F11：全屏开关\n"
            "  鼠标点击按钮也可以操作。",
            parent=self,
        )

    def _open_editor(self, entry=None):
        if entry is None:
            entry = self.deck.current if self.deck else None
        if entry is None or not entry.ok:
            messagebox.showinfo("提示", "当前没有可修正的题目。", parent=self)
            return
        if self._editor is not None and self._editor.winfo_exists():
            self._editor.lift()
            self._editor.focus_set()
            return
        self._editor = EditorDialog(self, self.store, entry, on_saved=self._on_editor_saved)

    def _on_editor_saved(self, entry):
        self._editor = None
        self._img_cache.pop((entry.name, False), None)
        self._img_cache.pop((entry.name, True), None)
        self._photo_cache.clear()
        self._refresh_status()
        self._refresh_home()
        if self.deck is not None and self.deck.current is entry:
            self.revealed = False
            self._show_current()
        nxt = self._next_needs_fix(exclude=entry)
        if nxt is not None:
            self.after(250, lambda: self._open_editor(nxt))

    def _next_needs_fix(self, exclude=None):
        for e in self.entries:
            if e.ok and e.needs_fix and not e.verified and e is not exclude:
                return e
        return None

    # ================= 全屏 =================

    def _toggle_fullscreen(self):
        if self.fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self):
        try:
            self.attributes("-fullscreen", True)
        except tk.TclError:
            return
        self.fullscreen = True
        self._relayout()
        self._apply_cursor()
        self.after(50, self._refit)

    def _exit_fullscreen(self):
        self.fullscreen = False
        try:
            self.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        self._relayout()
        self._apply_cursor()
        self.after(50, self._refit)

    # ================= 状态显示 =================

    def _update_progress(self):
        if self.deck is not None and self.deck.current is not None:
            self.progress_label.configure(text=f"第 {self.deck.shown_in_round} / {self.deck.total} 题")
        else:
            self.progress_label.configure(text="")

    def _refresh_status(self):
        n = len(self.entries)
        ok = sum(1 for e in self.entries if e.ok)
        failed = n - ok
        fix = sum(1 for e in self.entries if e.ok and e.needs_fix and not e.verified)
        parts = [f"共 {n} 张"]
        if failed:
            parts.append(f"{failed} 张打不开")
        if fix:
            parts.append(f"{fix} 张待修正")
        self.status_label.configure(text=" · ".join(parts))

    def _refresh_home(self):
        total = len(self.entries)
        ok = sum(1 for e in self.entries if e.ok)
        fix = sum(1 for e in self.entries if e.ok and e.needs_fix and not e.verified)
        if self._syncing:
            text = "正在处理图片，请稍候…"
        elif total == 0:
            text = "还没有图片，点【添加图片】把角色图加进来"
        elif ok == 0:
            text = f"{total} 张图片都打不开，请检查文件是否损坏"
        else:
            text = f"已就绪：共 {ok} 张图片"
            if fix:
                text += f" · {fix} 张待修正（开始后可点【修正这张】）"

        if self._syncing:
            dot = WARN
        elif total and ok == 0:
            dot = ERROR
        elif ok:
            dot = WARN if fix else ACCENT
        else:
            dot = FG_FAINT

        self.home_status.configure(text=text)
        self.status_dot.itemconfigure(self._dot, fill=dot)
        self._can_start = ok > 0 and not self._syncing
        self.start_btn.configure(state="normal" if self._can_start else "disabled")

    # ================= 键盘 =================

    def _bind_keys(self):
        self.bind_all("<space>", self._on_space, add="+")
        self.bind_all("<Return>", self._on_return, add="+")
        self.bind_all("<Right>", self._on_right, add="+")
        self.bind_all("<F11>", self._on_f11, add="+")
        self.bind_all("<Escape>", self._on_escape, add="+")

    def _on_space(self, _event=None):
        if not self._key_ok():
            return "break"
        self.handle_space()
        return "break"

    def _on_return(self, _event=None):
        if not self._key_ok():
            return "break"
        if self.view in (HOME, DONE):
            self.handle_space()
        return "break"

    def _on_right(self, _event=None):
        if not self._key_ok():
            return "break"
        if self.view == GAME:
            self.next_question()
        return "break"

    def _on_f11(self, _event=None):
        self._toggle_fullscreen()
        return "break"

    def _on_escape(self, _event=None):
        if self.view in (GAME, DONE):
            self.end_round()
        return "break"

    def _key_ok(self) -> bool:
        """焦点在弹窗或按钮上时不响应游戏快捷键。"""
        w = self.focus_get()
        if w is None:
            return True
        try:
            if w.winfo_toplevel() is not self:
                return False
        except tk.TclError:
            return True
        return not isinstance(w, (tk.Button, ttk.Button))


class _ProgressDialog(tk.Toplevel):
    """处理进度窗口（图片多时会显示）。"""

    def __init__(self, parent, total: int):
        super().__init__(parent)
        self.title("正在处理图片")
        self.configure(bg=SURFACE)
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # 处理完成前不可关闭

        tk.Label(self, text="正在生成剪影，请稍候…", bg=SURFACE, fg=FG,
                 font=(FONT, 13, "bold")).pack(padx=32, pady=(22, 10))
        self.bar = ttk.Progressbar(self, length=440, maximum=max(total, 1), value=0)
        self.bar.pack(padx=32)
        self.detail = tk.Label(self, text="", bg=SURFACE, fg=FG_DIM, font=(FONT, 10))
        self.detail.pack(padx=32, pady=(10, 2))
        self.count = tk.Label(self, text=f"0 / {total}", bg=SURFACE, fg=FG_FAINT, font=(FONT, 10))
        self.count.pack(pady=(0, 20))
        self.after(50, self._center)
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _center(self):
        try:
            self.update_idletasks()
            parent = self.master
            x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except tk.TclError:
            pass

    def set_progress(self, done: int, total: int, name: str):
        self.bar.configure(maximum=max(total, 1), value=done)
        self.count.configure(text=f"{done} / {total}")
        self.detail.configure(text=name)

    def close_now(self):
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class _SyncRunner:
    """后台跑 Store.sync：秒完就不弹窗，慢了才显示进度对话框。"""

    DIALOG_DELAY = 0.35

    def __init__(self, parent, store, force, on_done, total: int):
        self.parent = parent
        self.store = store
        self.force = force
        self.on_done = on_done
        self.total = total
        self.q: queue.Queue = queue.Queue()
        self.dialog: _ProgressDialog | None = None
        self.finished = False
        self.started = time.monotonic()
        self._last: tuple | None = None
        threading.Thread(target=self._worker, daemon=True).start()
        parent.after(80, self._poll)

    def _worker(self):
        try:
            summary = self.store.sync(
                force=self.force,
                progress=lambda i, t, n: self.q.put(("progress", i, t, n)),
            )
            self.q.put(("done", summary))
        except Exception as exc:  # noqa: BLE001 - 后台线程兜底
            self.q.put(("error", exc))

    def _poll(self):
        if self.finished or not self.parent.winfo_exists():
            return
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    self._last = (msg[1], msg[2], msg[3])
                    if self.dialog is not None:
                        self.dialog.set_progress(*self._last)
                elif msg[0] == "done":
                    self._finish(msg[1], None)
                    return
                else:
                    self._finish(None, msg[1])
                    return
        except queue.Empty:
            pass
        if self.dialog is None and time.monotonic() - self.started > self.DIALOG_DELAY:
            self.dialog = _ProgressDialog(self.parent, self.total)
            if self._last is not None:
                self.dialog.set_progress(*self._last)
        self.parent.after(80, self._poll)

    def _finish(self, summary, error):
        self.finished = True
        if self.dialog is not None:
            self.dialog.close_now()
            self.dialog = None
        self.on_done(summary, error)