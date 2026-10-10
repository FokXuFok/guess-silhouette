"""猜剪影入口。

用法：
- 直接运行（PyCharm / 命令行）：打开界面
      python guess.py
- 命令行批处理模式（不开界面，用于活动前检查抠图质量）：
      python guess.py --cli [--folder 路径] [--force]
- 打包后的 猜剪影.exe：双击打开界面
"""
import argparse
import logging
import sys
from pathlib import Path


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.stderr is not None:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)
    try:
        from app.paths import base_dir
        log_dir = base_dir() / "缓存"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "运行日志.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:
        pass


def _enable_dpi_awareness() -> None:
    """Windows 高分屏下让界面更清晰。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def _print_ai_status(store) -> None:
    """CLI 报告 AI 抠图状态，方便活动前确认。"""
    from app import segmentation

    seg = getattr(store, "segmenter", None)
    if seg is None:
        if not getattr(store.config, "ai_matting", True):
            print("AI 抠图：已在配置中关闭（config.json 的 ai_matting=false）")
        elif not segmentation.default_model_path().is_file():
            print("AI 抠图：未启用（未找到 models/isnetis.onnx，使用传统抠图）")
        else:
            print("AI 抠图：未启用（使用传统抠图）")
    elif seg.is_ready:
        print("AI 抠图：已启用（isnet 本地模型）")
    else:
        print("AI 抠图：模型加载失败，本次使用传统抠图（详见 缓存/运行日志.log）")


def run_cli(image_dir: Path, config, force: bool) -> int:
    from app.store import Store

    store = Store(image_dir, config)
    files = store.scan_images()
    if not files:
        print(f"图片文件夹为空：{image_dir}")
        print("请把角色图（文件名就是角色名）放进去后重试。")
        return 1

    print(f"图片文件夹：{image_dir}")
    _print_ai_status(store)
    print(f"共 {len(files)} 张图片，开始处理…")
    summary = store.sync(force=force, progress=lambda i, t, n: print(f"  [{i}/{t}] {n}"))

    tier_names = {
        0: "AI 智能抠图",
        1: "透明背景",
        2: "纯色背景",
        3: "渐变背景",
        4: "复杂背景",
    }
    print(f"\n完成：新处理 {summary.processed}，跳过 {summary.skipped}，失败 {summary.failed}")
    for e in summary.entries:
        if not e.ok:
            print(f"  [打不开] {e.name}：{e.error}")
        elif e.needs_fix and not e.verified:
            print(f"  [建议修正] {e.name}（{tier_names.get(e.tier, '?')}）：{'；'.join(e.reasons)}")
        else:
            print(f"  [OK] {e.name}（{tier_names.get(e.tier, '?')}）")
    if summary.needs_fix_count:
        print(f"\n共有 {summary.needs_fix_count} 张建议修正，可在界面里用【修正这张】处理。")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="猜剪影", description="动漫社活动现场投影互动：猜剪影")
    parser.add_argument("--cli", action="store_true", help="命令行批量处理模式（不打开界面）")
    parser.add_argument("--folder", type=str, default=None, help="图片文件夹（默认程序目录下的 图片）")
    parser.add_argument("--force", action="store_true", help="强制重新生成所有剪影")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    from app.config import Config
    from app.paths import default_image_dir

    image_dir = Path(args.folder) if args.folder else default_image_dir()
    config = Config.load()
    if not Config.default_path().exists():
        config.save()  # 首次运行生成配置文件，方便现场微调风格
    _setup_logging()

    if args.cli:
        return run_cli(image_dir, config, force=bool(args.force))

    _enable_dpi_awareness()
    from app.ui import App
    App(image_dir, config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())