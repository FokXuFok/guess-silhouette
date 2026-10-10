"""打包脚本（推荐入口，避免 .bat 中文编码问题）。

用法：
    python build.py         单文件模式：dist\\猜剪影.exe（首次启动稍慢）
    python build.py dir     文件夹模式：dist\\猜剪影\\（启动更快，推荐现场用）

AI 抠图模型：models/isnetis.onnx 存在时自动复制到输出目录（models/ 子目录）；
缺失时生成不含 AI 的版本（传统四档抠图仍可用）。
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_SRC = ROOT / "models" / "isnetis.onnx"


def _copy_model(dest_dir: Path) -> None:
    """把 AI 模型复制到打包输出目录（放 exe 旁边,不进单文件 exe）。"""
    if not MODEL_SRC.is_file():
        print("[提示] 未找到 models/isnetis.onnx，将生成不含 AI 抠图的版本（传统四档仍可用）。")
        return
    models_dir = dest_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "isnetis.onnx"
    if target.is_file() and target.stat().st_size == MODEL_SRC.stat().st_size:
        print(f"[模型] 已存在，跳过复制：{target}")
        return
    shutil.copy2(MODEL_SRC, target)
    print(f"[模型] 已复制到：{target}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    folder_mode = any(a.lower() in ("dir", "folder") for a in argv)

    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except ImportError:
        print("[错误] 未找到 pyinstaller，请先执行： pip install -r requirements.txt")
        return 1

    args = ["--clean", "--noconfirm", "--distpath", "dist", "--workpath", "build",
            "-w", "-n", "猜剪影", "guess.py"]
    if folder_mode:
        print("[打包] 文件夹模式：dist\\猜剪影\\（包含 AI 模型，启动快）")
    else:
        args.insert(0, "-F")
        print("[打包] 单文件模式：dist\\猜剪影.exe（首次启动稍慢）")

    pyinstaller_run(args)

    dist_root = ROOT / "dist"
    _copy_model(dist_root / "猜剪影" if folder_mode else dist_root)

    print()
    print("[完成] 输出在 dist 文件夹。")
    print(r"把 dist 里的 猜剪影（整个文件夹/含 models 子目录）复制到活动电脑即可；")
    print(r"首次运行会在它旁边自动创建 图片\ 缓存\ config.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
