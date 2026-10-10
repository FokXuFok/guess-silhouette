"""打包脚本（推荐入口，避免 .bat 中文编码问题）。

用法：
    python build.py         单文件模式：dist\\猜剪影.exe（首次启动稍慢）
    python build.py dir     文件夹模式：dist\\猜剪影\\（启动更快，推荐现场用）
"""
import sys


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
        print("[打包] 文件夹模式：dist\\猜剪影\\")
    else:
        args.insert(0, "-F")
        print("[打包] 单文件模式：dist\\猜剪影.exe")

    pyinstaller_run(args)
    print()
    print("[完成] 输出在 dist 文件夹。")
    print(r"把 dist 里的 猜剪影 复制到活动电脑即可；首次运行会在它旁边自动创建 图片\ 缓存\ config.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())