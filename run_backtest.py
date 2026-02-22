"""バックテスト実行スクリプト（UTF-8エンコーディング設定済み）。"""
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

# stdoutをUTF-8に強制
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# main.pyのmain()を呼び出す
sys.argv = ["main.py", "--backtest", "--no-notify"]

from main import main
sys.exit(main())
