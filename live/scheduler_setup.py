"""
Live Trading — Windows Task Scheduler 自動登録

毎週 火〜土曜 07:00 JST に live/daily.py を実行するタスクを登録する。

タイミング説明:
  米国株式市場 16:00 ET 引け
  ├─ 冬時間 (EST = UTC-5): 翌日 06:00 JST → 07:00 で余裕あり
  └─ 夏時間 (EDT = UTC-4): 翌日 05:00 JST → 07:00 で余裕あり
  米国月〜金取引 = 日本の火〜土朝

使い方（管理者権限の PowerShell / コマンドプロンプトで実行）:
  python live/scheduler_setup.py           タスク登録
  python live/scheduler_setup.py --remove  タスク削除
  python live/scheduler_setup.py --status  登録状態確認
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

TASK_NAME = "SwingSignal_DailyRun"
RUN_TIME = "07:00"
# 火〜土曜（MON は米国の月曜取引がないため不要）
DAYS_OF_WEEK = "TUE,WED,THU,FRI,SAT"


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows Task Scheduler 登録ツール")
    parser.add_argument("--remove", action="store_true", help="タスクを削除")
    parser.add_argument("--status", action="store_true", help="タスクの登録状態を確認")
    args = parser.parse_args()

    if args.status:
        _check_status()
    elif args.remove:
        _remove_task()
    else:
        _register_task()


def _register_task() -> None:
    """タスクを Windows Task Scheduler に登録する。"""
    # Python 実行ファイルのパス
    python_exe = sys.executable
    # daily.py のフルパス
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "live", "daily.py",
    )
    # 作業ディレクトリ（プロジェクトルート）
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Task Name:   {TASK_NAME}")
    print(f"Python:      {python_exe}")
    print(f"Script:      {script_path}")
    print(f"Work Dir:    {work_dir}")
    print(f"Schedule:    毎週 {DAYS_OF_WEEK} {RUN_TIME} JST")
    print()

    # まず既存タスクを削除（エラーは無視）
    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
    )

    # 新規タスク登録
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", f'"{python_exe}" "{script_path}"',
        "/SC", "WEEKLY",
        "/D", DAYS_OF_WEEK,
        "/ST", RUN_TIME,
        "/SD", "01/01/2026",
        "/RL", "HIGHEST",           # 最高権限で実行
        "/IT",                      # インタラクティブモード（ログインユーザーで実行）
        "/F",                       # 強制上書き
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ タスク登録完了")
        print(f"\n登録タスク名: {TASK_NAME}")
        print(f"次回実行:    {RUN_TIME} JST（火〜土曜）")
        print()
        print("確認コマンド:")
        print(f"  schtasks /Query /TN {TASK_NAME} /FO LIST")
        print()
        print("手動実行テスト:")
        print(f"  schtasks /Run /TN {TASK_NAME}")
    else:
        print("❌ タスク登録失敗")
        print(f"エラー: {result.stderr}")
        print()
        print("管理者権限で実行してください:")
        print("  1. PowerShell を右クリック → 「管理者として実行」")
        print("  2. cd <プロジェクトフォルダ>")
        print("  3. python live/scheduler_setup.py")
        sys.exit(1)


def _remove_task() -> None:
    """タスクを削除する。"""
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"✅ タスク '{TASK_NAME}' を削除しました")
    else:
        print(f"❌ タスク削除失敗（登録されていない可能性があります）")
        print(result.stderr)


def _check_status() -> None:
    """タスクの登録状態を確認する。"""
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"✅ タスク '{TASK_NAME}' は登録済みです")
        print()
        print(result.stdout)
    else:
        print(f"⚠  タスク '{TASK_NAME}' は未登録です")
        print("登録するには: python live/scheduler_setup.py")


if __name__ == "__main__":
    main()
