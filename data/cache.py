"""
Swing Trade Signal System — SQLite キャッシュ管理

4つのテーブルを管理する:
  - price_history    : 日足OHLCVデータ
  - fundamentals     : ファンダメンタルデータ（JSON）
  - signals_log      : 生成されたシグナルの履歴
  - screening_cache  : スクリーニング結果キャッシュ（JSON）

全てのDB操作はこのモジュールを経由することで、
接続管理・スキーマ変更を一箇所に集約する。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

from config.settings import SQLITE_DB_PATH, FETCHER_CONFIG


class CacheDB:
    """SQLite キャッシュデータベースの管理クラス。

    使い方:
        db = CacheDB()            # デフォルトパスで初期化
        db = CacheDB(path)        # カスタムパスで初期化
        db.initialize()           # テーブル作成（初回のみ必要）

        # 株価の保存・取得
        db.upsert_price("NVDA", "2025-01-15", 130.5, 132.0, 129.0, 131.0, 50000000)
        rows = db.get_prices("NVDA", days=30)

        # ファンダメンタルの保存・取得
        db.upsert_fundamentals("NVDA", {"pe_ratio": 45.2, "roe": 0.85})
        data = db.get_fundamentals("NVDA")
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """CacheDBを初期化する。

        Args:
            db_path: SQLiteデータベースのパス。Noneの場合はデフォルトパスを使用。
        """
        self._db_path = Path(db_path) if db_path else SQLITE_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        """データベースファイルのパスを返す。"""
        return self._db_path

    # ================================================================
    # 接続管理
    # ================================================================

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """SQLite接続のコンテキストマネージャ。

        自動コミット・ロールバック・クローズを保証する。

        Yields:
            sqlite3.Connection: Row型を返す接続オブジェクト。
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ================================================================
    # テーブル初期化
    # ================================================================

    def initialize(self) -> None:
        """全テーブルを作成する（IF NOT EXISTS）。

        アプリケーション起動時に1回呼び出すこと。
        既存テーブルがある場合は何もしない。
        """
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ================================================================
    # price_history CRUD
    # ================================================================

    def upsert_price(
        self,
        ticker: str,
        date: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> None:
        """株価データを1行挿入（存在すれば更新）。

        Args:
            ticker: ティッカーシンボル。
            date:   日付文字列 (YYYY-MM-DD)。
            open_:  始値。
            high:   高値。
            low:    安値。
            close:  終値。
            volume: 出来高。
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO price_history (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open   = excluded.open,
                    high   = excluded.high,
                    low    = excluded.low,
                    close  = excluded.close,
                    volume = excluded.volume
                """,
                (ticker, date, open_, high, low, close, volume),
            )

    def upsert_prices_bulk(
        self,
        rows: list[tuple[str, str, float, float, float, float, int]],
    ) -> None:
        """株価データを一括挿入（存在すれば更新）。

        Args:
            rows: (ticker, date, open, high, low, close, volume) のタプルリスト。
        """
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO price_history (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open   = excluded.open,
                    high   = excluded.high,
                    low    = excluded.low,
                    close  = excluded.close,
                    volume = excluded.volume
                """,
                rows,
            )

    def get_prices(
        self,
        ticker: str,
        days: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """指定銘柄の株価データを取得する。

        Args:
            ticker:     ティッカーシンボル。
            days:       直近N日分を取得（start_date/end_dateより優先）。
            start_date: 取得開始日 (YYYY-MM-DD)。
            end_date:   取得終了日 (YYYY-MM-DD)。

        Returns:
            日付昇順の株価データリスト。各要素は
            {"ticker", "date", "open", "high", "low", "close", "volume"} の辞書。
        """
        with self._connect() as conn:
            if days is not None:
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                cursor = conn.execute(
                    "SELECT * FROM price_history WHERE ticker = ? AND date >= ? ORDER BY date ASC",
                    (ticker, cutoff),
                )
            elif start_date and end_date:
                cursor = conn.execute(
                    "SELECT * FROM price_history WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date ASC",
                    (ticker, start_date, end_date),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM price_history WHERE ticker = ? ORDER BY date ASC",
                    (ticker,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_price_date(self, ticker: str) -> str | None:
        """指定銘柄の最新株価日付を返す。

        Args:
            ticker: ティッカーシンボル。

        Returns:
            最新日付の文字列 (YYYY-MM-DD)、データなしの場合は None。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) as max_date FROM price_history WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            return row["max_date"] if row and row["max_date"] else None

    # ================================================================
    # fundamentals CRUD
    # ================================================================

    def upsert_fundamentals(self, ticker: str, data: dict[str, Any]) -> None:
        """ファンダメンタルデータを保存する（JSON形式）。

        Args:
            ticker: ティッカーシンボル。
            data:   ファンダメンタル指標の辞書。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fundamentals (ticker, date_fetched, data_json)
                VALUES (?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    date_fetched = excluded.date_fetched,
                    data_json    = excluded.data_json
                """,
                (ticker, now, json.dumps(data, ensure_ascii=False)),
            )

    def get_fundamentals(self, ticker: str) -> dict[str, Any] | None:
        """ファンダメンタルデータを取得する。

        キャッシュ有効期間（デフォルト7日）を超えている場合は None を返す。

        Args:
            ticker: ティッカーシンボル。

        Returns:
            ファンダメンタル指標の辞書。キャッシュ切れまたはデータなしの場合は None。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT date_fetched, data_json FROM fundamentals WHERE ticker = ?",
                (ticker,),
            ).fetchone()

        if row is None:
            return None

        # キャッシュ有効期限チェック
        fetched_at = datetime.strptime(row["date_fetched"], "%Y-%m-%d %H:%M:%S")
        expiry = timedelta(days=FETCHER_CONFIG.cache_fundamentals_days)
        if datetime.now() - fetched_at > expiry:
            return None

        return json.loads(row["data_json"])

    # ================================================================
    # signals_log CRUD
    # ================================================================

    def log_signal(
        self,
        ticker: str,
        signal: str,
        score: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reason: str = "",
    ) -> None:
        """シグナルをログに記録する。

        Args:
            ticker:      ティッカーシンボル。
            signal:      シグナル種別 (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL)。
            score:       統合スコア (-1.0 〜 +1.0)。
            stop_loss:   損切り価格。
            take_profit: 利確価格。
            reason:      シグナル判定理由の概要。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals_log
                    (timestamp, ticker, signal, score, stop_loss, take_profit, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now, ticker, signal, score, stop_loss, take_profit, reason),
            )

    def get_signal_history(
        self,
        ticker: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """シグナル履歴を取得する。

        Args:
            ticker: ティッカーシンボル。Noneの場合は全銘柄。
            days:   直近N日分を取得。

        Returns:
            新しい順のシグナルログリスト。
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            if ticker:
                cursor = conn.execute(
                    """
                    SELECT * FROM signals_log
                    WHERE ticker = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (ticker, cutoff),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM signals_log
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    (cutoff,),
                )
            return [dict(row) for row in cursor.fetchall()]

    # ================================================================
    # screening_cache CRUD
    # ================================================================

    def upsert_screening_cache(self, key: str, data: Any) -> None:
        """スクリーニング結果をキャッシュに保存する（JSON形式）。

        Args:
            key:  キャッシュキー（例: "iwv_holdings", "tier1_pass"）。
            data: 保存するデータ（JSON シリアライズ可能な任意の値）。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO screening_cache (key, data_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    data_json  = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(data, ensure_ascii=False), now),
            )

    def get_screening_cache(self, key: str, max_age_days: int = 7) -> Any | None:
        """スクリーニングキャッシュを取得する。

        Args:
            key:          キャッシュキー。
            max_age_days: キャッシュ有効期限（日数）。

        Returns:
            キャッシュされたデータ。期限切れまたは存在しない場合は None。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data_json, updated_at FROM screening_cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        updated_at = datetime.strptime(row["updated_at"], "%Y-%m-%d %H:%M:%S")
        if datetime.now() - updated_at > timedelta(days=max_age_days):
            return None

        return json.loads(row["data_json"])

    # ================================================================
    # ユーティリティ
    # ================================================================

    def get_table_counts(self) -> dict[str, int]:
        """各テーブルのレコード数を返す（デバッグ・確認用）。

        Returns:
            テーブル名→レコード数の辞書。
        """
        counts: dict[str, int] = {}
        with self._connect() as conn:
            for table in ("price_history", "fundamentals", "signals_log", "screening_cache"):
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                counts[table] = row["cnt"] if row else 0
        return counts

    def clear_table(self, table_name: str) -> int:
        """指定テーブルの全レコードを削除する。

        Args:
            table_name: テーブル名（price_history / fundamentals / signals_log）。

        Returns:
            削除されたレコード数。

        Raises:
            ValueError: 不正なテーブル名が指定された場合。
        """
        allowed = {"price_history", "fundamentals", "signals_log", "screening_cache"}
        if table_name not in allowed:
            raise ValueError(f"不正なテーブル名: {table_name}（許可: {allowed}）")

        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM {table_name}")
            return cursor.rowcount


# ============================================================
# スキーマ定義 (DDL)
# ============================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS price_history (
    ticker  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_price_ticker
    ON price_history (ticker);

CREATE INDEX IF NOT EXISTS idx_price_date
    ON price_history (date);


CREATE TABLE IF NOT EXISTS fundamentals (
    ticker       TEXT NOT NULL PRIMARY KEY,
    date_fetched TEXT NOT NULL,
    data_json    TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS signals_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    score       REAL    NOT NULL,
    stop_loss   REAL,
    take_profit REAL,
    reason      TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker
    ON signals_log (ticker);

CREATE INDEX IF NOT EXISTS idx_signals_timestamp
    ON signals_log (timestamp);


CREATE TABLE IF NOT EXISTS screening_cache (
    key        TEXT NOT NULL PRIMARY KEY,
    data_json  TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
