"""スクリーナーモジュールのユニットテスト。"""

import pytest

from data.screener import _parse_ishares_csv, _VALID_TICKER_RE, _TICKER_FIXUP


# ============================================================
# iShares CSV パーステスト
# ============================================================

_SAMPLE_CSV = """\
\ufeffiShares Russell 3000 ETF
Fund Holdings as of,"Feb 18, 2026"
Inception Date,"May 22, 2000"
Shares Outstanding,"47,400,000.00"
Stock,"-"
Bond,"-"
Cash,"-"
Other,"-"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,Exchange,Currency,FX Rate,Market Currency,Accrual Date
"NVDA","NVIDIA CORP","Information Technology","Equity","1,229,991,299.94","6.64","1,229,991,299.94","6,543,203.00","187.98","United States","NASDAQ","USD","1.00","USD",""
"AAPL","APPLE INC","Information Technology","Equity","1,077,450,683.15","5.81","1,077,450,683.15","4,075,849.00","264.35","United States","NASDAQ","USD","1.00","USD",""
"BRKB","BERKSHIRE HATHAWAY INC CLASS B","Financials","Equity","244,650,000.00","1.32","244,650,000.00","500,000.00","489.30","United States","NYSE","USD","1.00","USD",""
"XXYYZZ","INVALID TICKER","","Cash","100,000.00","0.01","100,000.00","100.00","1000.00","United States","","USD","1.00","USD",""
"P5N994","SOME WEIRD CODE","","Equity","50,000.00","0.00","50,000.00","50.00","1000.00","United States","","USD","1.00","USD",""
"MSFT","MICROSOFT CORP","Information Technology","Equity","828,548,622.00","4.47","828,548,622.00","2,073,445.00","399.60","United States","NASDAQ","USD","1.00","USD",""
"""


class TestParseISharesCSV:
    """iShares CSV パースのテスト。"""

    def test_extracts_equity_tickers(self):
        tickers = _parse_ishares_csv(_SAMPLE_CSV)
        assert "NVDA" in tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_excludes_non_equity(self):
        """Cash 行は除外される。"""
        tickers = _parse_ishares_csv(_SAMPLE_CSV)
        assert "XXYYZZ" not in tickers

    def test_excludes_invalid_tickers(self):
        """英字1-5文字に合致しないティッカーは除外される。"""
        tickers = _parse_ishares_csv(_SAMPLE_CSV)
        assert "P5N994" not in tickers

    def test_fixup_brkb(self):
        """BRKB → BRK-B に変換される。"""
        tickers = _parse_ishares_csv(_SAMPLE_CSV)
        assert "BRK-B" in tickers
        assert "BRKB" not in tickers

    def test_empty_csv(self):
        result = _parse_ishares_csv("")
        assert result == []

    def test_no_header(self):
        result = _parse_ishares_csv("no header here\njust random data")
        assert result == []

    def test_total_count(self):
        """サンプルCSVから4銘柄（Equity）が抽出され、うち有効は4つ。"""
        tickers = _parse_ishares_csv(_SAMPLE_CSV)
        # NVDA, AAPL, BRK-B, MSFT (P5N994 は除外, XXYYZZ は非Equity)
        assert len(tickers) == 4


# ============================================================
# ティッカーバリデーションテスト
# ============================================================

class TestTickerValidation:
    """ティッカーの正規表現バリデーションテスト。"""

    @pytest.mark.parametrize("ticker", ["NVDA", "A", "MSFT", "AVGO", "BRK-B"])
    def test_valid_tickers(self, ticker: str):
        assert _VALID_TICKER_RE.match(ticker) is not None

    @pytest.mark.parametrize("ticker", ["P5N994", "123", "", "TOOLONGSYM", "nvda"])
    def test_invalid_tickers(self, ticker: str):
        assert _VALID_TICKER_RE.match(ticker) is None

    def test_fixup_map(self):
        assert _TICKER_FIXUP["BRKB"] == "BRK-B"


# ============================================================
# config/screening.py テスト
# ============================================================

class TestScreeningParams:
    """スクリーニングパラメータのテスト。"""

    def test_defaults(self):
        from config.screening import SCREENING_PARAMS
        assert SCREENING_PARAMS.min_market_cap == 3e8
        assert SCREENING_PARAMS.min_avg_volume == 500_000
        assert SCREENING_PARAMS.min_price == 5.0
        assert SCREENING_PARAMS.min_momentum_score == 0.0
        assert SCREENING_PARAMS.max_screen_candidates == 100

    def test_frozen(self):
        from config.screening import SCREENING_PARAMS
        with pytest.raises(AttributeError):
            SCREENING_PARAMS.min_market_cap = 1e9  # type: ignore


# ============================================================
# CacheDB screening_cache テスト
# ============================================================

class TestScreeningCache:
    """screening_cache テーブルのテスト。"""

    def test_upsert_and_get(self, tmp_path):
        from data.cache import CacheDB
        db = CacheDB(tmp_path / "test.db")
        db.initialize()

        tickers = ["NVDA", "AAPL", "MSFT"]
        db.upsert_screening_cache("test_key", tickers)

        result = db.get_screening_cache("test_key", max_age_days=1)
        assert result == tickers

    def test_cache_miss(self, tmp_path):
        from data.cache import CacheDB
        db = CacheDB(tmp_path / "test.db")
        db.initialize()

        result = db.get_screening_cache("nonexistent", max_age_days=1)
        assert result is None

    def test_cache_overwrite(self, tmp_path):
        from data.cache import CacheDB
        db = CacheDB(tmp_path / "test.db")
        db.initialize()

        db.upsert_screening_cache("key1", ["A", "B"])
        db.upsert_screening_cache("key1", ["C", "D", "E"])

        result = db.get_screening_cache("key1", max_age_days=1)
        assert result == ["C", "D", "E"]
