"""
Swing Trade Signal System — 分析モジュール共通ユーティリティ
"""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_get(data: dict[str, Any], key: str) -> float | None:
    """辞書から値を安全に取得する。None/NaN/Inf は None として返す。

    Args:
        data: データ辞書。
        key:  取得するキー。

    Returns:
        float 値。無効値の場合は None。
    """
    val = data.get(key)
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f
