"""
policy (2187 次元 numpy 配列) を 9x9 ヒートマップに変換するモジュール。

変換手順:
  1. softmax (inference.py 側で実施済みの場合はそのまま)
  2. reshape(81, 27)
  3. sum(axis=1)  → shape (81,)
  4. reshape(9, 9)
"""
from __future__ import annotations

import numpy as np


def policy_to_heatmap(policy: np.ndarray) -> np.ndarray:
    """
    policy を 9x9 ヒートマップへ変換する。

    Parameters
    ----------
    policy : np.ndarray
        shape (2187,) の確率配列（softmax 済みを想定）。
        未正規化の場合は softmax を適用してから変換する。

    Returns
    -------
    heatmap : np.ndarray
        shape (9, 9) の確率合計マップ。各セルの値は 0〜1。
    """
    if policy.shape != (2187,):
        raise ValueError(f"policy の shape が不正です: {policy.shape} (期待値: (2187,))")

    # 未正規化の場合のために softmax を適用（すでに適用済みでも問題ない）
    policy = policy.astype(np.float64)
    if not np.isclose(policy.sum(), 1.0, atol=1e-3):
        policy = np.exp(policy - policy.max())
        policy /= policy.sum()

    # reshape → sum → reshape
    heatmap = policy.reshape(81, 27).sum(axis=1).reshape(9, 9)
    return heatmap.astype(np.float32)
