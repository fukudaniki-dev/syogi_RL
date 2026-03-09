"""
policy (2187 次元 numpy 配列) を 9x9 ヒートマップに変換するモジュール。

dlshogi の policy レイアウト:
  policy_idx = direction * 81 + to_sq  (direction: 0-26, to_sq: 0-80)
  to_sq = file_idx * 9 + rank_idx

盤面座標への変換:
  先手: file_idx = 8 - col, rank_idx = row → col = 8 - file_idx, row = rank_idx
  後手: file_idx = col,     rank_idx = 8 - row → col = file_idx, row = 8 - rank_idx
"""
from __future__ import annotations

import numpy as np


def policy_to_heatmap(policy: np.ndarray, is_black: bool = True) -> np.ndarray:
    """
    policy を board[row][col] 座標系の 9x9 ヒートマップへ変換する。

    Parameters
    ----------
    policy   : shape (2187,) の確率配列（softmax 済みを想定）
    is_black : 先手視点なら True、後手視点なら False

    Returns
    -------
    heatmap : np.ndarray shape (9, 9)  board[row][col] 座標系
    """
    if policy.shape != (2187,):
        raise ValueError(f"policy の shape が不正です: {policy.shape} (期待値: (2187,))")

    policy = policy.astype(np.float64)
    if not np.isclose(policy.sum(), 1.0, atol=1e-3):
        policy = np.exp(policy - policy.max())
        policy /= policy.sum()

    # direction * 81 + to_sq レイアウトを正しく集約
    per_sq = policy.reshape(27, 81).sum(axis=0)  # shape (81,)

    # dlshogi 内部座標 → board[row][col] に変換
    board_heatmap = np.zeros((9, 9), dtype=np.float32)
    for to_sq in range(81):
        file_idx = to_sq // 9
        rank_idx = to_sq % 9
        if is_black:
            row = rank_idx
            col = 8 - file_idx
        else:
            row = 8 - rank_idx
            col = file_idx
        board_heatmap[row][col] = float(per_sq[to_sq])

    return board_heatmap
