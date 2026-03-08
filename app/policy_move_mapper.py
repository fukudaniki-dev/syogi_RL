"""
合法手 → dlshogi policy インデックス のマッピング。

dlshogi の policy 出力形状: (2187,) = 27 directions × 81 squares
  policy_idx = direction * 81 + to_sq

座標変換 (board[row][col] → dlshogi 内部座標):
  先手: to_sq = (8-col)*9 + row
  後手: to_sq = col*9 + (8-row)

方向 (direction):
  0-9: 非成り移動 (UP=0, UP_LEFT=1, UP_RIGHT=2, LEFT=3, RIGHT=4,
                   DOWN=5, DOWN_LEFT=6, DOWN_RIGHT=7, UP2_LEFT=8, UP2_RIGHT=9)
  10-19: 成り移動 (上記+10)
  20-26: 打ち (P=20, L=21, N=22, S=23, B=24, R=25, G=26)

方向は手番プレイヤー視点:
  先手: UP = (dr=-1, dc=0), UP_LEFT = (dr=-1, dc=-1), ...
  後手: UP = (dr=+1, dc=0), UP_LEFT = (dr=+1, dc=+1), ... (180度回転)
"""

from typing import Optional

# 先手の (dr, dc) → direction インデックス (0-9)
_BLACK_DIRS = {
    (-1,  0): 0,   # UP
    (-1, -1): 1,   # UP_LEFT
    (-1, +1): 2,   # UP_RIGHT
    ( 0, -1): 3,   # LEFT
    ( 0, +1): 4,   # RIGHT
    (+1,  0): 5,   # DOWN
    (+1, -1): 6,   # DOWN_LEFT
    (+1, +1): 7,   # DOWN_RIGHT
    (-2, -1): 8,   # UP2_LEFT (桂馬)
    (-2, +1): 9,   # UP2_RIGHT (桂馬)
}

# 後手の (dr, dc) → direction インデックス (0-9)  ※180度回転
_WHITE_DIRS = {
    (+1,  0): 0,   # UP
    (+1, +1): 1,   # UP_LEFT
    (+1, -1): 2,   # UP_RIGHT
    ( 0, +1): 3,   # LEFT
    ( 0, -1): 4,   # RIGHT
    (-1,  0): 5,   # DOWN
    (-1, +1): 6,   # DOWN_LEFT
    (-1, -1): 7,   # DOWN_RIGHT
    (+2, +1): 8,   # UP2_LEFT (桂馬)
    (+2, -1): 9,   # UP2_RIGHT (桂馬)
}

# 打ち駒の種類 → direction オフセット (20 + idx)
_DROP_IDX = {
    "P": 20, "p": 20,
    "L": 21, "l": 21,
    "N": 22, "n": 22,
    "S": 23, "s": 23,
    "B": 24, "b": 24,
    "R": 25, "r": 25,
    "G": 26, "g": 26,
}


def move_to_policy_idx(move: dict, is_black: bool) -> Optional[int]:
    """
    合法手辞書を dlshogi policy インデックス (0-2186) に変換する。

    Parameters
    ----------
    move : {"from": [fr, fc] or None, "to": [tr, tc], "promote": bool, "piece": str}
    is_black : 先手の手番かどうか

    Returns
    -------
    int or None: policy インデックス。対応する方向が存在しない場合は None
    """
    tr, tc = move["to"]

    # to_sq の計算
    if is_black:
        to_sq = (8 - tc) * 9 + tr
    else:
        to_sq = tc * 9 + (8 - tr)

    if move["from"] is None:
        # 持ち駒打ち
        piece = move["piece"]
        direction = _DROP_IDX.get(piece)
        if direction is None:
            return None
    else:
        fr, fc = move["from"]
        dr = tr - fr
        dc = tc - fc
        dir_table = _BLACK_DIRS if is_black else _WHITE_DIRS
        base_dir = dir_table.get((dr, dc))
        if base_dir is None:
            return None
        direction = base_dir + (10 if move.get("promote") else 0)

    return direction * 81 + to_sq
