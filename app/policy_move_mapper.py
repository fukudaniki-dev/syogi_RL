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

C++ の get_move_direction は不等号で判定するため、スライド移動（複数マス）も
単純な dir < 0 などの条件で正しい方向に対応できる。
本実装も同様に不等号ベースの判定を行う。

C++ 座標系との対応:
  dir_y = to_rank - from_rank = tr - fr = dr
  dir_x = from_file - to_file = (8-fc) - (8-tc) = tc - fc = dc

後手の場合、C++ 内部で盤面を180度回転させてから方向を計算するため、
後手の (dr, dc) を先手視点に変換するには符号を反転させる。
"""

from typing import Optional


def _direction_from_black_view(dr: int, dc: int) -> Optional[int]:
    """
    先手視点の (dr, dc) を direction インデックス (0-9) に変換する。
    C++ get_move_direction の不等号ベースロジックに対応。
    """
    # 桂馬（knight）: 先に確認（UP_LEFT/RIGHT 条件に包含されるため）
    if dr == -2 and dc == -1:
        return 8   # UP2_LEFT
    if dr == -2 and dc == 1:
        return 9   # UP2_RIGHT
    # 縦移動
    if dr < 0 and dc == 0:
        return 0   # UP
    if dr > 0 and dc == 0:
        return 5   # DOWN
    # 横移動
    if dr == 0 and dc < 0:
        return 3   # LEFT   (dc<0: toward 9筋)
    if dr == 0 and dc > 0:
        return 4   # RIGHT  (dc>0: toward 1筋)
    # 斜め移動
    if dr < 0 and dc < 0:
        return 1   # UP_LEFT
    if dr < 0 and dc > 0:
        return 2   # UP_RIGHT
    if dr > 0 and dc < 0:
        return 6   # DOWN_LEFT
    if dr > 0 and dc > 0:
        return 7   # DOWN_RIGHT
    return None


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

    # to_sq: dlshogi 内部座標 (current player 視点)
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

        # 後手は180度回転した視点で方向を計算する
        if not is_black:
            dr, dc = -dr, -dc

        base_dir = _direction_from_black_view(dr, dc)
        if base_dir is None:
            return None
        direction = base_dir + (10 if move.get("promote") else 0)

    return direction * 81 + to_sq
