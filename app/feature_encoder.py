"""
dlshogi 形式の特徴量エンコーダー

C++ cppshogi の make_input_features に対応する Python 実装。
利き計算（FEATURES1の14-30ch）は省略し、駒位置と持ち駒のみエンコードする。

座標系:
  sq = file_idx * 9 + rank_idx
  file_idx = 8 - col   (col=0 が9筋→file_idx=8、col=8 が1筋→file_idx=0)
  rank_idx = row        (row=0 が9段目=盤面上部→rank_idx=0)

テンソル形状: (batch, channels, 9, 9)
  dim2 = file_idx = 8 - col
  dim3 = rank_idx = row
"""

import numpy as np
from typing import Dict, List

# 駒文字 → PieceType インデックス (0始まり, cppshogi の PieceType-1)
PIECE_TO_IDX: Dict[str, int] = {
    "P": 0,  "p": 0,   # 歩
    "L": 1,  "l": 1,   # 香
    "N": 2,  "n": 2,   # 桂
    "S": 3,  "s": 3,   # 銀
    "B": 4,  "b": 4,   # 角
    "R": 5,  "r": 5,   # 飛
    "G": 6,  "g": 6,   # 金
    "K": 7,  "k": 7,   # 王
    "+P": 8, "+p": 8,  # と
    "+L": 9, "+l": 9,  # 杏
    "+N": 10,"+n": 10, # 圭
    "+S": 11,"+s": 11, # 全
    "+B": 12,"+b": 12, # 馬
    "+R": 13,"+r": 13, # 龍
}

# 持ち駒の種類と最大枚数 (先手表記, 先手の順)
HAND_TYPES = [("P", 8), ("L", 4), ("N", 4), ("S", 4), ("G", 4), ("B", 2), ("R", 2)]
MAX_PIECES_IN_HAND_SUM = sum(n for _, n in HAND_TYPES)  # 28

PIECETYPE_NUM = 14
MAX_ATTACK_NUM = 3
FEATURES1_GROUP = PIECETYPE_NUM + PIECETYPE_NUM + MAX_ATTACK_NUM  # 31


def _is_black_piece(piece: str) -> bool:
    if piece.startswith("+"):
        return piece[1].isupper()
    return piece.isupper()


def encode_features(
    board_2d: List[List[str]],
    hands: Dict[str, Dict[str, int]],
    turn: str,
    in_check: bool,
):
    """
    盤面状態を dlshogi の入力特徴量にエンコードする。

    Parameters
    ----------
    board_2d : 9x9 盤面リスト (board[row][col])
    hands    : {"black": {piece: count}, "white": {piece: count}}
    turn     : "black" or "white"
    in_check : 現在手番の玉が王手されているか

    Returns
    -------
    f1 : np.ndarray  shape (1, FEATURES1_NUM, 9, 9)   dtype float32
    f2 : np.ndarray  shape (1, FEATURES2_NUM, 9, 9)   dtype float32
    """
    try:
        from dlshogi import common
        f1_num = common.FEATURES1_NUM
        f2_num = common.FEATURES2_NUM
    except ImportError:
        f1_num, f2_num = 62, 57

    is_black = (turn == "black")

    f1 = np.zeros((1, f1_num, 9, 9), dtype=np.float32)
    f2 = np.zeros((1, f2_num, 9, 9), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # FEATURES1: 駒の位置 (利き情報は省略)
    # ------------------------------------------------------------------ #
    for row in range(9):
        for col in range(9):
            piece = board_2d[row][col]
            if not piece:
                continue
            piece_idx = PIECE_TO_IDX.get(piece, -1)
            if piece_idx < 0:
                continue

            piece_is_black = _is_black_piece(piece)
            is_own = (piece_is_black == is_black)

            if is_black:
                # 先手視点: file_idx = 8-col, rank_idx = row
                fi = 8 - col
                ri = row
            else:
                # 後手視点: 盤面180度回転
                fi = col
                ri = 8 - row

            # 自駒: ch 0-13、相手駒: ch 31-44
            ch_offset = 0 if is_own else FEATURES1_GROUP
            f1[0, ch_offset + piece_idx, fi, ri] = 1.0

    # ------------------------------------------------------------------ #
    # FEATURES2: 持ち駒
    # ------------------------------------------------------------------ #
    cur_key = "black" if is_black else "white"
    opp_key = "white" if is_black else "black"
    cur_order = [p for p, _ in HAND_TYPES]        # 大文字 (先手表記)
    opp_order = [p.lower() for p, _ in HAND_TYPES] # 小文字 (後手表記)
    if not is_black:
        cur_order, opp_order = opp_order, cur_order

    ch = 0
    # 自分の持ち駒 (c2=0 : channels 0-27)
    for (_, max_count), piece in zip(HAND_TYPES, cur_order):
        count = min(hands[cur_key].get(piece, 0), max_count)
        for i in range(count):
            f2[0, ch + i, :, :] = 1.0
        ch += max_count

    # 相手の持ち駒 (c2=1 : channels 28-55)
    for (_, max_count), piece in zip(HAND_TYPES, opp_order):
        count = min(hands[opp_key].get(piece, 0), max_count)
        for i in range(count):
            f2[0, ch + i, :, :] = 1.0
        ch += max_count

    # 王手フラグ (channel 56)
    if in_check:
        f2[0, ch, :, :] = 1.0

    return f1, f2
