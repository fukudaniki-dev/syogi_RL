"""
dlshogi 形式の特徴量エンコーダー

C++ cppshogi の make_input_features に対応する Python 実装。

座標系:
  sq = file_idx * 9 + rank_idx
  file_idx = 8 - col   (col=0 が9筋→file_idx=8、col=8 が1筋→file_idx=0)
  rank_idx = row        (row=0 が1段目=盤面上部→rank_idx=0)

テンソル形状: (batch, channels, 9, 9)
  dim2 = file_idx = 8 - col
  dim3 = rank_idx = row

FEATURES1 チャンネル構造 (62ch = 2 × 31ch):
  先手視点 (ch 0-30):
    0-13  : 自駒の位置 (駒種別)
    14-27 : 自駒の利き (駒種別の攻撃可能マス)
    28-30 : 自駒の利き数カウント (1以上/2以上/3以上)
  後手視点 (ch 31-61):
    31-44 : 相手駒の位置
    45-58 : 相手駒の利き
    59-61 : 相手駒の利き数カウント
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


def _set_attack_planes(
    f1: np.ndarray,
    board_2d: List[List[str]],
    is_black: bool,
) -> None:
    """
    FEATURES1 の利きチャンネル (ch 14-27, 28-30, 45-58, 59-61) を埋める。

    cppshogi make_input_features の攻撃チャンネル仕様:
      ch = group_offset + PIECETYPE_NUM + piece_idx : 駒種別の利き先マス
      ch = group_offset + PIECETYPE_NUM*2 + k       : 利き数カウント (k=0,1,2 で 1以上/2以上/3以上)
    """
    from app.shogi_engine import _piece_attacks

    # [color_slot][fi][ri] で攻撃数を集計 (color_slot: 0=自駒, 1=相手駒)
    attack_count = [
        [[0] * 9 for _ in range(9)],
        [[0] * 9 for _ in range(9)],
    ]

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
            color_slot = 0 if is_own else 1
            ch_base = FEATURES1_GROUP * color_slot  # 0 (自駒) or 31 (相手駒)

            for (tr, tc) in _piece_attacks(piece, row, col, board_2d):
                # 攻撃先マスを現在の手番視点の座標に変換
                if is_black:
                    fi, ri = 8 - tc, tr
                else:
                    fi, ri = tc, 8 - tr

                # 駒種別の利きチャンネル (ch 14-27 or 45-58)
                f1[0, ch_base + PIECETYPE_NUM + piece_idx, fi, ri] = 1.0

                # 利き数カウントを蓄積
                attack_count[color_slot][fi][ri] += 1

    # 利き数カウントチャンネル (ch 28-30 or 59-61)
    for color_slot in range(2):
        ch_base = FEATURES1_GROUP * color_slot
        for fi in range(9):
            for ri in range(9):
                cnt = attack_count[color_slot][fi][ri]
                for k in range(min(cnt, MAX_ATTACK_NUM)):
                    f1[0, ch_base + PIECETYPE_NUM * 2 + k, fi, ri] = 1.0


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
    # FEATURES1: 駒の位置 (ch 0-13, 31-44)
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
                fi = 8 - col
                ri = row
            else:
                fi = col
                ri = 8 - row

            # 自駒: ch 0-13、相手駒: ch 31-44
            ch_offset = 0 if is_own else FEATURES1_GROUP
            f1[0, ch_offset + piece_idx, fi, ri] = 1.0

    # ------------------------------------------------------------------ #
    # FEATURES1: 利きチャンネル (ch 14-27, 28-30, 45-58, 59-61)
    # ------------------------------------------------------------------ #
    _set_attack_planes(f1, board_2d, is_black)

    # ------------------------------------------------------------------ #
    # FEATURES2: 持ち駒
    # ------------------------------------------------------------------ #
    cur_key = "black" if is_black else "white"
    opp_key = "white" if is_black else "black"
    cur_order = [p for p, _ in HAND_TYPES]
    opp_order = [p.lower() for p, _ in HAND_TYPES]
    if not is_black:
        cur_order, opp_order = opp_order, cur_order

    ch = 0
    for (_, max_count), piece in zip(HAND_TYPES, cur_order):
        count = min(hands[cur_key].get(piece, 0), max_count)
        for i in range(count):
            f2[0, ch + i, :, :] = 1.0
        ch += max_count

    for (_, max_count), piece in zip(HAND_TYPES, opp_order):
        count = min(hands[opp_key].get(piece, 0), max_count)
        for i in range(count):
            f2[0, ch + i, :, :] = 1.0
        ch += max_count

    # 王手フラグ (channel 56)
    if in_check:
        f2[0, ch, :, :] = 1.0

    return f1, f2
