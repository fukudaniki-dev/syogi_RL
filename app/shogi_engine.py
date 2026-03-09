"""
平手将棋の盤面管理モジュール。

座標系:
  board[row][col]
  - row=0: 9段目（後手陣・盤面上部）
  - row=8: 1段目（先手陣・盤面下部）
  - col=0: 9筋, col=8: 1筋

駒表記（文字列）:
  先手（大文字）: P(歩) L(香) N(桂) S(銀) G(金) B(角) R(飛) K(王)
  後手（小文字）: p l n s g b r k
  成り（先手）:   +P +L +N +S +B +R
  成り（後手）:   +p +l +n +s +b +r
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #

BLACK = "black"
WHITE = "white"

# 先手の成れる駒と成り後
PROMOTE_MAP: Dict[str, str] = {
    "P": "+P",
    "L": "+L",
    "N": "+N",
    "S": "+S",
    "B": "+B",
    "R": "+R",
    "p": "+p",
    "l": "+l",
    "n": "+n",
    "s": "+s",
    "b": "+b",
    "r": "+r",
}

# 成り駒 → 元の駒（持ち駒に戻す際）
DEMOTE_MAP: Dict[str, str] = {
    "+P": "P",
    "+L": "L",
    "+N": "N",
    "+S": "S",
    "+B": "B",
    "+R": "R",
    "+p": "p",
    "+l": "l",
    "+n": "n",
    "+s": "s",
    "+b": "b",
    "+r": "r",
}

BLACK_PIECES = set("PLNSGBRK") | {"+P", "+L", "+N", "+S", "+B", "+R"}
WHITE_PIECES = set("plnsgbrk") | {"+p", "+l", "+n", "+s", "+b", "+r"}

HAND_PIECES_BLACK = ["P", "L", "N", "S", "G", "B", "R"]
HAND_PIECES_WHITE = ["p", "l", "n", "s", "g", "b", "r"]


def _is_black(piece: str) -> bool:
    return piece in BLACK_PIECES


def _is_white(piece: str) -> bool:
    return piece in WHITE_PIECES


def _owner(piece: str) -> Optional[str]:
    if _is_black(piece):
        return BLACK
    if _is_white(piece):
        return WHITE
    return None


# --------------------------------------------------------------------------- #
# 盤面ユーティリティ
# --------------------------------------------------------------------------- #

def _in_board(r: int, c: int) -> bool:
    return 0 <= r <= 8 and 0 <= c <= 8


def _piece_attacks(
    piece: str, row: int, col: int, board_2d: List[List[str]]
) -> List[Tuple[int, int]]:
    """
    指定駒が (row, col) から攻撃できるマスの一覧を返す。
    盤面状態（board_2d）を利用してスライド駒の遮蔽を処理する。
    """
    results: List[Tuple[int, int]] = []

    def add(r: int, c: int) -> None:
        if _in_board(r, c):
            results.append((r, c))

    def slide(dr: int, dc: int) -> None:
        r, c = row + dr, col + dc
        while _in_board(r, c):
            results.append((r, c))
            if board_2d[r][c]:  # 駒があったら止まる
                break
            r += dr
            c += dc

    p = piece
    if p == "P":
        add(row - 1, col)
    elif p == "p":
        add(row + 1, col)
    elif p == "L":
        slide(-1, 0)
    elif p == "l":
        slide(+1, 0)
    elif p == "N":
        add(row - 2, col - 1); add(row - 2, col + 1)
    elif p == "n":
        add(row + 2, col - 1); add(row + 2, col + 1)
    elif p == "S":
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)]:
            add(row + dr, col + dc)
    elif p == "s":
        for dr, dc in [(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)]:
            add(row + dr, col + dc)
    elif p in ("G", "+P", "+L", "+N", "+S"):
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]:
            add(row + dr, col + dc)
    elif p in ("g", "+p", "+l", "+n", "+s"):
        for dr, dc in [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)]:
            add(row + dr, col + dc)
    elif p in ("B", "b"):
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            slide(dr, dc)
    elif p in ("R", "r"):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            slide(dr, dc)
    elif p in ("+B", "+b"):
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            slide(dr, dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            add(row + dr, col + dc)
    elif p in ("+R", "+r"):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            slide(dr, dc)
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            add(row + dr, col + dc)
    elif p in ("K", "k"):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                add(row + dr, col + dc)
    return results


def _is_square_attacked(
    board_2d: List[List[str]], row: int, col: int, by_color: str
) -> bool:
    """(row, col) が by_color の駒に攻撃されているか判定する。"""
    attacker_set = BLACK_PIECES if by_color == BLACK else WHITE_PIECES
    for r in range(9):
        for c in range(9):
            piece = board_2d[r][c]
            if not piece or piece not in attacker_set:
                continue
            if (row, col) in _piece_attacks(piece, r, c, board_2d):
                return True
    return False


# --------------------------------------------------------------------------- #
# ShogiBoard
# --------------------------------------------------------------------------- #

class ShogiBoard:
    """平手将棋の盤面を管理するクラス。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """初期配置に戻す。"""
        # 9x9 board。空マスは ""
        self.board: List[List[str]] = [[""] * 9 for _ in range(9)]

        # 後手 9段目 (row=0): l n s g k g s n l
        self.board[0] = ["l", "n", "s", "g", "k", "g", "s", "n", "l"]
        # 後手 8段目 (row=1): _ r _ _ _ _ _ b _
        self.board[1] = ["", "r", "", "", "", "", "", "b", ""]
        # 後手 7段目 (row=2): p p p p p p p p p
        self.board[2] = ["p"] * 9
        # 空 6段目〜4段目
        for r in range(3, 6):
            self.board[r] = [""] * 9
        # 先手 3段目 (row=6): P P P P P P P P P
        self.board[6] = ["P"] * 9
        # 先手 2段目 (row=7): _ B _ _ _ _ _ R _
        self.board[7] = ["", "B", "", "", "", "", "", "R", ""]
        # 先手 1段目 (row=8): L N S G K G S N L
        self.board[8] = ["L", "N", "S", "G", "K", "G", "S", "N", "L"]

        # 持ち駒
        self.hands: Dict[str, Dict[str, int]] = {
            BLACK: {p: 0 for p in HAND_PIECES_BLACK},
            WHITE: {p: 0 for p in ["p", "l", "n", "s", "g", "b", "r"]},
        }

        self.turn: str = BLACK
        self.last_move: Optional[dict] = None  # {"from": [r,c] or None, "to": [r,c]}

    # ------------------------------------------------------------------ #
    # 合法手生成
    # ------------------------------------------------------------------ #

    def legal_moves(self) -> List[dict]:
        """
        現在の手番の合法手をリストで返す。
        自玉が王手になる手を除外した真の合法手を返す。
        各要素: {"from": [row, col] or None, "to": [row, col], "promote": bool, "piece": str}
        """
        pseudo = self._pseudo_legal_moves()
        king = "K" if self.turn == BLACK else "k"
        opponent = WHITE if self.turn == BLACK else BLACK
        legal: List[dict] = []

        for m in pseudo:
            new_board = self._simulate_board(m)
            # 自玉の位置を探す
            king_pos: Optional[Tuple[int, int]] = None
            for r in range(9):
                for c in range(9):
                    if new_board[r][c] == king:
                        king_pos = (r, c)
                        break
                if king_pos:
                    break
            if king_pos is None:
                continue  # 王が盤上にない（通常は起こらない）
            # 自玉が敵に攻撃されていなければ合法手
            if not _is_square_attacked(new_board, king_pos[0], king_pos[1], opponent):
                legal.append(m)

        return legal

    def _pseudo_legal_moves(self) -> List[dict]:
        """王手チェックなしの疑似合法手を生成する。"""
        moves: List[dict] = []
        is_black = self.turn == BLACK

        # 盤上の駒移動
        for r in range(9):
            for c in range(9):
                piece = self.board[r][c]
                if not piece:
                    continue
                if is_black and not _is_black(piece):
                    continue
                if not is_black and not _is_white(piece):
                    continue
                for tr, tc, can_promote in self._piece_destinations(piece, r, c):
                    if not _in_board(tr, tc):
                        continue
                    target = self.board[tr][tc]
                    if target and _owner(target) == self.turn:
                        continue  # 自駒には移動不可
                    moves.append(
                        {"from": [r, c], "to": [tr, tc], "promote": False, "piece": piece}
                    )
                    if can_promote:
                        moves.append(
                            {"from": [r, c], "to": [tr, tc], "promote": True, "piece": piece}
                        )

        # 持ち駒打ち
        hand_pieces = HAND_PIECES_BLACK if is_black else ["p", "l", "n", "s", "g", "b", "r"]
        for piece in hand_pieces:
            if self.hands[self.turn].get(piece, 0) <= 0:
                continue
            for r in range(9):
                for c in range(9):
                    if self.board[r][c]:
                        continue
                    if piece in ("P", "p") and self._has_pawn_in_col(c):
                        continue
                    if not self._can_drop(piece, r):
                        continue
                    moves.append({"from": None, "to": [r, c], "promote": False, "piece": piece})

        return moves

    def _simulate_board(self, move: dict) -> List[List[str]]:
        """指し手をシミュレートした盤面コピーを返す（hands は変更しない）。"""
        new_board = [row[:] for row in self.board]
        fr_list = move["from"]
        tr, tc = move["to"]

        if fr_list is None:
            # 持ち駒打ち
            new_board[tr][tc] = move["piece"]
        else:
            fr, fc = fr_list
            piece = new_board[fr][fc]
            new_board[fr][fc] = ""
            if move.get("promote") and piece in PROMOTE_MAP:
                new_board[tr][tc] = PROMOTE_MAP[piece]
            else:
                new_board[tr][tc] = piece

        return new_board

    def is_in_check(self) -> bool:
        """現在の手番の玉が王手されているか返す。"""
        king = "K" if self.turn == BLACK else "k"
        opponent = WHITE if self.turn == BLACK else BLACK
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == king:
                    return _is_square_attacked(self.board, r, c, opponent)
        return False

    def _has_pawn_in_col(self, col: int) -> bool:
        """現在の手番の歩が同じ列にあるか確認（二歩チェック）。"""
        target = "P" if self.turn == BLACK else "p"
        for r in range(9):
            if self.board[r][col] == target:
                return True
        return False

    def _can_drop(self, piece: str, row: int) -> bool:
        """行き所のない駒を打てないルール。"""
        base = piece.lower()
        if base == "p" or base == "l":
            # 先手: row=0 (9段目) に打てない, 後手: row=8 に打てない
            if self.turn == BLACK and row == 0:
                return False
            if self.turn == WHITE and row == 8:
                return False
        if base == "n":
            if self.turn == BLACK and row <= 1:
                return False
            if self.turn == WHITE and row >= 7:
                return False
        return True

    def _piece_destinations(
        self, piece: str, row: int, col: int
    ) -> List[Tuple[int, int, bool]]:
        """
        駒の移動先リストを返す。
        戻り値: [(to_row, to_col, can_promote), ...]
        can_promote は成れる条件（敵陣 or 出発が敵陣）を満たす場合 True
        """
        results: List[Tuple[int, int, bool]] = []
        is_black = self.turn == BLACK

        def enemy_zone(r: int) -> bool:
            """その行が敵陣かどうか（先手: rows 0-2, 後手: rows 6-8）。"""
            if is_black:
                return r <= 2
            else:
                return r >= 6

        in_enemy = enemy_zone(row)

        def add(tr: int, tc: int) -> None:
            if not _in_board(tr, tc):
                return
            cp = (
                (in_enemy or enemy_zone(tr))
                and piece not in ("K", "k", "G", "g", "+P", "+L", "+N", "+S", "+B", "+R",
                                  "+p", "+l", "+n", "+s", "+b", "+r")
            )
            results.append((tr, tc, cp))

        def add_slide(dr: int, dc: int) -> None:
            r, c = row + dr, col + dc
            while _in_board(r, c):
                target = self.board[r][c]
                cp = (
                    (in_enemy or enemy_zone(r))
                    and piece not in ("K", "k", "G", "g", "+P", "+L", "+N", "+S", "+B", "+R",
                                      "+p", "+l", "+n", "+s", "+b", "+r")
                )
                results.append((r, c, cp))
                if target:
                    break  # 駒があったら止まる（自駒への着地は legal_moves でフィルタ）
                r += dr
                c += dc

        p = piece

        if p == "P":  # 先手 歩
            add(row - 1, col)
        elif p == "p":  # 後手 歩
            add(row + 1, col)
        elif p == "L":  # 先手 香
            add_slide(-1, 0)
        elif p == "l":  # 後手 香
            add_slide(1, 0)
        elif p == "N":  # 先手 桂
            add(row - 2, col - 1)
            add(row - 2, col + 1)
        elif p == "n":  # 後手 桂
            add(row + 2, col - 1)
            add(row + 2, col + 1)
        elif p in ("S", "s"):  # 銀
            dirs = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)]
            if p == "s":
                dirs = [(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)]
            for dr, dc in dirs:
                add(row + dr, col + dc)
        elif p in ("G", "g", "+P", "+L", "+N", "+S", "+p", "+l", "+n", "+s"):
            # 金・金将相当
            if p in ("G", "+P", "+L", "+N", "+S"):
                dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
            else:
                dirs = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)]
            for dr, dc in dirs:
                add(row + dr, col + dc)
        elif p in ("B", "b"):  # 角
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                add_slide(dr, dc)
        elif p in ("R", "r"):  # 飛
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                add_slide(dr, dc)
        elif p in ("+B", "+b"):  # 龍馬（成角）
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                add_slide(dr, dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                add(row + dr, col + dc)
        elif p in ("+R", "+r"):  # 龍王（成飛）
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                add_slide(dr, dc)
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                add(row + dr, col + dc)
        elif p in ("K", "k"):  # 王・玉
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    add(row + dr, col + dc)

        return results

    # ------------------------------------------------------------------ #
    # 指し手実行
    # ------------------------------------------------------------------ #

    def apply_move(
        self,
        from_sq: Optional[List[int]],
        to_sq: List[int],
        promote: bool = False,
    ) -> bool:
        """
        指し手を適用する。

        Parameters
        ----------
        from_sq : [row, col] or None (持ち駒打ちの場合)
        to_sq   : [row, col]
        promote : 成るかどうか

        Returns
        -------
        bool : 成功したか
        """
        tr, tc = to_sq

        if from_sq is None:
            # 持ち駒打ち
            # どの駒を打つか: legal_moves で決めるが、ここでは手番に合う駒を探す
            # 呼び出し元が piece を指定できないため legal_moves から合致するものを使う
            # ※ API 側で from_sq=None の場合は piece も送ってもらう設計が理想だが、
            #   ここでは持ち駒の中で盤上に置ける駒を決定する（piece は別途渡す）
            return False  # drop は apply_drop を使う

        fr, fc = from_sq
        piece = self.board[fr][fc]
        if not piece:
            return False
        if _owner(piece) != self.turn:
            return False

        target = self.board[tr][tc]
        if target and _owner(target) == self.turn:
            return False  # 自駒に移動不可

        # 取った駒を持ち駒に
        if target:
            base = DEMOTE_MAP.get(target, target)
            # 後手の駒を先手が取る → 先手の持ち駒に
            captured = base.upper() if self.turn == BLACK else base.lower()
            self.hands[self.turn][captured] = self.hands[self.turn].get(captured, 0) + 1

        # 移動
        self.board[fr][fc] = ""
        if promote and piece in PROMOTE_MAP:
            self.board[tr][tc] = PROMOTE_MAP[piece]
        else:
            self.board[tr][tc] = piece

        self.last_move = {"from": [fr, fc], "to": [tr, tc]}
        self.turn = WHITE if self.turn == BLACK else BLACK
        return True

    def apply_drop(self, piece: str, to_sq: List[int]) -> bool:
        """持ち駒を打つ。"""
        tr, tc = to_sq
        if self.board[tr][tc]:
            return False
        hand_piece = piece if self.turn == BLACK else piece.lower()
        if self.hands[self.turn].get(hand_piece, 0) <= 0:
            return False
        self.hands[self.turn][hand_piece] -= 1
        self.board[tr][tc] = hand_piece if self.turn == WHITE else hand_piece.upper()
        self.last_move = {"from": None, "to": [tr, tc]}
        self.turn = WHITE if self.turn == BLACK else BLACK
        return True

    # ------------------------------------------------------------------ #
    # 特徴量（スタブ）
    # ------------------------------------------------------------------ #

    def to_features(self):
        """
        dlshogi 用特徴量テンソルを返す。
        feature_encoder.encode_features() を使って盤面状態をエンコードする。
        """
        from app.feature_encoder import encode_features

        in_check = self.is_in_check()
        f1_np, f2_np = encode_features(self.board, self.hands, self.turn, in_check)
        f1 = torch.from_numpy(f1_np)
        f2 = torch.from_numpy(f2_np)
        return f1, f2

    # ------------------------------------------------------------------ #
    # シリアライズ
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """API レスポンス用の辞書を返す。"""
        moves = self.legal_moves()
        legal = [
            {
                "from": m["from"],
                "to": m["to"],
                "promote": m["promote"],
            }
            for m in moves
        ]
        in_check = self.is_in_check()
        checkmate = in_check and len(moves) == 0
        return {
            "board": [row[:] for row in self.board],
            "hands": deepcopy(self.hands),
            "turn": self.turn,
            "legal_moves": legal,
            "in_check": in_check,
            "checkmate": checkmate,
            "last_move": self.last_move,
        }
