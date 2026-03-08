"""
将棋 UI バックエンド (FastAPI)
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.inference import DlshogiInference
from app.policy_decoder import policy_to_heatmap
from app.policy_move_mapper import move_to_policy_idx
from app.shogi_engine import ShogiBoard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# グローバルインスタンス
# --------------------------------------------------------------------------- #

app = FastAPI(title="Shogi UI")

board = ShogiBoard()
engine = DlshogiInference()

# 対局モード: human_color は "black" or "white"
game_mode: dict = {"human_color": "black"}

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# ヘルパー
# --------------------------------------------------------------------------- #

def _run_inference(b: ShogiBoard) -> tuple[list[list[float]], float]:
    """盤面から推論を実行してヒートマップと評価値を返す。"""
    try:
        f1, f2 = b.to_features()
        policy, value = engine.infer(f1, f2)
        heatmap = policy_to_heatmap(policy).tolist()
    except Exception as exc:
        logger.warning("Inference failed: %s", exc)
        heatmap = [[0.0] * 9 for _ in range(9)]
        value = 0.0
    return heatmap, value


def _board_response(b: ShogiBoard) -> dict:
    d = b.to_dict()
    heatmap, value = _run_inference(b)
    d["policy_heatmap"] = heatmap
    d["value"] = value
    return d


# --------------------------------------------------------------------------- #
# ルーティング
# --------------------------------------------------------------------------- #

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/new_game")
async def new_game():
    """初期局面にリセットして局面情報を返す。"""
    board.reset()
    resp = _board_response(board)
    resp["human_color"] = game_mode["human_color"]
    return resp


class MoveRequest(BaseModel):
    from_sq: Optional[list[int]] = None  # [row, col] or None（持ち駒打ち）
    to_sq: list[int]                     # [row, col]
    promote: bool = False
    piece: Optional[str] = None          # 持ち駒打ちの場合に必要


@app.post("/api/move")
async def make_move(req: MoveRequest):
    """指し手を適用して局面情報を返す。"""
    if req.from_sq is None:
        # 持ち駒打ち
        if req.piece is None:
            raise HTTPException(status_code=400, detail="piece is required for drop moves")
        success = board.apply_drop(req.piece, req.to_sq)
    else:
        success = board.apply_move(req.from_sq, req.to_sq, req.promote)

    if not success:
        raise HTTPException(status_code=400, detail="Invalid move")

    resp = _board_response(board)
    resp["human_color"] = game_mode["human_color"]
    return resp


@app.get("/api/state")
async def get_state():
    """現在の局面情報を返す。"""
    resp = _board_response(board)
    resp["human_color"] = game_mode["human_color"]
    return resp


class SetModeRequest(BaseModel):
    human_color: str  # "black" or "white"


@app.post("/api/set_mode")
async def set_mode(req: SetModeRequest):
    """対局モード（人間の手番色）を設定してゲームをリセット。"""
    if req.human_color not in ("black", "white"):
        raise HTTPException(status_code=400, detail="human_color must be 'black' or 'white'")
    game_mode["human_color"] = req.human_color
    board.reset()
    resp = _board_response(board)
    resp["human_color"] = game_mode["human_color"]
    return resp


@app.post("/api/ai_move")
async def ai_move():
    """AIが合法手の中から指し手を選択して実行する。"""
    moves = board.legal_moves()
    if not moves:
        raise HTTPException(status_code=400, detail="No legal moves")

    # 推論でpolicyを取得し、合法手にマッピングして最良手を選択
    try:
        f1, f2 = board.to_features()
        policy_np, _ = engine.infer(f1, f2)
        probs = np.exp(policy_np - policy_np.max())
        probs /= probs.sum()

        is_black = (board.turn == "black")
        scores = []
        for m in moves:
            idx = move_to_policy_idx(m, is_black)
            if idx is not None and 0 <= idx < len(probs):
                scores.append(float(probs[idx]))
            else:
                scores.append(0.0)

        best_idx = int(np.argmax(scores))
    except Exception:
        best_idx = random.randint(0, len(moves) - 1)

    best_move = moves[best_idx]

    if best_move["from"] is None:
        board.apply_drop(best_move["piece"], best_move["to"])
    else:
        board.apply_move(best_move["from"], best_move["to"], best_move.get("promote", False))

    resp = _board_response(board)
    resp["human_color"] = game_mode["human_color"]
    return resp
