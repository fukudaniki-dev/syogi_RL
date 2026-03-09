/**
 * 将棋 UI — board.js
 * Canvas (540×540) に 9×9 盤面を描画する。
 * 1 マス = 60px
 */

"use strict";

// ====================================================
// 定数・テーブル
// ====================================================

const CELL = 60;          // 1 マスのピクセルサイズ
const BOARD_SIZE = 9 * CELL; // 540px

/** 駒の日本語表記 */
const PIECE_LABEL = {
  // 先手（大文字）
  P: "歩",  L: "香",  N: "桂",  S: "銀",
  G: "金",  B: "角",  R: "飛",  K: "王",
  "+P": "と", "+L": "杏", "+N": "圭", "+S": "全",
  "+B": "馬", "+R": "龍",
  // 後手（小文字）
  p: "歩",  l: "香",  n: "桂",  s: "銀",
  g: "金",  b: "角",  r: "飛",  k: "玉",
  "+p": "と", "+l": "杏", "+n": "圭", "+s": "全",
  "+b": "馬", "+r": "龍",
};

/** 持ち駒表示用（先手側の大文字キー → 日本語） */
const HAND_LABEL = {
  P: "歩", L: "香", N: "桂", S: "銀", G: "金", B: "角", R: "飛",
  p: "歩", l: "香", n: "桂", s: "銀", g: "金", b: "角", r: "飛",
};

// ====================================================
// 状態
// ====================================================

let state = null;          // サーバーから返された最新局面
let humanColor = "black";  // 人間が操作する手番色
let aiThinking = false;    // AI思考中フラグ

let selectedFrom = null;   // { type: "board", row, col } or { type: "hand", piece }
let selectedDropPiece = null; // 持ち駒打ち選択中の駒

// ヒントモード: チェックボックスの状態
function isHintMode() {
  const cb = document.getElementById("hint-mode-checkbox");
  return cb ? cb.checked : false;
}

// ====================================================
// Canvas 取得
// ====================================================

const canvas = document.getElementById("shogi-board");
const ctx = canvas.getContext("2d");

// ====================================================
// API 通信
// ====================================================

async function apiPost(url, body = null) {
  const opts = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  };
  if (body !== null) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ====================================================
// 描画: ヒートマップ色
// ====================================================

/**
 * 値 0〜1 を透明→黄→赤のグラデーション色へ変換
 * 低確率は透明に近く、高確率は濃い赤になる
 */
function heatmapColor(v) {
  const clamped = Math.max(0, Math.min(1, v));
  // 0→透明, 0.5→黄色, 1→赤
  let r, g, b, a;
  if (clamped < 0.5) {
    const t = clamped * 2;
    r = Math.round(255 * t);
    g = Math.round(200 * t);
    b = 0;
    a = 0.15 + 0.4 * t;
  } else {
    const t = (clamped - 0.5) * 2;
    r = 255;
    g = Math.round(200 * (1 - t));
    b = 0;
    a = 0.55 + 0.35 * t;
  }
  return `rgba(${r}, ${g}, ${b}, ${a.toFixed(2)})`;
}

// ====================================================
// 描画: メイン
// ====================================================

function drawBoard() {
  ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);

  if (!state) return;

  const { board, legal_moves, last_move, hint_moves } = state;
  const hint = isHintMode();
  console.log("[drawBoard] hint=", hint, "hint_moves=", hint_moves);

  // ================================================================
  // LAYER 1: マス背景 + オーバーレイ（直前の手・選択・合法手）
  // ================================================================
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const x = c * CELL;
      const y = r * CELL;

      // 背景
      ctx.fillStyle = "#e8c87a";
      ctx.fillRect(x, y, CELL, CELL);

      // 直前の手ハイライト（常時表示）
      if (last_move) {
        if (last_move.from && last_move.from[0] === r && last_move.from[1] === c) {
          ctx.fillStyle = "rgba(255, 230, 80, 0.35)";
          ctx.fillRect(x, y, CELL, CELL);
        }
        if (last_move.to[0] === r && last_move.to[1] === c) {
          ctx.fillStyle = "rgba(255, 200, 0, 0.55)";
          ctx.fillRect(x, y, CELL, CELL);
        }
      }

      // ヒントモード: 上位手の移動元(通常手)または移動先(打ち手)をライトブルーで強調
      if (hint && hint_moves && hint_moves.length > 0) {
        for (let i = 0; i < hint_moves.length; i++) {
          const hm = hint_moves[i];
          const alpha = (0.55 - i * 0.08).toFixed(2);
          if (hm.from && hm.from[0] === r && hm.from[1] === c) {
            // 通常手: 移動元を青ハイライト
            ctx.fillStyle = `rgba(60, 160, 255, ${alpha})`;
            ctx.fillRect(x, y, CELL, CELL);
            break;
          }
          if (!hm.from && hm.to[0] === r && hm.to[1] === c) {
            // 打ち手: 移動先を紫ハイライト
            ctx.fillStyle = `rgba(180, 80, 255, ${alpha})`;
            ctx.fillRect(x, y, CELL, CELL);
            break;
          }
        }
      }

      // 選択ハイライト
      if (
        selectedFrom &&
        selectedFrom.type === "board" &&
        selectedFrom.row === r &&
        selectedFrom.col === c
      ) {
        ctx.fillStyle = "rgba(255, 200, 0, 0.6)";
        ctx.fillRect(x, y, CELL, CELL);
      }

      // 合法手ハイライト
      if (selectedFrom) {
        const isLegal = legal_moves.some((m) => {
          if (selectedFrom.type === "board") {
            return (
              m.from &&
              m.from[0] === selectedFrom.row &&
              m.from[1] === selectedFrom.col &&
              m.to[0] === r &&
              m.to[1] === c
            );
          } else {
            return m.from === null && m.to[0] === r && m.to[1] === c;
          }
        });
        if (isLegal) {
          ctx.fillStyle = "rgba(0, 180, 80, 0.35)";
          ctx.fillRect(x, y, CELL, CELL);
        }
      }
    }
  }

  // ================================================================
  // LAYER 2: グリッド線
  // ================================================================
  ctx.strokeStyle = "#6b4f2a";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 9; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL, 0);
    ctx.lineTo(i * CELL, BOARD_SIZE);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i * CELL);
    ctx.lineTo(BOARD_SIZE, i * CELL);
    ctx.stroke();
  }

  // ================================================================
  // LAYER 3: 駒
  // ================================================================
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const piece = board[r][c];
      if (!piece) continue;

      const x = c * CELL;
      const y = r * CELL;
      const isBlack = piece === piece.toUpperCase() && piece.toUpperCase() !== piece.toLowerCase();
      const isPromoted = piece.startsWith("+");

      drawPieceShape(ctx, x, y, CELL, isBlack, isPromoted);

      const label = PIECE_LABEL[piece] || piece;
      ctx.save();
      if (!isBlack) {
        ctx.translate(x + CELL / 2, y + CELL / 2);
        ctx.rotate(Math.PI);
        ctx.translate(-(x + CELL / 2), -(y + CELL / 2));
      }
      ctx.fillStyle = isPromoted ? "#cc2200" : "#111";
      ctx.font = `bold ${Math.round(CELL * 0.42)}px "Hiragino Kaku Gothic ProN", serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + CELL / 2, y + CELL / 2);
      ctx.restore();
    }
  }

  // ================================================================
  // LAYER 4: 直前の手の枠線（駒の上に重ねる）
  // ================================================================
  if (last_move) {
    ctx.lineWidth = 2.5;
    if (last_move.from) {
      const [fr, fc] = last_move.from;
      ctx.strokeStyle = "rgba(200, 160, 0, 0.7)";
      ctx.strokeRect(fc * CELL + 1.5, fr * CELL + 1.5, CELL - 3, CELL - 3);
    }
    const [tr, tc] = last_move.to;
    ctx.strokeStyle = "rgba(220, 140, 0, 0.95)";
    ctx.strokeRect(tc * CELL + 1.5, tr * CELL + 1.5, CELL - 3, CELL - 3);
  }

  // ================================================================
  // LAYER 5: ヒントモード矢印（駒の上に重ねる）
  // ================================================================
  if (hint && hint_moves && hint_moves.length > 0) {
    const topScore = hint_moves[0].score || 1;
    hint_moves.forEach((hm, i) => {
      const [tr, tc] = hm.to;
      const alpha = Math.max(0.25, hm.score / topScore);
      const color = i === 0 ? "#ff2200" : i === 1 ? "#ff7700" : "#ffcc00";

      if (!hm.from) {
        // 打ち手: 目的地に「駒名＋打」マークを円で表示
        drawDropHint(tc * CELL + CELL / 2, tr * CELL + CELL / 2, hm.piece, color, alpha);
      } else {
        // 通常手: 矢印
        const [fr, fc] = hm.from;
        drawHintArrow(
          fc * CELL + CELL / 2, fr * CELL + CELL / 2,
          tc * CELL + CELL / 2, tr * CELL + CELL / 2,
          color, alpha
        );
      }
    });
  }
}

/**
 * ヒント矢印を描画する（移動元 → 移動先）
 */
function drawHintArrow(x1, y1, x2, y2, color, alpha) {
  const headLen = 14;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  // 矢先が駒に少し刺さらないよう終点を少し手前に
  const ex = x2 - Math.cos(angle) * 6;
  const ey = y2 - Math.sin(angle) * 6;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";

  // 軸
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(ex, ey);
  ctx.stroke();

  // 矢頭
  ctx.beginPath();
  ctx.moveTo(ex, ey);
  ctx.lineTo(
    ex - headLen * Math.cos(angle - Math.PI / 6),
    ey - headLen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    ex - headLen * Math.cos(angle + Math.PI / 6),
    ey - headLen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

/**
 * 打ち手ヒントを描画する（目的地に駒名＋「打」を円で表示）
 */
function drawDropHint(cx, cy, piece, color, alpha) {
  const label = HAND_LABEL[piece] || piece;
  const r = CELL * 0.38;

  ctx.save();
  ctx.globalAlpha = alpha;

  // 塗り円
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  // 円の枠
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.stroke();

  // テキスト: 上段に駒名、下段に「打」
  ctx.globalAlpha = Math.min(1.0, alpha * 1.4);
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `bold ${Math.round(CELL * 0.28)}px "Hiragino Kaku Gothic ProN", serif`;
  ctx.fillText(label, cx, cy - CELL * 0.1);
  ctx.font = `${Math.round(CELL * 0.22)}px "Hiragino Kaku Gothic ProN", serif`;
  ctx.fillText("打", cx, cy + CELL * 0.14);

  ctx.restore();
}

/**
 * 駒の五角形を描画する
 */
function drawPieceShape(ctx, x, y, size, isBlack, isPromoted) {
  const pad = 4;
  const tipInset = size * 0.22;

  ctx.beginPath();
  if (isBlack) {
    // 先手: 上が尖った五角形
    ctx.moveTo(x + size / 2, y + pad);              // 上頂点
    ctx.lineTo(x + size - pad, y + tipInset);        // 右上
    ctx.lineTo(x + size - pad, y + size - pad);      // 右下
    ctx.lineTo(x + pad, y + size - pad);             // 左下
    ctx.lineTo(x + pad, y + tipInset);               // 左上
  } else {
    // 後手: 下が尖った五角形（180度反転）
    ctx.moveTo(x + size / 2, y + size - pad);        // 下頂点
    ctx.lineTo(x + size - pad, y + size - tipInset); // 右下
    ctx.lineTo(x + size - pad, y + pad);             // 右上
    ctx.lineTo(x + pad, y + pad);                    // 左上
    ctx.lineTo(x + pad, y + size - tipInset);        // 左下
  }
  ctx.closePath();

  ctx.fillStyle = "#fdf6e3";
  ctx.fill();
  ctx.strokeStyle = "#6b4f2a";
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

// ====================================================
// 描画: Value ゲージ
// ====================================================

function updateValueGauge(value) {
  // value: -1(後手有利) 〜 +1(先手有利)
  const bar = document.getElementById("value-gauge-bar");
  const text = document.getElementById("value-text");

  // ゲージバー: 中央(50%) を基点にして先手側(左)または後手側(右)へ
  const pct = ((value + 1) / 2) * 100; // 0〜100
  if (value >= 0) {
    bar.style.left = "50%";
    bar.style.width = `${pct - 50}%`;
    bar.style.background = "rgba(30, 30, 180, 0.55)";
  } else {
    bar.style.left = `${pct}%`;
    bar.style.width = `${50 - pct}%`;
    bar.style.background = "rgba(200, 30, 30, 0.55)";
  }

  text.textContent = value.toFixed(3);
}

// ====================================================
// 描画: 持ち駒
// ====================================================

function updateHandDisplay() {
  if (!state) return;

  renderHand("hand-black-pieces", state.hands.black, "black", Object.keys(HAND_LABEL));
  renderHand("hand-white-pieces", state.hands.white, "white", Object.keys(HAND_LABEL).map((k) => k.toLowerCase()));
}

function renderHand(containerId, handData, owner, pieceOrder) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  for (const piece of pieceOrder) {
    const count = handData[piece] || 0;
    if (count === 0) continue;

    const btn = document.createElement("button");
    btn.className = "hand-piece-btn";
    if (
      selectedFrom &&
      selectedFrom.type === "hand" &&
      selectedFrom.piece === piece
    ) {
      btn.classList.add("selected");
    }

    const nameSpan = document.createElement("span");
    nameSpan.className = "piece-name";
    nameSpan.textContent = HAND_LABEL[piece] || piece;

    const countSpan = document.createElement("span");
    countSpan.className = "piece-count";
    countSpan.textContent = `×${count}`;

    btn.appendChild(nameSpan);
    btn.appendChild(countSpan);

    btn.addEventListener("click", () => onHandPieceClick(piece, owner));
    container.appendChild(btn);
  }
}

// ====================================================
// イベント: クリック（盤面）
// ====================================================

canvas.addEventListener("click", async (e) => {
  if (!state) return;
  if (aiThinking) return;                        // AI思考中は操作不可
  if (state.checkmate) return;                   // 詰みは操作不可
  if (state.turn !== humanColor) return;         // 自分の手番以外は操作不可

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const col = Math.floor(x / CELL);
  const row = Math.floor(y / CELL);

  if (col < 0 || col > 8 || row < 0 || row > 8) return;

  const piece = state.board[row][col];
  const turn = state.turn;
  const isBlackTurn = turn === "black";

  // ---- 持ち駒打ち選択中 ----
  if (selectedFrom && selectedFrom.type === "hand") {
    const isLegal = state.legal_moves.some(
      (m) => m.from === null && m.to[0] === row && m.to[1] === col
    );
    if (isLegal) {
      await doMove(null, [row, col], false, selectedFrom.piece);
    } else {
      selectedFrom = null;
      drawBoard();
      updateHandDisplay();
    }
    return;
  }

  // ---- 盤上の駒が選択済み ----
  if (selectedFrom && selectedFrom.type === "board") {
    const fr = selectedFrom.row;
    const fc = selectedFrom.col;

    // 同じマスをクリック → 選択解除
    if (fr === row && fc === col) {
      selectedFrom = null;
      drawBoard();
      return;
    }

    // 合法手か確認
    const candidates = state.legal_moves.filter(
      (m) =>
        m.from &&
        m.from[0] === fr &&
        m.from[1] === fc &&
        m.to[0] === row &&
        m.to[1] === col
    );

    if (candidates.length > 0) {
      // 成れる場合は confirm で確認
      const canPromote = candidates.some((m) => m.promote);
      const mustPromote = candidates.length === 1 && candidates[0].promote;
      let promote = false;

      if (mustPromote) {
        promote = true;
      } else if (canPromote) {
        promote = confirm("成りますか？");
      }

      await doMove([fr, fc], [row, col], promote, null);
      return;
    }

    // 自分の駒をクリック → 選択変更
    if (piece && isOwnPiece(piece, isBlackTurn)) {
      selectedFrom = { type: "board", row, col };
      drawBoard();
      return;
    }

    // それ以外 → 選択解除
    selectedFrom = null;
    drawBoard();
    return;
  }

  // ---- 何も選択していない ----
  if (piece && isOwnPiece(piece, isBlackTurn)) {
    selectedFrom = { type: "board", row, col };
    drawBoard();
  }
});

function isOwnPiece(piece, isBlackTurn) {
  if (isBlackTurn) {
    return piece === piece.toUpperCase() && !piece.startsWith("+") ||
      piece.startsWith("+") && piece[1] === piece[1].toUpperCase();
  } else {
    return piece === piece.toLowerCase() ||
      piece.startsWith("+") && piece[1] === piece[1].toLowerCase();
  }
}

// ====================================================
// イベント: 持ち駒クリック
// ====================================================

function onHandPieceClick(piece, owner) {
  if (!state) return;
  if (aiThinking) return;
  if (state.turn !== humanColor) return;         // 自分の手番以外は操作不可

  const isBlackTurn = state.turn === "black";
  if ((owner === "black") !== isBlackTurn) return; // 手番外は無視

  if (selectedFrom && selectedFrom.type === "hand" && selectedFrom.piece === piece) {
    selectedFrom = null;
  } else {
    selectedFrom = { type: "hand", piece };
  }
  drawBoard();
  updateHandDisplay();
}

// ====================================================
// 指し手実行
// ====================================================

async function doMove(fromSq, toSq, promote, piece) {
  setStatus("");
  try {
    const body = { from_sq: fromSq, to_sq: toSq, promote: promote };
    if (piece !== null) body.piece = piece;
    const newState = await apiPost("/api/move", body);
    applyState(newState);
    // 人間が指した後、AIの手番かつ詰みでなければ自動でAI指し手を実行
    if (!newState.checkmate && newState.turn !== humanColor && newState.legal_moves.length > 0) {
      await triggerAiMove();
    }
  } catch (err) {
    setStatus(`エラー: ${err.message}`);
  } finally {
    selectedFrom = null;
  }
}

async function triggerAiMove() {
  aiThinking = true;
  setAiThinking(true);
  try {
    const newState = await apiPost("/api/ai_move");
    applyState(newState);
  } catch (err) {
    setStatus(`AI エラー: ${err.message}`);
  } finally {
    aiThinking = false;
    setAiThinking(false);
  }
}

function setAiThinking(on) {
  const el = document.getElementById("ai-thinking");
  if (el) el.style.display = on ? "inline" : "none";
}

// ====================================================
// 状態適用 & 全体再描画
// ====================================================

function applyState(newState) {
  state = newState;
  if (newState.human_color) humanColor = newState.human_color;
  selectedFrom = null;
  drawBoard();
  updateHandDisplay();
  updateValueGauge(state.value ?? 0);

  const turnLabel = state.turn === "black" ? "先手" : "後手";
  const isMyTurn = state.turn === humanColor;

  if (state.checkmate) {
    // 詰み: 手番側の負け
    const loser = state.turn === "black" ? "先手" : "後手";
    const winner = state.turn === "black" ? "後手" : "先手";
    setStatus(`詰み！ ${winner}の勝ち`);
    document.getElementById("turn-label").textContent = `${loser} 詰み`;
  } else if (state.in_check) {
    document.getElementById("turn-label").textContent =
      `手番: ${turnLabel}${isMyTurn ? " (あなた)" : " (AI)"} 【王手！】`;
  } else {
    document.getElementById("turn-label").textContent =
      `手番: ${turnLabel}${isMyTurn ? " (あなた)" : " (AI)"}`;
  }
}

function setStatus(msg) {
  document.getElementById("status-msg").textContent = msg;
}

// ====================================================
// ヒートマップ凡例
// ====================================================

function drawLegend() {
  const lc = document.getElementById("legend-canvas");
  const lctx = lc.getContext("2d");
  const grad = lctx.createLinearGradient(0, 0, lc.width, 0);
  grad.addColorStop(0, heatmapColor(0));
  grad.addColorStop(0.5, heatmapColor(0.5));
  grad.addColorStop(1, heatmapColor(1));
  lctx.fillStyle = grad;
  lctx.fillRect(0, 0, lc.width, lc.height);
}

// ====================================================
// ボタン
// ====================================================

document.getElementById("hint-mode-checkbox").addEventListener("change", () => {
  drawBoard();
});

document.getElementById("btn-new-game").addEventListener("click", async () => {
  setStatus("");
  aiThinking = false;
  setAiThinking(false);
  try {
    const select = document.getElementById("human-color-select");
    const chosenColor = select ? select.value : "black";
    // モード設定 → ゲームリセット
    const newState = await apiPost("/api/set_mode", { human_color: chosenColor });
    applyState(newState);
    // 人間が後手を選んだ場合、先にAIが指す
    if (newState.turn !== humanColor) {
      await triggerAiMove();
    }
  } catch (err) {
    setStatus(`エラー: ${err.message}`);
  }
});

// ====================================================
// 初期化
// ====================================================

(async function init() {
  drawLegend();
  try {
    const s = await apiGet("/api/state");
    applyState(s);
  } catch {
    // 初回起動時はゲーム未開始の場合もある → new_game を呼ぶ
    try {
      const s = await apiPost("/api/new_game");
      applyState(s);
    } catch (err) {
      setStatus(`初期化エラー: ${err.message}`);
    }
  }
})();
