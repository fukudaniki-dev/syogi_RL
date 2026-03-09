# shogi_RL

dlshogi（DeepLearningShogi）を使った将棋AIとブラウザUIのリポジトリ。

## 動作環境

- Docker + NVIDIA Container Toolkit
- GPU: CUDA 12.x 対応（RTX 2060 / CUDA 13.1 ドライバで動作確認）
- dlshogi は git submodule として含まれる

---

## セットアップ

### 1. リポジトリのクローン

```bash
git clone --recurse-submodules <このリポジトリのURL>
cd shogi_RL
```

サブモジュールを後から初期化する場合:

```bash
git submodule update --init --recursive
```

### 2. 学習済みモデルの準備（任意）

学習済みモデルがない場合はランダム重みで動作します（AIは強くありません）。
dlshogi の GitHub Releases からモデルをダウンロードし、`models/` ディレクトリに配置してください。

```bash
mkdir -p models
cd models

# 例: WCSC29 モデル（wideresnet10 アーキテクチャ）
wget https://github.com/TadaoYamaoka/DeepLearningShogi/releases/download/wcsc29/dlshogi-wcsc29.zip
unzip dlshogi-wcsc29.zip
```

展開後のモデルファイルパスを確認して環境変数に設定します（次のステップ参照）。

### 3. 環境変数の設定（学習済みモデルを使う場合）

`docker/` ディレクトリに `.env` ファイルを作成します:

```bash
# wideresnet10 モデルを使う例（WCSC29 zip を展開した場合）
cat > docker/.env << 'EOF'
DLSHOGI_MODEL_PATH=/workspace/models/for_learning/model_rl_val_wideresnet10_selfplay_179
DLSHOGI_NETWORK=wideresnet10
EOF
```

> **注意**: WCSC29 zip を展開すると `models/for_learning/` 以下にモデルファイルが配置されます。

| 環境変数 | 説明 | デフォルト |
|---|---|---|
| `DLSHOGI_MODEL_PATH` | コンテナ内のモデルファイルパス | `""` (ランダム重み) |
| `DLSHOGI_NETWORK` | ネットワークアーキテクチャ名 | `resnet10_swish` |

利用可能なアーキテクチャ: `resnet10_swish`, `wideresnet10`, `resnet15_swish`, `resnet20_swish` など

---

## 起動

```bash
cd docker

# 初回ビルド（dlshogi C++ 拡張のビルドで数分かかる）
docker compose build

# 起動（.env ファイルがある場合）
docker compose --env-file .env up

# .env なし（ランダム重みで起動）
docker compose up
```

ブラウザで http://localhost:8000 にアクセスしてください。

---

## 将棋UIの使い方

1. **新規対局**: 「あなたの手番」で先手/後手を選び、「新規対局」ボタンをクリック
2. **駒の移動**: クリックで駒を選択 → 移動先をクリック（成れる場合は自動的に選択肢が出ます）
3. **持ち駒を打つ**: 持ち駒エリアの駒をクリック → 盤上の空きマスをクリック
4. **AI**: 人間が指した後、AIが自動的に応答します

### 表示内容

| 表示 | 説明 |
|---|---|
| Policy ヒートマップ | AIが注目しているマス（赤いほど確率が高い） |
| 評価値ゲージ | 先手有利 ←→ 後手有利（学習済みモデルが必要） |
| 【王手！】 | 現在の手番の玉が王手されている |

---

## ディレクトリ構成

```
shogi_RL/
├── app/                    # FastAPI バックエンド
│   ├── main.py             # API エンドポイント
│   ├── shogi_engine.py     # 将棋盤面・合法手生成
│   ├── feature_encoder.py  # dlshogi 入力特徴量エンコーダ
│   ├── policy_move_mapper.py # policy インデックス ↔ 指し手マッピング
│   ├── inference.py        # dlshogi モデル推論ラッパー
│   ├── policy_decoder.py   # policy → ヒートマップ変換
│   └── static/             # フロントエンド (HTML/CSS/JS)
├── dlshogi/                # git submodule (DeepLearningShogi)
├── docker/
│   ├── Dockerfile          # CUDA 12.1 + PyTorch 2.3 + dlshogi
│   ├── docker-compose.yml  # コンテナ設定
│   └── inference_test.py   # 推論動作確認スクリプト
└── models/                 # 学習済みモデルを置くディレクトリ
```

---

## 推論テスト単体実行

UI を起動せずに dlshogi の推論だけ確認したい場合:

```bash
cd docker
docker compose run --rm dlshogi python inference_test.py
```

期待される出力:
```
Using device: cuda
Policy output shape : torch.Size([1, 2187])
Value output        : 0.XXXX
dlshogi inference OK!
```
