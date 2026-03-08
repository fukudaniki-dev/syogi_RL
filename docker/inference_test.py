import torch
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi import common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ResNet10+Swish モデルを初期化（ランダム重み）
model = policy_value_network('resnet10_swish')
model.to(device)
model.eval()

# 将棋盤の入力特徴量（バッチサイズ=1、ゼロテンソル）
features1 = torch.zeros(1, common.FEATURES1_NUM, 9, 9, device=device)
features2 = torch.zeros(1, common.FEATURES2_NUM, 9, 9, device=device)

# 推論
with torch.no_grad():
    policy, value = model(features1, features2)

print(f"Policy output shape : {policy.shape}")   # -> (1, 2187)
print(f"Value output        : {value.item():.4f}")
print("dlshogi inference OK!")
