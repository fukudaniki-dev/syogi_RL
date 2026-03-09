"""
dlshogi モデルをシングルトンでロードし、推論を提供するモジュール。
モデルファイルが存在しない場合はランダム重みにフォールバックする。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MODEL_PATH = os.environ.get("DLSHOGI_MODEL_PATH", "")
_NETWORK = os.environ.get("DLSHOGI_NETWORK", "resnet10_swish")


class DlshogiInference:
    """dlshogi policy-value ネットワークのシングルトンラッパー。"""

    _instance: "DlshogiInference | None" = None

    def __new__(cls) -> "DlshogiInference":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)
        self._load_model()

    def _load_model(self) -> None:
        from dlshogi.network.policy_value_network import policy_value_network
        from dlshogi import common  # noqa: F401  (FEATURES*_NUM を確認するため)

        logger.info("Loading network architecture: %s", _NETWORK)
        self.model = policy_value_network(_NETWORK)
        self.model.to(self.device)

        model_path = Path(_MODEL_PATH)
        if model_path.is_file():
            logger.info("Loading model weights from %s", model_path)
            state_dict = self._load_state_dict(model_path)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                "Model file not found (%s). Using random weights.", model_path
            )

        self.model.eval()

    def _load_state_dict(self, model_path: Path) -> dict:
        """
        PyTorch の新旧両フォーマット、および Chainer 形式（zip内 .npy）に対応したモデルロード。
        """
        import zipfile
        import numpy as np

        # Chainer 形式の判定: zip 内に .npy ファイルを含む
        try:
            with zipfile.ZipFile(model_path) as zf:
                names = zf.namelist()
                if any(n.endswith(".npy") for n in names):
                    logger.info("Detected Chainer format. Converting to PyTorch state_dict...")
                    return self._convert_chainer_to_state_dict(zf, names)
        except zipfile.BadZipFile:
            pass

        # 通常の PyTorch 形式
        return torch.load(model_path, map_location=self.device, weights_only=False)

    def _convert_chainer_to_state_dict(self, zf, names) -> dict:
        """
        Chainer 形式（zip 内 layer/W.npy, layer/b.npy, norm/gamma.npy …）を
        PyTorch state_dict に変換する。

        変換ルール:
          lX/W.npy        → lX.weight
          lX/b.npy        → lX.bias
          normX/gamma.npy → normX.weight  (BatchNorm γ)
          normX/beta.npy  → normX.bias    (BatchNorm β)
          normX/avg_mean.npy → normX.running_mean
          normX/avg_var.npy  → normX.running_var
          normX/N.npy     → (スキップ)
        """
        import numpy as np

        state_dict = {}
        for name in names:
            parts = name.split("/")
            if len(parts) != 2:
                continue
            layer, param_file = parts
            param_name = param_file.replace(".npy", "")

            key_map = {
                "W": "weight",
                "b": "bias",
                "gamma": "weight",
                "beta": "bias",
                "avg_mean": "running_mean",
                "avg_var": "running_var",
            }
            pt_param = key_map.get(param_name)
            if pt_param is None:
                continue  # N.npy などはスキップ

            pt_key = f"{layer}.{pt_param}"
            with zf.open(name) as f:
                arr = np.load(f)
            state_dict[pt_key] = torch.from_numpy(arr.copy()).to(self.device)

        logger.info("Converted %d parameters from Chainer format.", len(state_dict))
        return state_dict

    def infer(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """
        推論を実行する。

        Parameters
        ----------
        features1 : torch.Tensor  shape (1, FEATURES1_NUM, 9, 9)
        features2 : torch.Tensor  shape (1, FEATURES2_NUM, 9, 9)

        Returns
        -------
        policy : np.ndarray  shape (2187,)  softmax 済み確率
        value  : float       局面評価値 (-1 〜 1)
        """
        features1 = features1.to(self.device)
        features2 = features2.to(self.device)

        with torch.no_grad():
            policy_logits, value_tensor = self.model(features1, features2)

        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        value = float(value_tensor[0].item())
        return policy, value
