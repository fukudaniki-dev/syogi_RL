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

        self.model = policy_value_network("resnet10_swish")
        self.model.to(self.device)

        model_path = Path(_MODEL_PATH)
        if model_path.is_file():
            logger.info("Loading model weights from %s", model_path)
            state_dict = torch.load(model_path, map_location=self.device)
            # チェックポイント形式への対応
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                "Model file not found (%s). Using random weights.", model_path
            )

        self.model.eval()

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
