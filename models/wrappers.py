from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class YScalerRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        base_estimator: Any,
        scale_y: bool = True,
        augmented_2m: bool = False,
        m_targets: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.scale_y = bool(scale_y)
        self.augmented_2m = bool(augmented_2m)
        self.m_targets = m_targets

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            "base_estimator": self.base_estimator,
            "scale_y": self.scale_y,
            "augmented_2m": self.augmented_2m,
            "m_targets": self.m_targets,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            inner = self.base_estimator.get_params(deep=True)
            for k, v in inner.items():
                if k in params:
                    params[f"base_estimator__{k}"] = v
                else:
                    params[k] = v
        return params

    def set_params(self, **params: Any):
        wrapper_keys = {"base_estimator", "scale_y", "augmented_2m", "m_targets"}
        inner_params = {}

        for k, v in params.items():
            if k in wrapper_keys:
                setattr(self, k, v)
            elif k.startswith("base_estimator__"):
                inner_params[k.replace("base_estimator__", "", 1)] = v
            else:
                inner_params[k] = v

        if inner_params:
            if not hasattr(self.base_estimator, "set_params"):
                raise ValueError("base_estimator does not support set_params, but params were provided.")
            self.base_estimator.set_params(**inner_params)
        return self

    def _split_aug(self, Y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if not self.augmented_2m:
            return Y, None
        if self.m_targets is None:
            raise ValueError("augmented_2m=True requires m_targets.")
        m = int(self.m_targets)
        if Y.shape[1] != 2 * m:
            raise ValueError(f"Expected Y with 2m columns={2 * m}, got {Y.shape[1]}.")
        return Y[:, :m], Y[:, m:]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._y_scaler = None
        self._m_ = None

        if not self.scale_y:
            self.base_estimator_ = self.base_estimator
            self.base_estimator_.fit(X, y)
            return self

        y_true, y_phy = self._split_aug(y)
        self._m_ = y_true.shape[1]
        self._y_scaler = StandardScaler()
        y_true_s = self._y_scaler.fit_transform(y_true)

        if y_phy is not None:
            y_phy_s = self._y_scaler.transform(y_phy)
            y_s = np.concatenate([y_true_s, y_phy_s], axis=1)
        else:
            y_s = y_true_s

        self.base_estimator_ = self.base_estimator
        self.base_estimator_.fit(X, y_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.base_estimator_.predict(X)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if not self.scale_y:
            return y_pred

        if self._y_scaler is None:
            raise RuntimeError("YScalerRegressor used with scale_y=True but scaler is not fitted.")

        m = int(self._m_ or y_pred.shape[1])
        if not self.augmented_2m:
            return self._y_scaler.inverse_transform(y_pred)

        if y_pred.shape[1] == m:
            return self._y_scaler.inverse_transform(y_pred)

        if y_pred.shape[1] == 2 * m:
            a = self._y_scaler.inverse_transform(y_pred[:, :m])
            b = self._y_scaler.inverse_transform(y_pred[:, m:])
            return np.concatenate([a, b], axis=1)

        return self._y_scaler.inverse_transform(y_pred[:, :m])
