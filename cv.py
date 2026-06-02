from typing import Optional, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold


def resolve_n_splits(groups: np.ndarray, config_n_splits: int = 3, max_splits: int = 5) -> int:
    unique = np.unique(groups)
    return min(len(unique), max_splits) if len(unique) >= 2 else config_n_splits


def _normalize_origin_label(x) -> str:
    s = str(x).strip().lower()
    s = s.replace("á", "a").replace("ã", "a").replace("â", "a")
    s = s.replace("é", "e").replace("ê", "e")
    s = s.replace("í", "i")
    s = s.replace("ó", "o").replace("õ", "o").replace("ô", "o")
    s = s.replace("ú", "u")
    s = s.replace("-", "_").replace(" ", "_")
    return s


def infer_real_and_synth_masks(
    origin_values: Optional[np.ndarray],
    logger=None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if origin_values is None:
        return None, None

    norm = np.array([_normalize_origin_label(v) for v in origin_values], dtype=object)

    synth_tokens = {"sintetico", "sintetica", "synthetic", "synth", "augmented", "artificial"}
    real_tokens = {"experimental", "experimento", "real", "medido", "observado", "exp"}

    synth_mask = np.array([any(tok in s for tok in synth_tokens) for s in norm], dtype=bool)
    real_mask = np.array([any(tok in s for tok in real_tokens) for s in norm], dtype=bool)

    unresolved = ~(synth_mask | real_mask)
    if np.any(unresolved):
        real_mask[unresolved] = True

    if logger is not None:
        logger.info(f"Origin split | real={int(real_mask.sum())} | synthetic={int(synth_mask.sum())}")

    return real_mask, synth_mask


class AugmentedTrainRealTestGroupKFold:
    def __init__(self, n_splits: int, origin_values: np.ndarray, logger=None):
        self.n_splits = int(n_splits)
        self.origin_values = np.asarray(origin_values)
        self.logger = logger

        real_mask, synth_mask = infer_real_and_synth_masks(self.origin_values, logger=logger)
        if real_mask is None or synth_mask is None:
            raise ValueError("origin_values is required for AugmentedTrainRealTestGroupKFold.")

        self.real_mask_ = real_mask
        self.synth_mask_ = synth_mask
        self.real_idx_ = np.flatnonzero(real_mask)
        self.synth_idx_ = np.flatnonzero(synth_mask)

        if len(self.real_idx_) == 0:
            raise ValueError("No real/experimental rows found in Origem_dados.")

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("AugmentedTrainRealTestGroupKFold requires groups.")

        groups = np.asarray(groups)
        real_idx = self.real_idx_
        synth_idx = self.synth_idx_

        gkf = GroupKFold(n_splits=self.n_splits)

        X_real_dummy = np.zeros((len(real_idx), 1), dtype=float)
        y_real_dummy = np.zeros(len(real_idx), dtype=float)
        g_real = groups[real_idx]

        for tr_real_pos, te_real_pos in gkf.split(X_real_dummy, y_real_dummy, groups=g_real):
            tr_idx = np.concatenate([real_idx[tr_real_pos], synth_idx], axis=0)
            te_idx = real_idx[te_real_pos]

            tr_idx = np.unique(tr_idx)
            te_idx = np.unique(te_idx)

            yield tr_idx, te_idx


def make_cv(n_splits: int, origin_values: Optional[np.ndarray] = None, logger=None):
    if origin_values is not None:
        real_mask, synth_mask = infer_real_and_synth_masks(origin_values, logger=logger)
        if real_mask is not None and synth_mask is not None and np.any(real_mask) and np.any(synth_mask):
            if logger:
                logger.info("CV rule: TEST = only real rows | TRAIN = real-train + all synthetic rows")
            return AugmentedTrainRealTestGroupKFold(
                n_splits=n_splits, origin_values=origin_values, logger=logger
            ), real_mask

    if logger:
        logger.info("Using standard GroupKFold.")
    return GroupKFold(n_splits=n_splits), np.ones(1, dtype=bool)
