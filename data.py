from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np


def read_tabular_csv(
    csv_path: str,
    decimal_comma: bool = True,
    sep: str = ",",
    encoding: Optional[str] = None,
    logger=None,
) -> pd.DataFrame:
    if logger:
        logger.info(f"Reading CSV: {csv_path}")
        logger.info(f"  sep='{sep}' | decimal_comma={decimal_comma}")

    df = pd.read_csv(csv_path, sep=sep, encoding=encoding)

    if decimal_comma:
        str_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype) or df[c].dtype == object]
        if logger:
            logger.info(f"  String/Object columns to attempt numeric conversion: {len(str_cols)}")
        for c in str_cols:
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            converted = pd.to_numeric(s, errors="coerce")
            if converted.notna().any():
                df[c] = converted

    return df


def build_XY_groups_with_model_map(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    group_col: str,
    dropna: bool = True,
    model_map: Optional[Dict[str, str]] = None,
    origin_col: Optional[str] = None,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    needed = list(feature_cols) + list(target_cols) + [group_col]

    if model_map is not None:
        for t in target_cols:
            if t not in model_map:
                raise KeyError(f"model_map missing mapping for target '{t}'.")
        needed += [model_map[t] for t in target_cols]

    if origin_col is not None:
        needed += [origin_col]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    used = df[needed].copy()

    if dropna:
        before = len(used)
        used = used.dropna(axis=0, how="any")
        after = len(used)
        if logger:
            logger.info(f"Dropped NA rows: {before - after} (kept {after})")

    X = used[list(feature_cols)].to_numpy(dtype=float)
    Y_true = used[list(target_cols)].to_numpy(dtype=float)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)

    groups = used[group_col].to_numpy()

    Y_model = None
    if model_map is not None:
        model_cols_ordered = [model_map[t] for t in target_cols]
        Y_model = used[model_cols_ordered].to_numpy(dtype=float)
        if Y_model.ndim == 1:
            Y_model = Y_model.reshape(-1, 1)
        if Y_model.shape[1] != Y_true.shape[1]:
            raise ValueError(f"Y_model has {Y_model.shape[1]} columns but Y_true has {Y_true.shape[1]}.")

    origin_values = None
    if origin_col is not None:
        origin_values = used[origin_col].astype(str).to_numpy()

    if logger:
        logger.info(f"X shape: {X.shape} | Y_true shape: {Y_true.shape} | groups: {len(groups)}")
        if Y_model is not None:
            logger.info(f"Y_model shape: {Y_model.shape}")
        if origin_values is not None:
            logger.info(f"Origin column detected: {origin_col}")

    return X, Y_true, groups, used, Y_model, origin_values
