from pathlib import Path
import os

_CONFIG_YAML = None
_yaml_path = Path(__file__).resolve().parent / "config.yaml"
if _yaml_path.exists():
    try:
        import yaml
        with open(_yaml_path, "r", encoding="utf-8") as f:
            _CONFIG_YAML = yaml.safe_load(f)
    except Exception:
        _CONFIG_YAML = None

if _CONFIG_YAML is not None:
    c = _CONFIG_YAML
    FEATURE_COLS = c["columns"]["features"]
    TARGET_COLS = c["columns"]["targets"]
    MODEL_MAP = c["columns"]["model_map"]
    GROUP_COL = c["data"].get("group_col")
    ORIGIN_COL = c["data"].get("origin_col")
    SEED = c["training"]["seed"]
    N_SPLITS = c["training"]["n_splits"]
    KERAS_CALLBACKS = dict(
        early_stop_patience=c["training"]["early_stop_patience"],
        early_stop_min_delta=c["training"]["early_stop_min_delta"],
        reduce_lr_factor=c["training"]["reduce_lr_factor"],
        reduce_lr_patience=c["training"]["reduce_lr_patience"],
        reduce_lr_min_lr=c["training"]["reduce_lr_min_lr"],
    )
    DATA_PATH = Path(c["data"]["path"])
else:
    FEATURE_COLS = ["Alim_T_in", "Ref_T_in", "Ref_V_in", "P_vacuum", "C_NaCl"]
    TARGET_COLS = ["Alim_T_out", "Ref_T_out", "Flux"]
    GROUP_COL = "Regime"
    ORIGIN_COL = "Origem_dados"
    SEED = 42
    N_SPLITS = 3
    MODEL_MAP = {
        "Alim_T_out": "Alim_T_out_phy",
        "Ref_T_out": "Ref_T_out_phy",
        "Flux": "Flux_phy_L_m2_h",
    }
    BASE_DIR = Path(r"C:\Users\frota\OneDrive\Documentos\TCC")
    DATA_PATH = BASE_DIR / "dados_att_com_var_com_phy.csv"
    KERAS_CALLBACKS = dict(
        early_stop_patience=12,
        early_stop_min_delta=1e-4,
        reduce_lr_factor=0.5,
        reduce_lr_patience=4,
        reduce_lr_min_lr=1e-6,
    )

DATASET_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

N_FEATURES_BASE = len(FEATURE_COLS)
N_TARGETS = len(TARGET_COLS)
N_FEATURES_PLUS_PHY = N_FEATURES_BASE + N_TARGETS
FLUX_INDEX = TARGET_COLS.index("Flux")

DEFAULT_GAP_FILTER_TOL = 0.02
DEFAULT_USE_GAP_FILTER = False
DEFAULT_USE_GAP_TIEBREAK = False
