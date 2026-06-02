# Hybrid V-AGMD Modeling Framework

Train and compare **hybrid models** (neural networks augmented with physics-based 0D prior) for Vacuum-Assisted Air Gap Membrane Distillation (V-AGMD) processes.

The framework implements a multi-stage selection pipeline:
- **Stage 1**: Search for the best baseline KerasMLP architecture
- **Stage 2**: Compare hybrid strategies built on top of the Stage 1 winner

## Project Structure

```
├── config.yaml              # User-editable configuration
├── config.py                # Python config (reads config.yaml, fallback to defaults)
├── data.py                  # CSV loading and preprocessing
├── cv.py                    # Cross-validation helpers (GroupKFold, n_splits resolution)
├── runner.py                # Core pipeline: FamilySpec, FamilyWinner, run_family
├── selection.py             # Grid/Randomized search, refit rules, OOF evaluation
├── sweep.py                 # Post-hoc parameter sensitivity sweeps
├── plots.py                 # All plotting functions
├── file_io.py               # Output saving (CSVs, plots, overlays)
├── consolidate.py           # Cross-stage results consolidation
│
├── models/
│   ├── keras_builders.py    # Keras model architectures (MLP, Luc, ZohanResidual, HRNN)
│   ├── wrappers.py          # YScalerRegressor (optional Y scaling wrapper)
│   └── classical.py         # Classical ML models (Ridge, Lasso, RF, etc.)
│
└── scripts/
    ├── stage1_focused.py    # Stage 1: baseline search
    ├── stage2_focused.py    # Stage 2: hybrid comparison
    ├── regenerate_plots.py  # Rebuild plots from saved results (no retrain)
    ├── run_all.py           # Full pipeline (all stages)
    ├── run_pipeline.py      # Alternative full pipeline
    └── ...                  # Additional utilities
```

## Setup

### Prerequisites
- Python 3.10+
- TensorFlow 2.x
- scikit-learn, scikeras, numpy, pandas, matplotlib, pyyaml

### Environment
```powershell
conda create -n agmd python=3.10
conda activate agmd
pip install tensorflow scikeras scikit-learn pandas numpy matplotlib PyYAML
```

Or use your existing environment:
```powershell
conda activate r-tf
pip install pyyaml
```

## Configuration

All project settings are in **`config.yaml`**:

```yaml
data:
  path: "dados_att_com_var_com_phy.csv"   # Path to your CSV
  decimal: ","                              # Decimal separator
  separator: ","                            # Column separator
  group_col: "Regime"                       # Group column for CV (optional)
  origin_col: "Origem_dados"                # Origin column (real vs synthetic)

columns:
  features: [Alim_T_in, Ref_T_in, ...]      # Input features
  targets: [Alim_T_out, Ref_T_out, Flux]    # Target variables
  model_map:                                 # Physics model output columns
    Alim_T_out: Alim_T_out_phy
    Ref_T_out: Ref_T_out_phy
    Flux: Flux_phy_L_m2_h

training:
  seed: 42
  n_splits: 3
  early_stop_patience: 12
  reduce_lr_patience: 4
  ...
```

> **Note:** If `config.yaml` is absent, `config.py` falls back to hardcoded defaults.

### Column Mapping for New Problems

To adapt the framework to a different dataset:

1. Set `features` → the model input columns
2. Set `targets` → the output variables to predict
3. Set `model_map` → which CSV columns contain the physics model predictions for each target
4. Set `group_col` → a categorical column whose unique values determine CV splits

### Group Column & CV

If `group_col` is specified, the number of CV splits equals `min(n_unique_groups, 5)`. This ensures each group appears in at least one fold. If `group_col` is empty/null, the fixed `n_splits` from config is used.

## Data Format

Your CSV must contain:

| Column | Description |
|---|---|
| Feature columns | Model inputs (e.g., temperatures, flow rates) |
| Target columns | Outputs to predict (e.g., temperatures, flux) |
| Physics columns | 0D model predictions for each target (prefix from `model_map`) |
| Group column | (Optional) Categorical column for grouped CV |
| Origin column | (Optional) "real" vs "synthetic" labels for augmented CV |

> **Decimal separator:** Configure `decimal: ","` if your CSV uses commas as decimal separators.

## Running the Pipeline

### Stage 1 — Baseline Search
```powershell
conda activate agmd
python scripts/stage1_focused.py
```
Searches KerasMLP architectures (hidden layers, learning rate, L2, activation, optimizer, batch size) using random search (80 iterations). Output is saved to `stage1_only_<dataset>_<timestamp>/`.

### Stage 2 — Hybrid Comparison
```powershell
python scripts/stage2_focused.py
```
Compares 4 hybrid strategies, each with 5 random search iterations:
- **FrozenBaseline**: plain MLP (no physics)
- **ZohanResidual**: residual connection around physics prior
- **ZohanHPD**: physics-informed hidden layer concatenation
- **Luc**: physics-informed loss function

Output goes to `stage2_only_<dataset>_<timestamp>/stage2_restricted_hybrid_comparison/`.

### Full Pipeline
```powershell
python scripts/run_all.py         # Runs all stages sequentially
python scripts/run_pipeline.py    # Alternative full pipeline entry
```

## Outputs

Each stage creates a timestamped directory containing:

### CSV Files
| File | Contents |
|---|---|
| `*_summary.csv` | Consolidated results across all families |
| `*_cv_results_<family>.csv` | Full grid/random search CV results |
| `*_family_table_<family>.csv` | Detailed family comparison table |
| `*_overlay_points_<family>.csv` | OOF predictions vs physics model for each sample |
| `*_sweep_<family>.csv` | Post-hoc parameter sensitivity sweep |
| `*_best_params.json` | Best hyperparameters per family |

### Plots (PNG)
| Plot | Error bars |
|---|---|
| `*_selection_plot_<family>.png` | Mean CV score vs complexity (± std) |
| `*_oof_pred_vs_true_<family>_<target>.png` | Predicted vs experimental (±RMSE band, vertical experimental error bars) |
| `*_overlay_<family>_<target>.png` | 0D model vs hybrid scatter (±RMSE band, experimental error bars) |
| `*_overlay_panel_<family>.png` | Panel of 3 targets (0D vs hybrid) |
| `*_error_vs_sweepparam_<family>.png` | RMSE vs sweep parameter (± std) |
| `*_gap_vs_sweepparam_<family>.png` | Train-test gap vs sweep parameter |
| `*_learning_curve_<family>.png` | Train/val loss over epochs |
| `*_winners_rmse_by_target.png` | RMSE per target per family (bar chart) |

### Experimental Error Bars

Scatter plots include vertical error bars representing experimental measurement uncertainty:
- **Flux**: ±10% of the measured value
- **Temperatures**: ±1 °C

These values are defined in the `_exp_yerr()` helper in `plots.py` and can be adjusted for your application.

## Regenerating Plots (Without Retraining)

If you only want to update the plots (e.g., after changing plot styles), the models do not need to be retrained:

```powershell
python scripts/regenerate_plots.py <path_to_stage2_comparison_dir>
```

Example:
```powershell
python scripts/regenerate_plots.py stage2_only_dados_20260602_012037/stage2_restricted_hybrid_comparison/
```

The script reads `*_overlay_points_<family>.csv` and `*_summary.csv` from the output directory to reconstruct predictions and regenerate all scatter plots, overlay panels, and the winners RMSE bar chart.

> **Note:** Selection plots and sweep plots require the original grid search results and fitted models, so those are not regenerated by this script.

## Sweep Analysis vs Grid Search

The **sweep** is a post-hoc analysis that fixes all hyperparameters except one (e.g., L2 regularization) and re-evaluates the model. This shows the isolated sensitivity to a single parameter.

The **grid/random search** varies multiple hyperparameters simultaneously. The final selection (1-SE + min complexity rule) considers the coupled effect of all parameters, so the best sweep value may differ from the selected value.

## Selection Rule

The refit rule (in `selection.py`) follows the **1-SE + minimum complexity** heuristic:
1. Find the candidate with the best CV score
2. Build a 1-standard-error band around it
3. Among candidates within that band, pick the one with minimum complexity
4. Break ties by best test score

This is a standard regularization technique that prefers simpler models whose performance is statistically indistinguishable from the best.

## Customization

### Adding a New Hybrid Architecture
1. Implement the Keras model builder in `models/keras_builders.py`
2. Add a grid builder function in `sweep.py`
3. Add a `FamilySpec` entry in `scripts/stage2_focused.py`

### Changing Error Bar Magnitudes
Edit the `_exp_yerr()` function in `plots.py`:
```python
def _exp_yerr(yt, target_name):
    if "Flux" in target_name:
        return 0.10 * np.abs(yt)   # 10%
    return np.full_like(yt, 1.0)    # 1 °C
```

## License

Academic use. For questions, contact the repository owner.
