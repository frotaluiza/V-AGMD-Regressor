"""Per-output MLP baseline otimizado para cada saida."""
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PerOutputMLP:
    """
    Per-output MLP baseline: um MLPRegressor independente para cada target.

    Configuracoes otimas encontradas pelo agente:
      - Flux: keras [256] tanh l2=1e-5 lr=0.005 bs=174 (via Keras, nao sklearn)
      - Alim_T_out: logistic alpha=0.1 (256,) lr=0.001
      - Ref_T_out: logistic alpha=0.1 (256,) lr=0.005
    """

    def __init__(self, flux_config=None, alim_config=None, ref_config=None):
        self.flux_config = flux_config or {
            "hidden_layer_sizes": (256,),
            "activation": "tanh",
            "alpha": 1e-5,
            "learning_rate_init": 0.005,
            "max_iter": 5000,
            "early_stopping": True,
            "batch_size": 174,
        }
        self.alim_config = alim_config or {
            "hidden_layer_sizes": (256,),
            "activation": "logistic",
            "alpha": 0.1,
            "learning_rate_init": 0.001,
            "max_iter": 5000,
            "early_stopping": True,
        }
        self.ref_config = ref_config or {
            "hidden_layer_sizes": (256,),
            "activation": "logistic",
            "alpha": 0.1,
            "learning_rate_init": 0.005,
            "max_iter": 5000,
            "early_stopping": True,
        }
        self.models = {}
        self.scalers = {}

    def fit(self, X, Y_true):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        names = ["Alim_T_out", "Ref_T_out", "Flux"]
        configs = [self.alim_config, self.ref_config, self.flux_config]
        for i, (name, cfg) in enumerate(zip(names, configs)):
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            m = MLPRegressor(**cfg, random_state=42)
            m.fit(Xs, Y_true[:, i])
            self.models[name] = m
            self.scalers[name] = scaler
        return self

    def predict(self, X):
        import numpy as np
        Y = np.zeros((len(X), 3))
        names = ["Alim_T_out", "Ref_T_out", "Flux"]
        for i, name in enumerate(names):
            Xs = self.scalers[name].transform(X)
            Y[:, i] = self.models[name].predict(Xs)
        return Y
