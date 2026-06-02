from typing import Any, Tuple
import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasRegressor

KERAS_AVAILABLE = True
try:
    import tensorflow as tf
    from scikeras.wrappers import KerasRegressor
except Exception:
    KERAS_AVAILABLE = False


def _infer_n_features_targets(
    *,
    n_features_in_=None,
    n_outputs_=None,
    n_features=None,
    n_targets=None,
    **kwargs,
) -> Tuple[int, int]:
    nf = n_features_in_ if n_features_in_ is not None else n_features
    nt = n_outputs_ if n_outputs_ is not None else n_targets
    if nf is None:
        raise TypeError("Could not infer n_features. Fix: set 'model__model__n_features'.")
    if nt is None:
        raise TypeError("Could not infer n_targets/n_outputs. Fix: set 'model__model__n_targets'.")
    return int(nf), int(nt)


def _make_optimizer(optimizer: str, learning_rate: float, clipnorm=1.0):
    lr = float(learning_rate)
    opt = str(optimizer).lower()
    kwargs = {}
    if clipnorm is not None:
        kwargs["clipnorm"] = float(clipnorm)
    if opt == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, **kwargs)
    if opt == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, **kwargs)
    if opt == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, **kwargs)
    return tf.keras.optimizers.Adam(learning_rate=lr, **kwargs)


@tf.keras.utils.register_keras_serializable(package="AGMD")
class SliceXPart(tf.keras.layers.Layer):
    def __init__(self, nt: int, **kwargs):
        super().__init__(**kwargs)
        self.nt = int(nt)

    def call(self, inputs):
        return inputs[:, :-self.nt]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"nt": self.nt})
        return cfg


@tf.keras.utils.register_keras_serializable(package="AGMD")
class SliceYPhyPart(tf.keras.layers.Layer):
    def __init__(self, nt: int, **kwargs):
        super().__init__(**kwargs)
        self.nt = int(nt)

    def call(self, inputs):
        return inputs[:, -self.nt:]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"nt": self.nt})
        return cfg


@tf.keras.utils.register_keras_serializable(package="AGMD")
class LucHybridLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        m: int,
        data_loss: str = "mse",
        huber_delta: float = 1.0,
        omega: float = 0.5,
        physics_norm: str = "mse",
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "luc_hybrid_loss",
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.m = int(m)
        self.data_loss = str(data_loss).lower()
        self.huber_delta = float(huber_delta)
        self.omega = float(omega)
        self.physics_norm = str(physics_norm).lower()
        self._huber = tf.keras.losses.Huber(delta=self.huber_delta, reduction=tf.keras.losses.Reduction.NONE)

    def _data_term(self, y_true, y_pred):
        if self.data_loss == "mae":
            return tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
        if self.data_loss == "log_cosh":
            return tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true + 1e-12)), axis=1)
        if self.data_loss == "huber":
            return self._huber(y_true, y_pred)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    def _phys_term(self, y_phy, y_pred):
        if self.physics_norm == "mae":
            return tf.reduce_mean(tf.abs(y_phy - y_pred), axis=1)
        return tf.reduce_mean(tf.square(y_phy - y_pred), axis=1)

    def call(self, y_true_aug, y_pred_2m):
        m = self.m
        y_true = y_true_aug[:, :m]
        y_phy = y_true_aug[:, m:(2 * m)]
        y_pred = y_pred_2m[:, :m]

        l_data = self._data_term(y_true, y_pred)
        l_phys = self._phys_term(y_phy, y_pred)

        w = tf.clip_by_value(tf.cast(self.omega, tf.float32), 0.0, 1.0)
        loss_vec = (1.0 - w) * l_data + w * l_phys
        return tf.reduce_mean(loss_vec)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "m": self.m,
            "data_loss": self.data_loss,
            "huber_delta": self.huber_delta,
            "omega": self.omega,
            "physics_norm": self.physics_norm,
        })
        return cfg


def build_keras_mlp(
    *,
    n_features_in_=None, n_outputs_=None, n_features=None, n_targets=None,
    hidden_layer_sizes=(64, 32), activation="relu",
    optimizer="adam", learning_rate=1e-3, loss="mse", huber_delta=1.0, l2=0.0,
    clipnorm=1.0,
    **kwargs,
) -> tf.keras.Model:
    nf, nt = _infer_n_features_targets(
        n_features_in_=n_features_in_, n_outputs_=n_outputs_,
        n_features=n_features, n_targets=n_targets, **kwargs,
    )

    opt = _make_optimizer(optimizer, learning_rate, clipnorm=clipnorm)
    reg = tf.keras.regularizers.l2(float(l2)) if (l2 is not None and float(l2) > 0) else None

    if isinstance(loss, str) and loss.lower() == "huber":
        loss_obj = tf.keras.losses.Huber(delta=float(huber_delta))
    else:
        loss_obj = loss

    inputs = tf.keras.Input(shape=(nf,))
    x = inputs
    for u in hidden_layer_sizes:
        x = tf.keras.layers.Dense(int(u), activation=str(activation), kernel_regularizer=reg)(x)
    outputs = tf.keras.layers.Dense(int(nt), activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt, loss=loss_obj)
    return model


def build_keras_mlp_luc_loss_2m_output(
    *,
    n_features_in_=None, n_features=None, n_targets=None,
    hidden_layer_sizes=(64, 32), activation="relu",
    optimizer="adam", learning_rate=1e-3,
    data_loss="mse", huber_delta=1.0, physics_norm="mse", omega=0.5, l2=0.0,
    **kwargs,
) -> tf.keras.Model:
    nf, _ = _infer_n_features_targets(
        n_features_in_=n_features_in_, n_outputs_=None,
        n_features=n_features, n_targets=n_targets, **kwargs,
    )

    m = int(n_targets) if n_targets is not None else None
    if m is None:
        raise TypeError("Luc loss requires explicit 'model__model__n_targets'.")

    opt = _make_optimizer(optimizer, learning_rate)
    reg = tf.keras.regularizers.l2(float(l2)) if (l2 is not None and float(l2) > 0) else None

    inputs = tf.keras.Input(shape=(nf,))
    x = inputs
    for u in hidden_layer_sizes:
        x = tf.keras.layers.Dense(int(u), activation=str(activation), kernel_regularizer=reg)(x)

    y_hat = tf.keras.layers.Dense(int(m), activation="linear", name="y_hat")(x)
    y_out = tf.keras.layers.Concatenate(name="y_out_2m")([y_hat, y_hat])

    loss_obj = LucHybridLoss(m=m, data_loss=data_loss, huber_delta=huber_delta, omega=omega, physics_norm=physics_norm)

    model = tf.keras.Model(inputs=inputs, outputs=y_out)
    model.compile(optimizer=opt, loss=loss_obj)
    return model


def build_keras_mlp_residual_add_phy_x_only(
    *,
    n_features_in_=None, n_outputs_=None, n_features=None, n_targets=None,
    hidden_layer_sizes=(64, 32), activation="relu",
    optimizer="adam", learning_rate=1e-3, loss="mse", huber_delta=1.0, l2=0.0,
    **kwargs,
) -> tf.keras.Model:
    nf, nt = _infer_n_features_targets(
        n_features_in_=n_features_in_, n_outputs_=n_outputs_,
        n_features=n_features, n_targets=n_targets, **kwargs,
    )

    if int(nf) <= int(nt):
        raise ValueError("Residual expects n_features_total > n_targets (X + y_phy).")

    opt = _make_optimizer(optimizer, learning_rate)
    reg = tf.keras.regularizers.l2(float(l2)) if (l2 is not None and float(l2) > 0) else None
    loss_obj = tf.keras.losses.Huber(delta=float(huber_delta)) if (isinstance(loss, str) and loss.lower() == "huber") else loss

    inputs = tf.keras.Input(shape=(nf,))
    x_part = SliceXPart(nt, name="X_part")(inputs)
    y_phy = SliceYPhyPart(nt, name="Yphy_part")(inputs)

    h = x_part
    for u in hidden_layer_sizes:
        h = tf.keras.layers.Dense(int(u), activation=str(activation), kernel_regularizer=reg)(h)

    y_res = tf.keras.layers.Dense(int(nt), activation="linear", name="Yres")(h)
    y_hat = tf.keras.layers.Add(name="Yhat")([y_phy, y_res])

    model = tf.keras.Model(inputs=inputs, outputs=y_hat)
    model.compile(optimizer=opt, loss=loss_obj)
    return model


def build_keras_mlp_hrnn_residual_xyphy(
    *,
    n_features_in_=None, n_outputs_=None, n_features=None, n_targets=None,
    hidden_layer_sizes=(64, 32), activation="relu",
    optimizer="adam", learning_rate=1e-3, loss="mse", huber_delta=1.0, l2=0.0,
    **kwargs,
) -> tf.keras.Model:
    nf, nt = _infer_n_features_targets(
        n_features_in_=n_features_in_, n_outputs_=n_outputs_,
        n_features=n_features, n_targets=n_targets, **kwargs,
    )

    if int(nf) <= int(nt):
        raise ValueError("HRNN expects n_features_total > n_targets (X + y_phy).")

    opt = _make_optimizer(optimizer, learning_rate)
    reg = tf.keras.regularizers.l2(float(l2)) if (l2 is not None and float(l2) > 0) else None
    loss_obj = tf.keras.losses.Huber(delta=float(huber_delta)) if (isinstance(loss, str) and loss.lower() == "huber") else loss

    inputs = tf.keras.Input(shape=(nf,))
    y_phy = SliceYPhyPart(nt, name="Yphy_part")(inputs)

    h = inputs
    for u in hidden_layer_sizes:
        h = tf.keras.layers.Dense(int(u), activation=str(activation), kernel_regularizer=reg)(h)

    y_res = tf.keras.layers.Dense(int(nt), activation="linear", name="Yres")(h)
    y_hat = tf.keras.layers.Add(name="Yhat")([y_phy, y_res])

    model = tf.keras.Model(inputs=inputs, outputs=y_hat)
    model.compile(optimizer=opt, loss=loss_obj)
    return model


def make_keras_mlp_estimator(n_targets: int):
    return KerasRegressor(model=build_keras_mlp, verbose=0)


def make_keras_mlp_luc_estimator(n_targets: int):
    return KerasRegressor(model=build_keras_mlp_luc_loss_2m_output, verbose=0)


def make_keras_mlp_residual_x_only_estimator(n_targets: int):
    return KerasRegressor(model=build_keras_mlp_residual_add_phy_x_only, verbose=0)


def make_keras_mlp_hrnn_estimator(n_targets: int):
    return KerasRegressor(model=build_keras_mlp_hrnn_residual_xyphy, verbose=0)
