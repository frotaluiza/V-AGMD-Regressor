from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, MultiTaskLasso, MultiTaskElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from config import SEED


def make_ols(n_targets: int):
    return LinearRegression()


def make_ridge(n_targets: int):
    try:
        return Ridge(random_state=SEED)
    except TypeError:
        return Ridge()


def make_lasso_multitask(n_targets: int):
    return MultiTaskLasso(max_iter=20000, random_state=SEED) if int(n_targets) > 1 else Lasso(max_iter=20000, random_state=SEED)


def make_elasticnet_multitask(n_targets: int):
    return MultiTaskElasticNet(max_iter=20000, random_state=SEED) if int(n_targets) > 1 else ElasticNet(max_iter=20000, random_state=SEED)


def make_lasso_indep(n_targets: int):
    return MultiOutputRegressor(Lasso(max_iter=20000, random_state=SEED))


def make_elasticnet_indep(n_targets: int):
    return MultiOutputRegressor(ElasticNet(max_iter=20000, random_state=SEED))


def make_dt(n_targets: int):
    return DecisionTreeRegressor(random_state=SEED)


def make_rf(n_targets: int):
    return RandomForestRegressor(random_state=SEED)


def make_gb(n_targets: int):
    return GradientBoostingRegressor(random_state=SEED)


def make_mlp_sklearn(n_targets: int):
    return MLPRegressor(random_state=SEED, max_iter=20000)
