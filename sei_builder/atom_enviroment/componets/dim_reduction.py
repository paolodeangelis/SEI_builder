import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
from joblib import parallel_backend
from numpy import ndarray
from sklearn.covariance import LedoitWolf, ShrunkCovariance
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score

from ...functions import message

VERBOSE_L = ["ERROR", "WARING", "INFO", "DEBUG"]


class DimensionalityReduction:
    def __init__(self, X=None, fit_all=False, verbose=0):
        self._X = None
        self.Nimage = None
        self.Npoints = None
        self.Ndim = None
        self.X = X
        self._verbose = verbose
        self._Method = None
        self._Optimizer = None
        self._is_optimized = False
        self.fit_all = fit_all

    def __call__(self, X, verbose=0, method=None, fit_all=False, optimizer=None):
        assert isinstance(X, ndarray)
        self.X = X
        if len(self.X.shape) == 2:
            self.X = self.X[np.newaxis, :, :]
        self.Nimage = self.X.shape[0]
        self.Npoints = self.X.shape[1]
        self.Ndim = self.X.shape[2]
        self._verbose = verbose
        if method:
            self.Method = method
        if optimizer:
            self.Optimizer = optimizer
        self._is_optimized = False
        self.fit_all = fit_all

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        if _X is not None:
            assert isinstance(_X, ndarray)
            self._X = _X
            if len(self._X.shape) == 2:
                self._X = self._X[np.newaxis, :, :]
            self.Nimage = self._X.shape[0]
            self.Npoints = self._X.shape[1]
            self.Ndim = self._X.shape[2]

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        # print(f"Vorbase level set to {verbose} ({VERBOSE_L[verbose]})")
        self._verbose = verbose

    @property
    def Method(self):
        return self._Method

    @Method.setter
    def Method(self, method):
        assert callable(method.fit)
        self._is_optimized = False
        self._Method = method
        if self.verbose >= 2:
            message(f"Set {self._Method.__str__()} as Speace Reduction Method", add_date=True)

    def set_method(self, method):
        self.Method = method

    def get_method(self):
        return self.Method

    @property
    def Optimizer(self):
        return self._Optimizer

    @Optimizer.setter
    def Optimizer(self, optimizer):
        if self._Method is None:
            message(
                "Set the Dimensionality Reduction Method before to set the Optimizer\n\te.g.:"
                " DimensionalityReduction.Method = PCA",
                msg_type="error",
                add_date=True,
            )
            raise AttributeError("Set the Dimensionality Reduction Method before to set the Optimizer")
        assert isinstance(optimizer, Oprimizer)
        self._Optimizer = optimizer

    @Optimizer.deleter
    def Optimizer(self):
        self._Optimizer = None

    def set_optimizer(self, optimizer):
        self.Optimizer = optimizer

    def get_optimizer(self):
        return self.Optimizer

    def optimize(self, X):
        if self._Optimizer is not None and self._is_optimized is False:
            self.Method = self.Optimizer(X, method=self._Method, verbose=self._verbose)
            if self._verbose >= 2:
                message(f"{self._Method.__str__()} optimized", add_date=True)
        if self._Optimizer is not None and self._is_optimized is True:
            if self._verbose >= 1:
                message(f"The method {self._Method.__str__()} is already optimized", msg_type="warning", add_date=True)
        if self._Optimizer is None:
            if self._verbose >= 1:
                message(
                    f"No Optimizer for the {self._Method.__str__()}  space reduction method",
                    msg_type="warning",
                    add_date=True,
                )

    def fit(self, X=None):
        if X is None:
            X = self._X

        if self.fit_all:
            Xfit = np.concatenate(X)
        else:
            Xfit = X[0, :, :]
        if self._is_optimized is False:
            self.optimize(Xfit)
        self.Method.fit(Xfit)

    def transform(self, X):
        Xout = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xtemp = self.Method.transform(X[i, :, :])
            Nred = Xtemp.shape[1]
            Xout[i, :, :Nred] = Xtemp
        return Xout[:, :, :Nred]

    def __str__(self):
        _str = "DimensionalityReduction("
        if self._X is not None:
            _str += f"X=array({self._X.shape}), "
        else:
            _str += f"X=None, "
        if self._Method:
            _str += f"method={self._Method.__str__()}, "
        else:
            _str += f"method=None, "
        if self._Optimizer:
            _str += f"optimizer=Yes, "
        else:
            _str += f"optimizer=No, "
        _str += f"verbose={self._verbose})"
        return _str


def _compute_scores(X, n_components, method, n_jobs=1):
    pca = copy.deepcopy(method)  # PCA(svd_solver="full")
    pca_scores = np.ones(n_components.shape) * -1 * np.inf
    for i, n in enumerate(n_components):
        pca.n_components = n
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n_score = cross_val_score(pca, X)
                pca_scores[i] = np.nanmean(n_score)
        except ValueError:
            pca_scores[i] = np.nan
    return pca_scores


def _shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 20)
    cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
    mle = cv.fit(X).best_estimator_
    score = np.mean(cross_val_score(mle, X))
    return score


def _lw_score(X):
    score = np.mean(cross_val_score(LedoitWolf(), X))
    return score


def _pca_n_componets_scanner(X, method, step=8, interval=None, n_jobs=1):
    if interval is None:
        Nd = X.shape[-1]
        n_components = np.arange(2, Nd, step, dtype=int)
    else:
        n_components = np.arange(interval[0], interval[1], step, dtype=int)
    pca_scores = _compute_scores(X, n_components, method, n_jobs=n_jobs)
    best = np.where(pca_scores == np.nanmax(pca_scores))[0]
    interval = [n_components[best] - step, n_components[best] + step]
    return (
        int(n_components[best]),
        n_components,
        float(pca_scores[best]),
        pca_scores,
        interval,
    )


def get_pca_n_componets(X, method, start_step=12, verbose=False, n_jobs=1):
    interval = None
    pca_scores_all = []
    n_components_all = []
    if start_step >= 2:
        # corse
        best_n_componets, n_components, pca_score, pca_scores, interval = _pca_n_componets_scanner(
            X, method, step=start_step, n_jobs=n_jobs
        )
        pca_scores_all = np.append(pca_scores_all, pca_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Corse search: n_components={best_n_componets} (cv score = {pca_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    if start_step >= 4:
        # fine
        best_n_componets, n_components, pca_score, pca_scores, interval = _pca_n_componets_scanner(
            X, method, step=start_step // 2, interval=interval, n_jobs=n_jobs
        )
        pca_scores_all = np.append(pca_scores_all, pca_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Fine search: n_components={best_n_componets} (cv score = {pca_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    if start_step >= 8:
        # finer
        best_n_componets, n_components, pca_score, pca_scores, interval = _pca_n_componets_scanner(
            X, method, step=start_step // 2, interval=interval, n_jobs=n_jobs
        )
        pca_scores_all = np.append(pca_scores_all, pca_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Finer search: n_components={best_n_componets} (cv score = {pca_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    # final
    best_n_componets, n_components, pca_score, pca_scores, intervall = _pca_n_componets_scanner(
        X, method, step=1, interval=interval, n_jobs=n_jobs
    )
    pca_scores_all = np.append(pca_scores_all, pca_scores)
    n_components_all = np.append(n_components_all, n_components)
    if verbose:
        message(
            f"Final search: n_components={best_n_componets} (cv score = {pca_score:1.3e}) ",
            msg_type="debug",
            add_date=True,
        )
    return best_n_componets, pca_scores_all, n_components_all


def _pca_optimizer(X, method=None, verbose=0, start_step=12, n_jobs=1):
    assert isinstance(method, PCA)
    Ndim = X.shape[-1]
    if Ndim < start_step * 2:
        start_step = 2
    _verbose = verbose >= 3
    best_n_componets, pca_scores_all, n_components_all = get_pca_n_componets(
        X, method, start_step=12, verbose=_verbose, n_jobs=n_jobs
    )
    if verbose >= 2:
        message(f"Best n_componets value = {best_n_componets}", add_date=True)
    method.n_components = best_n_componets
    return method


class Oprimizer:
    def __init__(self, X=None, method=None, verbose=0, n_jobs=1):
        self._Method = None
        self._X = None
        self.verbose = verbose
        self.Method = method
        self.X = X
        self.n_jobs = n_jobs
        self._data = None

    def __call__(self, X, method=None, verbose=0, n_jobs=1):
        self._Method = None
        self._X = None
        self.verbose = verbose
        self.Method = method
        self.X = X
        self.n_jobs = n_jobs
        return self.optimize(X, n_jobs=n_jobs)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        if _X is not None:
            assert isinstance(_X, ndarray)
            assert len(_X.shape) == 2
            self._X = _X

    @property
    def Method(self):
        return self._Method

    @Method.setter
    def Method(self, method):
        self._Method = method

    def optimize(self, X):
        pass

    def plot(self, ax=None):
        pass


class PCAOptimizer(Oprimizer):
    def __init__(self, X=None, method=None, verbose=0):
        super().__init__(X=X, method=method, verbose=verbose)
        self._data = dict(pca_scores=None, n_components=None, best_n_componet=None, shrunk_score=None, lw_score=None)

    @property
    def Method(self):
        return self._Method

    @Method.setter
    def Method(self, method):
        assert isinstance(method, PCA)
        del self.data
        self._Method = method

    @property
    def data(self):
        return self._data

    @data.deleter
    def data(self):
        self._data = dict(pca_scores=None, n_components=None, best_n_componet=None, shrunk_score=None, lw_score=None)

    @property
    def shrunk_score(self):
        return self._data["shrunk_score"]

    @shrunk_score.setter
    def shrunk_score(self, shrunk_score):
        self._data["shrunk_score"] = shrunk_score

    @property
    def lw_score(self):
        return self._data["lw_score"]

    @lw_score.setter
    def lw_score(self, lw_score):
        self._data["lw_score"] = lw_score

    @property
    def pca_scores(self):
        return self._data["pca_scores"]

    @pca_scores.setter
    def pca_scores(self, pca_scores):
        self._data["pca_scores"] = pca_scores

    @property
    def n_components(self):
        return self._data["n_components"]

    @n_components.setter
    def n_components(self, n_components):
        self._data["n_components"] = n_components

    @property
    def best_n_componet(self):
        return self._data["best_n_componet"]

    @best_n_componet.setter
    def best_n_componet(self, best_n_componet):
        self._data["best_n_componet"] = best_n_componet

    def _get_shrunk_score(self):
        if self.shrunk_score is None:
            self.shrunk_score = _shrunk_cov_score(self.X)

    def _get_lw_score(self):
        if self.lw_score is None:
            self.lw_score = _lw_score(self.X)

    def optimize(self, X=None, start_step=12, n_jobs=1):
        if X is not None:
            self.X = X
        elif X is None and self.X is not None:
            X = self.X
        Ndim = X.shape[-1]
        if Ndim < start_step * 2:
            start_step = 2
        _verbose = self.verbose >= 3
        best_n_componets, pca_scores_all, n_components_all = get_pca_n_componets(
            X,
            self.Method,
            start_step=start_step,
            verbose=_verbose,
            n_jobs=n_jobs,
        )
        self.best_n_componet = best_n_componets
        self.n_components = n_components_all
        self.pca_scores = pca_scores_all
        if self.verbose >= 2:
            message(f"Best n_componets value = {best_n_componets}", msg_type="info", add_date=True)
        self.Method.n_components = best_n_componets
        return self.Method

    def plot(self, ax=None, **kwarg):
        if self.best_n_componet is None:
            if self.verbose >= 0:
                message("No optimization performed.", add_date=True)
            raise RuntimeError("No optimization performed.")
        if ax is None:
            fig = plt.figure(figsize=[8, 4], dpi=100, facecolor="white")
            ax = fig.add_subplot(111)
        mask = np.logical_and(np.abs(self.pca_scores) != np.inf, self.pca_scores == self.pca_scores)
        sort_i = np.argsort(self.n_components[mask])
        ax.plot(self.n_components[mask][sort_i], self.pca_scores[mask][sort_i], label="PCA scores", **kwarg)
        ax.axvline(
            self.best_n_componet,
            color="k",
            label="PCA MLE: %d" % self.best_n_componet,
            linestyle=":",
        )
        self._get_shrunk_score()
        self._get_lw_score()
        # compare with other covariance estimators
        ax.axhline(
            self.shrunk_score,
            color="violet",
            label="Shrunk Covariance MLE",
            linestyle="-.",
        )
        ax.axhline(
            self.lw_score,
            color="orange",
            label="LedoitWolf MLE",
            linestyle="-.",
        )
        ax.set_xlabel("nb of components")
        ax.set_ylabel("CV scores")
        ax.set_yscale("symlog")
