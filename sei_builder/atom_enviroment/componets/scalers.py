import numpy as np


class ScalerBase:
    def __init__(self, fit_all=False) -> None:
        self.fit_all = fit_all
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def __str__(self) -> str:
        str_ = f"ScalerBase(fit_all={self.fit_all}) -> UNK"
        return str_


class StandardScaler(ScalerBase):
    def __init__(self, fit_all=False) -> None:
        self.mean = None
        self.std = None
        self.fit_all = fit_all

    def fit(self, X):
        if self.fit_all:
            X = np.concatenate(X)
            X = X[np.newaxis, :, :]

        self.mean = np.mean(np.mean(X, axis=2), axis=1)
        self.mean = self.mean[:, np.newaxis, np.newaxis]
        self.std = np.std(np.std(X, axis=2), axis=1)
        self.std[self.std == 0.0] = 1.0  # avoid dividing per 0
        self.std = self.std[:, np.newaxis, np.newaxis]

    def transform(self, X):
        return (X - self.mean) / self.std

    def __str__(self) -> str:
        str_ = f"StandardScaler(fit_all={self.fit_all}) -> (x - μ) / σ"
        return str_


class MinMaxScaler(ScalerBase):
    def __init__(self, fit_all=False) -> None:
        self.min = None
        self.max = None
        self.fit_all = fit_all

    def fit(self, X):
        if self.fit_all:
            X = np.concatenate(X)
            X = X[np.newaxis, :, :]

        self.max = np.max(np.max(X, axis=2), axis=1)
        self.max = self.max[:, np.newaxis, np.newaxis]
        self.min = np.min(np.min(X, axis=2), axis=1)
        self.min = self.min[:, np.newaxis, np.newaxis]

    def transform(self, X):
        den = self.max - self.min
        den[den == 0.0] = 1.0  # avoid dividing per 0
        return (X - self.min) / (self.max - self.min)

    def __str__(self) -> str:
        str_ = f"MinMaxScaler(fit_all={self.fit_all}) -> (x - x_min) / (x_max -x_min)"
        return str_
