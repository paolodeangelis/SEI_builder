import numpy as np
from ase import Atoms
from dscribe.descriptors import Descriptor as DscribeType

from ...functions import message


class Descriptors:
    def __init__(self, X=None, method=None, verbose=0):
        self._X = None
        self.Nimages = None
        self.Npoints = None
        self.Ndim = None
        self.X = X
        self.verbose = verbose
        self._Method = method
        self._X_features = None

    def __call__(self, X, verbose=0, method=None):
        self.X = X
        self.verbose = verbose
        if method:
            self.Method = method

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        if _X is not None:
            assert isinstance(_X, list) or isinstance(_X, Atoms)
            self._X = _X
            if isinstance(_X, Atoms):
                self._X = [self._X]
            self.Nimages = len(self._X)
            self.Npoints = len(self._X[0])
            self.Ndim = 3

    @property
    def X_features(self):
        if self._X_features is None and self.verbose >= 2:
            message("The data as to be featurazed yet, run the method 'compute'", msg_type="warning", add_date=True)
        return self._X_features

    @X_features.setter
    def X_features(self, X_features):
        if len(X_features.shape) == 2:
            X_features = X_features[np.newaxis, :, :]
        self._X_features = X_features

    @X_features.deleter
    def X_features(self):
        self._X_features = None

    @property
    def Method(self):
        return self._Method

    @Method.setter
    def Method(self, method):
        self._Method = method
        del self.X_features
        if self.verbose >= 2:
            message(f"Set {self._Method.__str__()} as Descriptors", add_date=True)

    def set_method(self, method):
        self.Method = method

    def get_method(self):
        return self.Method

    def get_number_of_features(self):
        return self._Method.get_number_of_features()

    def compute(self, X=None, n_jobs=8):
        if X is None and self._X is None:
            message("Missing the required System data 'X'", add_date=True)
        if X is None:
            X = self._X
        _verbose = self.verbose >= 3
        X_features = self._Method.compute(X, n_jobs=n_jobs, verbose=_verbose)
        return X_features

    def __str__(self):
        _str = "Descriptors("
        if self._X is not None:
            _str += f"X=array({self._X.shape}), "
        else:
            _str += f"X=None, "
        if self._Method:
            _str += f"method={self._Method.__str__()}, "
        else:
            _str += f"method=None, "
        _str += f"verbose={self.verbose})"
        return _str


class DscribeDescriptors(Descriptors):
    @property
    def Method(self):
        return self._Method

    @Method.setter
    def Method(self, method):
        assert isinstance(method, DscribeType)
        self._Method = method
        del self.X_features
        if self.verbose >= 2:
            message(
                f"Set {self._Method.__str__()} as Descriptor, with {self.Method.get_number_of_features()} descriptors",
                add_date=True,
            )

    def compute(self, X=None, n_jobs=8):
        if X is None and self._X is None:
            message("Missing the required System data 'X' i.e. list(ASE.Atoms)", add_date=True)
        if X is None:
            X = self._X
        _verbose = self.verbose >= 3
        self.X_features = self._Method.create(X, n_jobs=n_jobs, verbose=_verbose)
        # solve the fact that DSCRIVE treat the non-neighbor values as "infinitely far".
        self.X_features[self.X_features == np.inf] = 0.0
        self.X_features[self.X_features == -np.inf] = 0.0
        self.X_features[self.X_features != self.X_features] = 0.0
        # return self.X_features

    def __str__(self):
        _str = "DscribeDescriptors("
        if self._X is not None:
            _str += f"X=array({self._X.shape}), "
        else:
            _str += f"X=None, "
        if self._Method:
            _str += f"method={self._Method.__str__()}, "
        else:
            _str += f"method=None, "
        _str += f"verbose={self.verbose})"
        return _str
