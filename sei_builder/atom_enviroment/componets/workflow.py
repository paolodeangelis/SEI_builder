import imp

import numpy as np
from ase import Atoms
from joblib import parallel_backend
from sklearn.base import BaseEstimator

from ...functions import message
from .descriptros import Descriptors, DscribeDescriptors
from .dim_reduction import DimensionalityReduction, PCAOptimizer
from .filters import FilterBase, FilterMask, FilterSpecies
from .gaussian_mixture import CombinedGaussianMixture
from .scalers import MinMaxScaler, ScalerBase, StandardScaler


class WorkFlow:
    def __init__(self, verbose=0) -> None:
        self.verbose = verbose
        self.n_jobs = None
        self._X0 = None  # Starting system ASE Atoms or list
        self.Nimages = None
        self.Natoms = None
        self.Natoms_filtered = None
        self.Ndim = None
        self.Ndim_reduced = None
        self.Nfeature = None
        self.species = None
        self._Descriptor = None
        self._Xfeatures = None  # System featurized
        self._Filter = None
        self._Xfiltered = None  # Data filtered
        self._FeatureScaling = None
        self._Xscaled = None  # Data scales
        self._DimensionalityReduction = None
        self._Xreduced = None  # Data Reduced
        self._Clustering = None
        self._Ycluster = None  # Cluster labels
        self.n_clusters = None

    def __call__(self, systems, njobs=8, verbose=None) -> None:
        if verbose is not None:
            self.verbose = verbose
        self.n_jobs = njobs
        # Load data
        self.X0 = systems
        # compute descriptor
        self.compute_descriptor()
        # filering
        if self.Filter is not None:
            self.Xfiltered = self.filter_forward(self.Xfeatures)
        else:
            self.Xfiltered = self.Xfeatures
        # scaler
        if self.FeatureScaling is not None:
            self.fit_scaler(self.Xfiltered)
            self.Xscaled = self.get_scaled_data(self.Xfiltered)
        else:
            self.Xscaled = self.Xfiltered
        # reducing
        if self.DimensionalityReduction is not None:
            self.fit_dim_reduction(self.Xscaled)
            self.Xreduced = self.get_reduced_data(self.Xscaled)
        else:
            if self.verbose >= 1:
                message("No Dimesionality Redaction", msg_type="warning", add_date=True)
            self.Xreduced = self.Xscaled
        # clustering
        if self.ClusteringMethod is not None:
            self.fit_clustering(self.Xreduced)
            self.Ycluster = self.predict_clustering(self.Xreduced)
        else:
            if self.verbose >= 1:
                message("No Clustering Method set!", msg_type="warning", add_date=True)

    def run(self, *arg, **kwarg) -> None:
        self.__call__(*arg, **kwarg)

    @property
    def X0(self):
        return self._X0

    @X0.setter
    def X0(self, X0):
        assert isinstance(X0, list) or isinstance(X0, Atoms)
        if self._X0 is not None:
            del self.X0
        self._X0 = X0
        if isinstance(X0, Atoms):
            self._X0 = [self._X0]
        self.Nimages = len(self._X0)
        self.Natoms = len(self._X0[0])
        self.Ndim = 3
        self.species = self.get_species()
        if self.verbose >= 2:
            message(
                f"Starting system with {self.Nimages} images and {self.Natoms} atoms (species: {self.species})",
                add_date=True,
            )

    @X0.deleter
    def X0(self):
        self._X0 = None
        self.Nimages = None
        self.Natoms = None
        self.Ndim = None
        self.sepcies = None

    @property
    def Xfeatures(self):
        return self._Xfeatures

    @Xfeatures.setter
    def Xfeatures(self, Xfeatures):
        self._Xfeatures = Xfeatures
        self.Nfeature = Xfeatures.shape[2]

    @Xfeatures.deleter
    def Xfeatures(self):
        self._Xfeatures = None

    @property
    def Xfiltered(self):
        return self._Xfiltered

    @Xfiltered.setter
    def Xfiltered(self, Xfiltered):
        self._Xfiltered = Xfiltered

    @Xfiltered.deleter
    def Xfiltered(self):
        self._Xfiltered = None
        self.Natoms_filtered = None

    @property
    def Xscaled(self):
        return self._Xscaled

    @Xscaled.setter
    def Xscaled(self, Xscaled):
        self._Xscaled = Xscaled

    @Xscaled.deleter
    def Xscaled(self):
        self._Xscaled = None

    @property
    def Xreduced(self):
        return self._Xreduced

    @Xreduced.setter
    def Xreduced(self, Xreduced):
        self._Xreduced = Xreduced

    @Xreduced.deleter
    def Xreduced(self):
        self._Xreduced = None

    def get_species(self):
        species = set()
        for i in range(len(self.X0)):
            species.update(self.X0[i].get_chemical_symbols())
        return species

    @property
    def Ycluster(self):
        return self._Ycluster

    @Ycluster.setter
    def Ycluster(self, Ycluster):
        self._Ycluster = Ycluster

    @Ycluster.deleter
    def Ycluster(self):
        self.n_clusters
        self._Ycluster = None

    def get_labels(self):
        return self._Ycluster

    # Descriptors
    @property
    def Descriptor(self):
        return self._Descriptor

    @Descriptor.setter
    def Descriptor(self, descriptor):
        assert isinstance(descriptor, Descriptors)
        del self.Xfeatures
        del self.Xfiltered
        self._Descriptor = descriptor

    @Descriptor.deleter
    def Descriptor(self):
        self._Descriptor = None

    def get_descriptor(self):
        return self.Descriptor

    def set_descriptor(self, Filter):
        self.Descriptor = Filter

    def compute_descriptor(self):
        if self.verbose >= 2:
            message(
                f"Computing the descriptors ({self.Descriptor.get_number_of_features()} descriptors) ...",
                end="\r",
                add_date=True,
            )
        self.Descriptor.compute(X=self.X0, n_jobs=self.n_jobs)
        self.Xfeatures = self.Descriptor.X_features
        if self.verbose >= 2:
            message(
                f"Data featurized with {self.Nfeature} descriptors"
                + f"({self.Nimages}x{self.Natoms}x{self.Ndim} -> {self.Nimages}x{self.Natoms}x{self.Nfeature})",
                add_date=True,
            )

    # Filter
    @property
    def Filter(self):
        return self._Filter

    @Filter.setter
    def Filter(self, filter):
        assert isinstance(filter, FilterBase)
        del self.Xfiltered
        self._Filter = filter

    @Filter.deleter
    def Filter(self):
        self._Filter = None

    def get_filter(self):
        return self.Filter

    def set_filter(self, filter):
        self.Filter = filter

    def filter_forward(self, X):
        if self.verbose >= 3:
            message(
                f"Filtering data with {self.Filter.__str__()} descriptors ...",
                msg_type="debug",
                end="\r",
                add_date=True,
            )
        Xout = self.Filter.forward(X=X)
        if self.verbose >= 2:
            message(
                f"Data filtered with {self.Filter.__str__()} descriptors "
                + f"({X.shape[0]}x{X.shape[1]}x{X.shape[2]} -> {Xout.shape[0]}x{Xout.shape[1]}x{Xout.shape[2]})",
                add_date=True,
            )
        return Xout

    def filter_backward(self, X):
        Xout = self.Filter.backward(X=X)
        if self.verbose >= 2:
            message(
                f"Data filtered with {self.Filter.__str__()} descriptors "
                + f"({X.shape[0]}x{X.shape[1]}x{X.shape[2]} -> {Xout.shape[0]}x{Xout.shape[1]}x{Xout.shape[2]})",
                add_date=True,
            )
        return Xout

    # Feature Scaling
    @property
    def FeatureScaling(self):
        return self._FeatureScaling

    @FeatureScaling.setter
    def FeatureScaling(self, scaler):
        assert isinstance(scaler, ScalerBase)
        del self.Xscaled
        self._FeatureScaling = scaler

    @FeatureScaling.deleter
    def FeatureScaling(self):
        self._FeatureScaling = None

    def get_scaler(self):
        return self.FeatureScaling

    def set_scaler(self, scaler):
        self.FeatureScaling = scaler

    def fit_scaler(self, X):
        if self.verbose >= 3:
            message(
                f"Computing scaling prameters of {self.FeatureScaling.__str__()} scaler",
                msg_type="debug",
                add_date=True,
            )
        self.FeatureScaling.fit(X)

    def get_scaled_data(self, X):
        if self.verbose >= 2:
            message(f"Scaling data with {self.FeatureScaling.__str__()} scaler", add_date=True)
        return self.FeatureScaling.transform(X)

    # Dimensionality Reduction
    @property
    def DimensionalityReduction(self):
        return self._DimensionalityReduction

    @DimensionalityReduction.setter
    def DimensionalityReduction(self, dimensionality_reduction):
        assert isinstance(dimensionality_reduction, DimensionalityReduction)
        del self.Xscaled
        self._DimensionalityReduction = dimensionality_reduction

    @DimensionalityReduction.deleter
    def DimensionalityReduction(self):
        self._DimensionalityReduction = None

    def get_dim_reduction(self):
        return self.DimensionalityReduction

    def set_dim_reduction(self, dimensionality_reduction):
        self.DimensionalityReduction = dimensionality_reduction

    def fit_dim_reduction(self, X):
        if self.verbose >= 3:
            message(f"Training {self.DimensionalityReduction.__str__()} ...", msg_type="debug", add_date=True, end="\n")
        with parallel_backend("threading", n_jobs=self.n_jobs):
            self.DimensionalityReduction.fit(X)
        if self.verbose >= 3:
            message(" " * 1000, end="\r", msg_type="debug")
            message(f"{self.DimensionalityReduction.__str__()} Trained", msg_type="debug", add_date=True)

    def get_reduced_data(self, X):
        if self.verbose >= 2:
            message(
                f"Reducing data dimension with {self.DimensionalityReduction.__str__()} ...", add_date=True, end="\r"
            )
        with parallel_backend("threading", n_jobs=self.n_jobs):
            Xout = self.DimensionalityReduction.transform(X)
        self.Ndim_reduced = Xout.shape[2]
        if self.verbose >= 2:
            message(
                f"Data dimension reduced with {self.DimensionalityReduction.__str__()} ",
                f"({X.shape[0]}x{X.shape[1]}x{X.shape[2]} -> {Xout.shape[0]}x{Xout.shape[1]}x{Xout.shape[2]})",
                add_date=True,
            )
        return Xout

    # Clustering
    @property
    def ClusteringMethod(self):
        return self._Clustering

    @ClusteringMethod.setter
    def ClusteringMethod(self, clustering):
        assert isinstance(clustering, BaseEstimator)
        del self.Ycluster
        self._Clustering = clustering

    @ClusteringMethod.deleter
    def ClusteringMethod(self):
        del self.Ycluster
        self._Clustering = None

    def get_clustering(self):
        return self.ClusteringMethod

    def set_clustering(self, clustering):
        self.ClusteringMethod = clustering

    def fit_clustering(self, X):
        if self.verbose >= 3:
            message(
                f"Training the Clustering {self.ClusteringMethod.__str__()} method", msg_type="debug", add_date=True
            )
        elif self.verbose >= 2:
            message(f"Training the Clustering  {self.FeatureScaling.__str__()} method ...", add_date=True, end="\r")
        if callable(getattr(self, "warmup", None)):
            if self.verbose >= 3:
                message(
                    f"Warimg up the Method {self.ClusteringMethod.__str__()} method", msg_type="debug", add_date=True
                )
            with parallel_backend("threading", n_jobs=self.n_jobs):
                verbose_ = self.verbose >= 3
                self.warmup(X, verbose=verbose_)
        with parallel_backend("threading", n_jobs=self.n_jobs):
            self.ClusteringMethod.fit(X)
        if self.verbose >= 2:
            message(f"Clustering  {self.ClusteringMethod.__str__()} method trained", add_date=True, end="\n")

    def predict_clustering(self, X):
        if self.verbose >= 2:
            message(f"Predicting cluster with {self.ClusteringMethod.__str__()} method...", add_date=True, end="\r")
        with parallel_backend("threading", n_jobs=self.n_jobs):
            labels = self.ClusteringMethod.predict(X)
        if self.verbose >= 2:
            message(
                f"Found {len(np.unique(labels))} clusters with {self.ClusteringMethod.__str__()} method...",
                add_date=True,
                end="\n",
            )
        return labels
