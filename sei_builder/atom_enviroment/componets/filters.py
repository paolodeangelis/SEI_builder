import numpy as np
from ase import Atoms
from numpy import ndarray


class FilterBase:
    def __init__(self) -> None:
        self.X_source = None
        self.X_out = None
        self.Ntrue = None
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class FilterMask(FilterBase):
    def __init__(self, mask, axis=1) -> None:
        self.X_source = None
        self.X_out = None
        self.mask = mask
        self.axis = axis
        self.Ntrue = None

    def forward(self, X):
        self.X_source = X.copy()
        if self.axis is None:
            self.X_out = self.X_source[self.mask]
        elif self.axis == 0:
            self.X_out = self.X_source[self.mask, :, :]
        elif self.axis == 1:
            self.X_out = self.X_source[:, self.mask, :]
        elif self.axis == 2:
            self.X_out = self.X_source[:, :, self.mask]
        self.Ntrue = [len(i) for i in np.where(self.mask)]
        return self.X_out

    def backward(self, X):
        if self.X_source.dtype == np.dtype(float):
            X_back = np.ones(self.X_source.shape, dtype=self.X_source.dtype) * np.nan
        else:
            X_back = np.ones(self.X_source.shape, dtype=self.X_source.dtype) * -1
        if np.all(X == self.X_out):
            X_back = self.X_source.copy()
        else:
            if self.axis is None:
                X_back[self.mask] = X
            elif self.axis == 0:
                X_back[self.mask, :, :] = X
            elif self.axis == 1:
                X_back[:, self.mask, :] = X
            elif self.axis == 2:
                X_back[:, :, self.mask] = X
        return X_back

    def __str__(self) -> str:
        str = "FilterMask("
        if self.X_source is not None:
            shape0 = list(self.X_source.shape)
            str += f"{shape0[0]}x{shape0[1]}x{shape0[2]}"
        if self.mask is not None:
            if self.axis is not None:
                shape1 = list(self.X_source.shape)
                shape1[self.axis] = len(self.mask)
            else:
                shape1 = list(self.mask.shape)
            str += f" -> {shape1[0]}x{shape1[1]}x{shape1[2]}"
        str += f")"
        return str


class FilterSpecies(FilterBase):
    def __init__(self, species, images) -> None:
        self.X_source = None
        self.X_out = None
        self._species = None
        self.species = species
        self._images = None
        self.images = images
        self.mask = None
        self.Ntrue = None

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, species):
        if isinstance(species, str):
            species = [species]
        self._species = species
        self.mask = None

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        if images is not None:
            assert isinstance(images, list) or isinstance(images, Atoms)
            if isinstance(images, Atoms):
                images = [images]
            self._images = images
            self.mask = None

    def get_mask(self):
        if self.mask is None:
            mask = []
            for img in self.images:
                symbols = np.array(img.get_chemical_symbols())
                for j, s in enumerate(self.species):
                    if j == 0:
                        mask_ = symbols == s
                    else:
                        mask_ = np.logical_or(mask_, symbols == s)
                mask.append(mask_)
            self.mask = mask
        return self.mask

    def forward(self, X):
        self.X_source = X.copy()
        self.mask = self.get_mask()
        self.Ntrue = [len(i) for i in np.where(self.mask[0])]
        self.X_out = np.zeros((X.shape[0], self.Ntrue[0], X.shape[2]), dtype=self.X_source.dtype)
        for i, mask_ in enumerate(self.mask):
            self.X_out[i, :, :] = X[i, mask_, :]
        return self.X_out

    def backward(self, X):
        if self.X_source.dtype == np.dtype(float):
            X_back = np.ones(self.X_source.shape, dtype=self.X_source.dtype) * np.nan
        else:
            X_back = np.ones(self.X_source.shape, dtype=self.X_source.dtype) * -1
        if np.all(X == self.X_out):
            X_back = self.X_source.copy()
        else:
            for i, mask_ in enumerate(self.mask):
                X_back[i, mask_, :] = X[i, :, :]
        return X_back

    def __str__(self) -> str:
        str = f"FilterSpecies(species={self.species}"
        if self.X_source is not None:
            shape0 = list(self.X_source.shape)
            str += f", {shape0[0]}x{shape0[1]}x{shape0[2]}"
        if self.mask is not None:
            shape1 = list(self.X_source.shape)
            shape1[1] = self.Ntrue[0]
            str += f" -> {shape1[0]}x{shape1[1]}x{shape1[2]}"
        str += f")"
        return str


# class Filter(FilterBase):
#     def __init__(self, mask=None, species=None, axis=None) -> None:
#         self._mask = None
#         self._species = None
#         self.mask = mask
#         self.species = species
#         self.axis = None


#     @property
#     def mask(self):
#         return self._mask

#     @mask.setter
#     def mask(self, mask):
#         assert isinstance(mask, ndarray)
#         assert self._species is None
#         assert self.axis is not None or len(mask.shape) is 3
#         self._mask = mask

#     @property
#     def species(self):
#         return self._species

#     @species.setter
#     def species(self, species):
#         assert isinstance(species, list) or isinstance(species, str)
#         assert self._mask is None
#         self._species = species
