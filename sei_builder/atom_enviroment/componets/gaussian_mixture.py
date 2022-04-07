import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from numpy import ndarray
from scipy import linalg
from scipy.stats import chi2, linregress
from shapely import affinity
from shapely.geometry.point import Point
from sklearn.mixture import GaussianMixture

from ...functions import message


def _get_BIC_score(gmm, X, n_components, N=3):
    bic_score = np.zeros(N)
    for i in range(N):
        gmm.n_components = n_components
        gmm.fit(X)
        _ = gmm.predict(X)
        bic_score[i] = gmm.bic(X)
    return np.mean(bic_score)


def _GMM_n_componets_scanner(gmm, X, step=8, interval=None, early_stop=False):
    if interval is None:
        Nd = X.shape[-1]
        n_components = np.arange(1, Nd, step, dtype=int)
    else:
        n_components = np.arange(interval[0], interval[1], step, dtype=int)
        n_components = n_components[n_components > 0]
    bic_score = np.ones(n_components.shape) * 1e99
    for i, n_components_ in enumerate(n_components):
        bic_score[i] = _get_BIC_score(gmm, X, n_components_)
        if i > 0 and early_stop and bic_score[i] > bic_score[i - 1]:
            break
    n_components = n_components[bic_score < 1e99]
    bic_score = bic_score[bic_score < 1e99]
    bic_best_i = np.where(bic_score == np.min(bic_score))
    interval = [n_components[bic_best_i] - step, n_components[bic_best_i] + step]
    return (
        int(n_components[bic_best_i]),
        n_components,
        float(bic_score[bic_best_i]),
        bic_score,
        interval,
    )


def get_pca_n_componets(gmm, X, start_step=12, verbose=False, n_jobs=1):
    interval = None
    bic_scores_all = []
    n_components_all = []
    if start_step >= 2:
        # corse
        best_n_componets, n_components, bic_score, bic_scores, interval = _GMM_n_componets_scanner(
            gmm, X, step=start_step, early_stop=True
        )
        bic_scores_all = np.append(bic_scores_all, bic_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Corse search: n_components={best_n_componets} (BIC score = {bic_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    if start_step >= 4:
        # fine
        best_n_componets, n_components, bic_score, bic_scores, interval = _GMM_n_componets_scanner(
            gmm, X, step=start_step // 2, interval=interval, early_stop=True
        )
        bic_scores_all = np.append(bic_scores_all, bic_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Fine search: n_components={best_n_componets} (BIC score = {bic_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    if start_step >= 8:
        # finer
        best_n_componets, n_components, bic_score, bic_scores, interval = _GMM_n_componets_scanner(
            gmm, X, step=start_step // 2, interval=interval, early_stop=True
        )
        bic_scores_all = np.append(bic_scores_all, bic_scores)
        n_components_all = np.append(n_components_all, n_components)
        if verbose:
            message(
                f"Finer search: n_components={best_n_componets} (BIC score = {bic_score:1.3e}) ",
                msg_type="debug",
                add_date=True,
            )
    # final
    best_n_componets, n_components, bic_score, bic_scores, intervall = _GMM_n_componets_scanner(
        gmm, X, step=1, interval=interval, early_stop=False
    )
    bic_scores_all = np.append(bic_scores_all, bic_scores)
    n_components_all = np.append(n_components_all, n_components)
    if verbose:
        message(
            f"Final search: n_components={best_n_componets} (BIC score = {bic_score:1.3e}) ",
            msg_type="debug",
            add_date=True,
        )
    return best_n_componets, bic_scores_all, n_components_all


def _get_conditional_probability(prob_ik):
    """compute the conditiola probablity that x_i arises from the k-th mixture component P (x_i | k ) as defined by
    Baudry et al. (https://www.doi.org/10.1198/jcgs.2010.08111).
    $
        t_{ik}(\\hat{\theta_k}) =
        \\dfrac{\\hat{p}_k\\phi(\\mathbf{x}_i | \\hat{\\mathbf{a}}_k )}
        {\\sum_{j=1}^K \\hat{p}_j\\phi(\\mathbf{x}_i | \\hat{\\mathbf{a}}_j )}
    $
    """
    assert isinstance(prob_ik, ndarray)
    assert len(prob_ik.shape) == 2

    t_ik = prob_ik / prob_ik.sum(axis=1)  # Note that scikit-learn the probabilty is already normaized
    return t_ik


def _shannon_entropy(prob_ik, axis=0, log_f=np.log2):
    r"""compute the Shannon information entropy:
    S =  \sum_i^N p_i \log (p_i)
    """
    assert isinstance(prob_ik, ndarray)
    assert len(prob_ik.shape) == 2
    assert callable(log_f)

    def limit_log(prob_ik):
        close0_mask = np.isclose(np.abs(prob_ik), 0.0)
        log = np.zeros(prob_ik.shape)
        log[~close0_mask] = log_f(prob_ik[~close0_mask])
        return log

    S = -1 * prob_ik * limit_log(prob_ik)
    S = S.sum(axis=axis)
    return S


def get_entropy(prob_ik):
    """Compute the Entropy definde by equation (6) in Baudry et al. work (https://www.doi.org/10.1198/jcgs.2010.08111).
    $
        ENT(K) = − \\sum_{k=1}^{K} \\sum_{i=1}^{n} t_{ik}(\\hat{\theta_k}) \\log (t_{ik}(\\hat{\theta_k}))
    $
    """
    assert isinstance(prob_ik, ndarray)
    assert len(prob_ik.shape) == 2
    ent = _shannon_entropy(prob_ik, axis=0).sum()
    return ent


def _merge_probability(prob_ik, ki, kj):
    r"""Compute the new probability after merge 2 mixture components
    $
    p_{ik\cup k'} = p_{ik} +p_{ik'}
    $
    """
    assert isinstance(prob_ik, ndarray)
    # assert isinstance(ki, int) and isinstance(kj, int)
    assert len(prob_ik.shape) == 2
    N, K = prob_ik.shape
    new_prob_ik = np.zeros((N, K - 1))
    ind_k = min([ki, kj])
    ind = (np.arange(K) != ki) & (np.arange(K) != kj)
    ind_new = np.arange(K - 1) != ind_k
    new_prob_ik[:, ind_new] = prob_ik[:, ind]
    new_prob_ik[:, ind_k] = prob_ik[:, ki] + prob_ik[:, kj]
    new_prob_ik = new_prob_ik / new_prob_ik.sum(axis=1)[:, np.newaxis]
    return new_prob_ik


def _get_entropy_difference(prob_ik, ki, kj):
    r"""Compute the Entropy difeference after mergin 2 mixuture component using equation (7) in
    Baudry et al. work (https://www.doi.org/10.1198/jcgs.2010.08111).
    $
    \Delta Ent(k, k') =
    - \sum_{i=1}^{n} (t_{ik} \log (t_{ik}) + t_{ik'} \log (t_{ik'}) )
    + \sum_{i=1}^{n} t_{ik\cup k'} \log (t_{ik\cup k'})
    $
    """
    assert isinstance(prob_ik, ndarray)
    # assert isinstance(ki, int) and isinstance(kj, int)
    assert len(prob_ik.shape) == 2
    if ki == kj:
        return 0.0
    else:
        new_prob_ik = _merge_probability(prob_ik, ki, kj)
        prob_ij = prob_ik[:, [ki, kj]]
        new_prob_i = new_prob_ik[:, min(ki, kj)][:, np.newaxis]
        deltaS = _shannon_entropy(prob_ij).sum() - _shannon_entropy(new_prob_i).sum()
        return deltaS


def _merge_2_clusters(prob_ik):
    """
    Merge 2 clusters into new cluster based on the
    probabilities that particles initially belong to each of the original
    clusters with a certain probability and using an entropy criterion.

    See https://doi.org/10.1198/jcgs.2010.08111 (Baudry et al.)
    """
    N, K = prob_ik.shape
    d_evals = []
    to_merge = []
    for ki in range(K):
        for kj in range(ki):
            delta_ent = _get_entropy_difference(prob_ik, ki, kj)
            d_evals.append(delta_ent)
            to_merge.append([ki, kj])
    best_ind = d_evals.index(max(d_evals))
    ki, kj = to_merge[best_ind]
    new_prob_ik = _merge_probability(prob_ik, ki, kj)
    new_entropy = get_entropy(new_prob_ik)
    return ki, kj, new_prob_ik, new_entropy


def _merge_explore(prob_ik):
    """Merge 2, 3, .. K^BIC-1 clusters into new cluster based on the
    probabilities that particles initially belong to each of the original
    clusters with a certain probability and using an entropy criterion.
    """
    N, K = prob_ik.shape
    nk = np.arange(1, K)[::-1]
    entropy = np.zeros(K - 1)
    new_prob = prob_ik.copy()
    k2merge = np.zeros((K - 1, 2), dtype=int)
    for i in range(K - 1):
        ki, kj, new_prob, new_S = _merge_2_clusters(new_prob)
        entropy[i] = new_S
        k2merge[i, :] = [int(ki), int(kj)]
    return nk, entropy, k2merge


def _get_linear_fit(x, y):
    m, x0, _, _, _ = linregress(x, y)
    return m, x0


def _straigt_line(x, x0, m):
    return x * m + x0


def _get_best_merge(entropies, nk):
    """Get the best numerber of mixture componets to merge searching the breaking point for the 'elbow rule'
    as sugest by Baudry et al. (https://doi.org/10.1198/jcgs.2010.08111)
    """
    if len(entropies) <= 3:
        best_ind = 0
        return int(best_ind)

    nk_bp = np.arange(1, len(nk) - 1)
    residue = np.zeros(nk_bp.shape)

    for i, i_bp in enumerate(nk_bp):
        left_x = nk[: i_bp + 1]
        left_y = entropies[: i_bp + 1]
        right_x = nk[i_bp:]
        right_y = entropies[i_bp:]
        left_m, left_x0 = _get_linear_fit(left_x, left_y)
        right_m, right_x0 = _get_linear_fit(right_x, right_y)
        lr_y = np.concatenate(
            (
                _straigt_line(left_x[:-1], left_x0, left_m),
                [np.mean([_straigt_line(left_x[-1], left_x0, left_m), _straigt_line(right_x[0], right_x0, right_m)])],
                _straigt_line(right_x[1:], right_x0, right_m),
            )
        )
        residue[i] = np.power(entropies - lr_y, 2).sum()
    best_ind = nk_bp[residue == residue.min()]  # np.where(residue == residue.min())[0]
    return int(best_ind)


def merging_gausian_mixture(prob_ik, verbose=False):
    """Merging the gaussian Mixture following Baudry et al. (https://doi.org/10.1198/jcgs.2010.08111) work"""
    if verbose:
        message(f"Merging the gaussian Mixture following Baudry et al. ...", msg_type="debug", add_date=True, end="\r")
    nk, entropies, k2merge = _merge_explore(prob_ik)
    i_best = _get_best_merge(entropies, nk)
    if verbose:
        message(
            f"Following the Baudry et al. criteria we merged {i_best +1} gaussian mixture "
            + f"(best BIC-1 = {nk[0]} (ent {entropies[0]:1.2e}) -> best merged {nk[i_best]} (ent"
            f" {entropies[i_best]:1.2e}))",
            msg_type="debug",
            add_date=True,
        )
    return nk, entropies, k2merge[: i_best + 1]


# Plot functions


def _create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, lengths[0], lengths[1])
    ellr = affinity.rotate(ell, angle)
    return ellr


def _merge_ellipses(ellipses, mergeing_map):
    for [ei, ej] in mergeing_map:
        union = []
        for i in range(len(ellipses[0])):
            if ellipses[ei][i] is None:
                union.append(ellipses[ej][i])
            if ellipses[ej][i] is None:
                union.append(ellipses[ei][i])
            else:
                union.append(ellipses[ei][i].union(ellipses[ej][i]))
        i_new = min(ei, ej)
        i_pop = max(ei, ej)
        ellipses[i_new] = union
        _ = ellipses.pop(i_pop)
    return ellipses


def _plot_ellipses(ellipse, color, ax, alpha=0.1):
    ls_list = ["-", "-.", "--", ":"]
    for i, ellipse_ in enumerate(ellipse):
        if ellipse_.type == "MultiPolygon":
            verts = [np.array(x.exterior.coords.xy) for x in ellipse_.geoms]
            verts = np.concatenate(verts, axis=1)
        else:
            verts = np.array(ellipse_.exterior.coords.xy)
        patch = Polygon(verts.T, facecolor=color, edgecolor=color, lw=1.0, ls=ls_list[i])
        patch.set_clip_box(ax.bbox)
        patch.set_alpha(alpha)
        ax.add_artist(patch)
        patch = Polygon(verts.T, facecolor="none", edgecolor=color, lw=1.0, ls=ls_list[i])
        patch.set_clip_box(ax.bbox)
        patch.set_alpha(1)
        ax.add_artist(patch)


def _plot_mixture(
    X, Y_, Y_merged, means, covariances, colors, mergeing_map, ax=None, indexs=[0, 1], ci=[0.33, 0.66], **karg
):
    if ax is None:
        fig = plt.figure(figsize=[8, 4], dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
    i_x, i_y = indexs
    ls_list = ["-", "-.", "--", ":"]
    dof = covariances.shape[0]  # deegree of freedom = number of gaussian mixture
    ellipses = [[None] * len(ci)] * dof
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        idx = np.arange(len(v) - 1, -1, -1)
        # idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:, idx]
        v = np.sqrt(v)  # eighevalue = principal component varaince
        u = w[0] / linalg.norm(w[0])  # 1 principal axis
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[i_y] / u[i_x])
        angle = 180.0 * angle / np.pi  # convert to degrees
        i_ellipses = []
        for ci_ in ci:
            scale = np.sqrt(
                chi2.ppf(ci_, dof)
            )  # Geometry of the Multivariate Normal Distribution as describe in https://online.stat.psu.edu/stat505/lesson/4/4.6
            i_ellipses.append(_create_ellipse([mean[i_x], mean[i_y]], [v[i_x] * scale, v[i_y] * scale], angle=angle))
        ellipses[i] = i_ellipses
    # merge
    ellipses = _merge_ellipses(ellipses, mergeing_map)
    # plot
    for i, ellipse in enumerate(ellipses):
        _plot_ellipses(ellipse, colors[i], ax, alpha=0.03)
    for i in range(len(ellipses)):
        karg.update({"zorder": 10})
        ax.scatter(X[Y_merged == i, i_x], X[Y_merged == i, i_y], s=10, color=colors[i], label=f"cluster {i}", **karg)
    for ls_, ci_ in zip(ls_list, ci):
        ax.scatter(np.nan, np.nan, s=160, lw=0.8, ls=ls_, ec="k", fc="none", label=f"C.I. {ci_*100}%")

    ax.set_xlabel(f"$x_{i_x+1}$")
    ax.set_ylabel(f"$x_{i_y+1}$")
    ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])


def _plot_bic(n_components, n_components_best, bic_score, ax=None, **karg):
    if ax is None:
        fig = plt.figure(figsize=[8, 4], dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
    ax.bar(n_components, bic_score, **karg)
    ax.bar(n_components[n_components == n_components_best], bic_score[n_components == n_components_best], label="best")
    delta = np.max(bic_score) - np.min(bic_score)
    ax.set_ylim([np.min(bic_score) - delta * 0.05, np.max(bic_score) + delta * 0.05])
    ax.set_xlabel("n_components")
    ax.set_ylabel("BIC score")
    ax.set_xticks(np.unique(n_components))
    ax.legend()


def _plot_merge(n_mixture, merging_entropies, i_bp, ax=None, **kargs):
    if ax is None:
        fig = plt.figure(figsize=[8, 4], dpi=100, facecolor="white")
        ax = fig.add_subplot(111)
    left_x = n_mixture[: i_bp + 1]
    left_y = merging_entropies[: i_bp + 1]
    right_x = n_mixture[i_bp:]
    right_y = merging_entropies[i_bp:]
    left_m, left_x0 = _get_linear_fit(left_x, left_y)
    right_m, right_x0 = _get_linear_fit(right_x, right_y)
    ax.plot(n_mixture, merging_entropies, **kargs)
    ax.scatter(n_mixture[i_bp], merging_entropies[i_bp])
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_ = np.arange(-5, len(n_mixture) + 5)
    ax.plot(x_, _straigt_line(x_, left_x0, left_m), ls="--", lw=0.75, color="k")
    ax.plot(x_, _straigt_line(x_, right_x0, right_m), ls="--", lw=0.75, color="k")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"n_components ($K$)")
    ax.set_ylabel(r"Shannon's entropy $S(K) = − \sum_{k}^{K} \sum_{i}^{n} t_{ik} \log (t_{ik})$")


class CombinedGaussianMixture(GaussianMixture):
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.bic_data = {"best_n_componets": None, "bic_scores": None, "n_components_all": None}
        self.merge_data = {"n_mixture": None, "merging_entropies": None, "merging_path": None}

    @property
    def n_mixture(self):
        return self.merge_data["n_mixture"]

    @n_mixture.setter
    def n_mixture(self, n_mixture):
        self.merge_data["n_mixture"] = n_mixture

    @n_mixture.deleter
    def n_mixture(self):
        self.merge_data["n_mixture"] = None

    @property
    def merging_entropies(self):
        return self.merge_data["merging_entropies"]

    @merging_entropies.setter
    def merging_entropies(self, merging_entropies):
        self.merge_data["merging_entropies"] = merging_entropies

    @merging_entropies.deleter
    def merging_entropies(self):
        self.merge_data["merging_entropies"] = None

    @property
    def merging_path(self):
        return self.merge_data["merging_path"]

    @merging_path.setter
    def merging_path(self, merging_path):
        self.merge_data["merging_path"] = merging_path

    @merging_path.deleter
    def merging_path(self):
        self.merge_data["merging_path"] = None

    @property
    def best_n_componets(self):
        return self.bic_data["best_n_componets"]

    @best_n_componets.setter
    def best_n_componets(self, best_n_componets):
        self.bic_data["best_n_componets"] = best_n_componets

    @property
    def bic_scores(self):
        return self.bic_data["bic_scores"]

    @bic_scores.setter
    def bic_scores(self, bic_scores):
        self.bic_data["bic_scores"] = bic_scores

    @property
    def n_components_all(self):
        return self.bic_data["n_components_all"]

    @n_components_all.setter
    def n_components_all(self, n_components_all):
        self.bic_data["n_components_all"] = n_components_all

    def warmup(self, X, start_step=4, verbose=True, n_jobs=1):
        del self.n_mixture, self.merging_entropies, self.merging_path
        self.best_n_componets, self.bic_scores, self.n_components_all = get_pca_n_componets(
            self, X, start_step=start_step, verbose=verbose, n_jobs=n_jobs
        )
        self.n_components = self.best_n_componets
        prob_ik = self.predict_proba(X)
        self.n_mixture, self.merging_entropies, self.merging_path = merging_gausian_mixture(prob_ik, verbose=verbose)

    def fit(self, X, y=None):
        super().fit(X, y=y)

    def predict_merge_prob(self, X):
        prob_ik = self.predict_proba(X)
        for ki, kj in self.merging_path:
            prob_ik = _merge_probability(prob_ik, ki, kj)
        return prob_ik

    def predict_merg(self, X):
        prob_ik = self.predict_merge_prob(X)
        new_labels = np.zeros(prob_ik.shape[0], dtype=int)
        for i, wi in enumerate(prob_ik):
            new_labels[i] = np.argmax(wi)
        return new_labels

    def predict(self, X):
        labels = self.predict_not_merged(X)
        if self.merging_path is not None:
            labels = self.predict_merg(X)
        return labels

    def predict_not_merged(self, X):
        labels = super().predict(X)
        return labels

    def plot_mixture(self, X, colors=None, ax=None, coordinates=[0, 1], ci=[0.33, 0.66], **karg):
        if colors is None:
            colors = sns.color_palette("tab10", self.n_components + 1)
        Y = self.predict_not_merged(X)
        Y_merged = self.predict(X)
        means = self.means_
        covariances = self.covariances_
        mergeing_path = self.merging_path
        _plot_mixture(
            X, Y, Y_merged, means, covariances, colors, mergeing_path, ax=ax, indexs=coordinates, ci=ci, **karg
        )

    def plot_bic(self, ax=None, **karg):
        n_components = self.n_components_all
        n_components_best = self.best_n_componets
        bic_score = self.bic_scores
        _plot_bic(n_components, n_components_best, bic_score, ax=ax, **karg)

    def plot_merging(self, ax=None, **karg):
        n_mixture = self.n_mixture
        merging_entropies = self.merging_entropies
        i_bp = len(self.merging_path)
        _plot_merge(n_mixture, merging_entropies, i_bp, ax=ax, **karg)

    def __str__(self):
        if self.merging_path is not None:
            str_ = (
                f"CombinedGaussianMixture(n_components={self.n_components} merg into"
                f" {self.n_components - len(self.merging_path)})"
            )
        else:
            str_ = f"CombinedGaussianMixture(n_components=None, warmup=False)"
        return str_
