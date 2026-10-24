"""Ordinary kriging.

A compact, dependency-light ordinary-kriging implementation used to estimate
well properties (porosity, saturation, net thickness, ...) at the centres of
Voronoi cells that have no well in them.

The workflow is the textbook one:

1. Build an *experimental* (semi)variogram from the sample pairs.
2. Fit a theoretical model (spherical by default) to it - giving nugget, sill
   and range.
3. For each target location solve the ordinary-kriging system

   ::

       | Γ  1 | | λ |   | γ0 |
       | 1ᵀ 0 | | μ | = | 1  |

   where ``Γ_ij = γ(|xi - xj|)`` between samples, ``γ0_i = γ(|xi - x0|)`` to the
   target, ``λ`` are the weights and ``μ`` is the Lagrange multiplier.  The
   estimate is ``Σ λi · vi``.

The sample covariance matrix is factorised once and reused for every target,
so estimating thousands of cells is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist


VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear")

# Reliability rule of thumb (Bostan, 2017): every lag class/bin of the
# experimental semivariogram should hold at least this many data pairs, and the
# maximum lag should not exceed half the study-area diagonal.
_MIN_PAIRS_PER_LAG = 30


def _model_gamma(h: np.ndarray, nugget: float, sill: float, rng: float, model: str) -> np.ndarray:
    """Theoretical semivariance γ(h).  ``sill`` is the *total* sill (incl. nugget)."""
    h = np.asarray(h, dtype=float)
    rng = max(rng, 1e-9)
    psill = max(sill - nugget, 0.0)
    out = np.empty_like(h)
    if model == "spherical":
        hr = np.minimum(h / rng, 1.0)
        struct = psill * (1.5 * hr - 0.5 * hr ** 3)
        struct = np.where(h >= rng, psill, struct)
    elif model == "exponential":
        struct = psill * (1.0 - np.exp(-3.0 * h / rng))
    elif model == "gaussian":
        struct = psill * (1.0 - np.exp(-3.0 * (h / rng) ** 2))
    elif model == "linear":
        # Bounded linear: rises straight to the sill at the range, then flat.
        # Kept bounded (rather than the classic unbounded power model) so the
        # kriging system stays well-conditioned like the other three models.
        hr = np.minimum(h / rng, 1.0)
        struct = psill * hr
    else:
        raise ValueError(f"Unknown variogram model: {model}")
    out = nugget + struct
    out = np.where(h <= 0, 0.0, out)
    return out


@dataclass
class VariogramModel:
    """A fitted **isotropic** theoretical variogram."""

    model: str
    nugget: float
    sill: float
    rng: float           # correlation range (ft)

    def gamma(self, h) -> np.ndarray:
        return _model_gamma(np.asarray(h, dtype=float), self.nugget, self.sill, self.rng, self.model)

    def describe(self) -> str:
        return (f"{self.model}: nugget={self.nugget:.4g}, sill={self.sill:.4g}, "
                f"range={self.rng:.0f} ft")

    def equation(self, numeric: bool = True) -> str:
        """The fitted variogram equation γ(h).

        With ``numeric`` the nugget (C₀), partial sill (C) and range (a) are
        substituted; otherwise the symbolic form is returned.
        """
        if numeric:
            c0 = f"{self.nugget:.4g}"
            c = f"{max(self.sill - self.nugget, 0.0):.4g}"
            a = f"{self.rng:.0f}"
        else:
            c0, c, a = "C₀", "C", "a"
        if self.model == "spherical":
            return (f"γ(h) = {c0} + {c}·[1.5(h/{a}) − 0.5(h/{a})³] for h<{a}; "
                    f"{c0}+{c} for h≥{a};  γ(0)=0")
        if self.model == "exponential":
            return f"γ(h) = {c0} + {c}·[1 − exp(−3h/{a})];  γ(0)=0"
        if self.model == "gaussian":
            return f"γ(h) = {c0} + {c}·[1 − exp(−3(h/{a})²)];  γ(0)=0"
        if self.model == "linear":
            return f"γ(h) = {c0} + {c}·min(h/{a}, 1);  γ(0)=0"
        return "γ(h) = -"

    def latex(self, numeric: bool = True) -> str:
        """The fitted variogram as a matplotlib-mathtext string (``$...$``)."""
        if numeric:
            c0 = f"{self.nugget:.4g}"
            c = f"{max(self.sill - self.nugget, 0.0):.4g}"
            a = f"{self.rng:.0f}"
        else:
            c0, c, a = "C_0", "C", "a"
        if self.model == "spherical":
            return (rf"$\gamma(h)={c0}+{c}\left[\frac{{3}}{{2}}\frac{{h}}{{{a}}}"
                    rf"-\frac{{1}}{{2}}\left(\frac{{h}}{{{a}}}\right)^3\right],\ h<{a}$")
        if self.model == "exponential":
            return rf"$\gamma(h)={c0}+{c}\left[1-e^{{-3h/{a}}}\right]$"
        if self.model == "gaussian":
            return rf"$\gamma(h)={c0}+{c}\left[1-e^{{-3(h/{a})^2}}\right]$"
        if self.model == "linear":
            return rf"$\gamma(h)={c0}+{c}\,\mathrm{{min}}(h/{a},\,1)$"
        return r"$\gamma(h)$"


def experimental_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count experimental semivariogram ``(lag_centres, gamma)``.

    Rather than a fixed number of equal-width lag classes, the lag axis (up to
    ``L/2`` — half the maximum pair distance, i.e. ~half the study-area
    diagonal) is split into the **largest** number of bins such that *every* bin
    still contains at least ``min_pairs`` data pairs, following the reliability
    rule of thumb of Bostan (2017).  Each bin then holds an (almost) equal number
    of pairs; its lag centre is the **mean** pair distance in the bin and its
    semivariance the mean of ``½·(v_i − v_j)²`` over those pairs.

    Returns empty arrays when there are too few pairs to form even one bin of
    ``min_pairs`` pairs.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    min_pairs = max(2, int(min_pairs))
    if n < 4:
        return np.empty(0), np.empty(0)
    D = cdist(coords, coords)
    iu = np.triu_indices(n, k=1)
    h = D[iu]
    g = 0.5 * (values[iu[0]] - values[iu[1]]) ** 2
    # Keep only pairs within ~half the maximum lag (the classic L/2 rule of
    # thumb); beyond it the pair counts collapse and gamma*(h) is unreliable.
    hmax = 0.5 * float(h.max())
    if hmax <= 0:
        return np.empty(0), np.empty(0)
    sel = h <= hmax
    h, g = h[sel], g[sel]
    order = np.argsort(h, kind="mergesort")        # sort pairs by lag distance
    h, g = h[order], g[order]
    N = len(h)
    # Optimal bin count: as many equal-count bins as possible while each still
    # holds >= min_pairs pairs.  n_bins = floor(N / min_pairs) guarantees every
    # (almost) equal split has at least min_pairs pairs.
    n_bins = N // min_pairs
    if n_bins < 1:
        return np.empty(0), np.empty(0)
    centres, gamma_exp = [], []
    for grp in np.array_split(np.arange(N), n_bins):
        if grp.size == 0:
            continue
        centres.append(float(np.mean(h[grp])))     # mean lag of the bin
        gamma_exp.append(float(np.mean(g[grp])))   # mean semivariance
    return np.asarray(centres), np.asarray(gamma_exp)


def lag_bin_summary(
    coords: np.ndarray,
    values: np.ndarray,
    min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> list:
    """Per-bin detail of the equal-count experimental-semivariogram binning.

    Uses exactly the same binning as :func:`experimental_variogram` and returns a
    list of ``(lag_lo, lag_hi, n_pairs)`` — the smallest and largest lag distance
    that fall in each bin (same length unit as *coords*) and how many well pairs
    it contains.  Returns an empty list when there are too few pairs to form even
    one bin of ``min_pairs`` pairs.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    min_pairs = max(2, int(min_pairs))
    if n < 4:
        return []
    D = cdist(coords, coords)
    iu = np.triu_indices(n, k=1)
    h = D[iu]
    hmax = 0.5 * float(h.max())
    if hmax <= 0:
        return []
    h = np.sort(h[h <= hmax], kind="mergesort")
    N = len(h)
    n_bins = N // min_pairs
    if n_bins < 1:
        return []
    out = []
    for grp in np.array_split(np.arange(N), n_bins):
        if grp.size == 0:
            continue
        seg = h[grp]
        out.append((float(seg[0]), float(seg[-1]), int(seg.size)))
    return out


def _fit_isotropic(
    coords: np.ndarray, values: np.ndarray, model: str,
    min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> VariogramModel:
    """Isotropic (omnidirectional) fit: nugget, sill and range by least squares."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)

    var = float(np.var(values)) if n > 1 else 1.0
    if var <= 0:
        var = 1.0
    span = float(np.hypot(*(coords.max(0) - coords.min(0)))) if n > 1 else 1.0
    default = VariogramModel(model, nugget=0.0, sill=var, rng=max(span / 3.0, 1.0))
    if n < 4:
        return default

    centres, gamma_exp = experimental_variogram(coords, values, min_pairs)
    if len(centres) < 3:
        return default

    try:
        from scipy.optimize import curve_fit

        def f(hh, nugget, psill, rng):
            return _model_gamma(hh, nugget, nugget + psill, rng, model)

        p0 = [0.0, var, span / 3.0]
        bounds = ([0.0, 0.0, span * 1e-3], [var * 2 + 1e-9, var * 4 + 1e-9, span * 2 + 1e-9])
        popt, _ = curve_fit(f, centres, gamma_exp, p0=p0, bounds=bounds, maxfev=10000)
        nugget, psill, rng = popt
        fitted = VariogramModel(model, nugget=float(nugget), sill=float(nugget + psill), rng=float(rng))
        if not np.isfinite(fitted.sill) or fitted.sill <= 0 or fitted.rng <= 0:
            return default
        return fitted
    except Exception:  # noqa: BLE001 - robust fallback
        return default


def fit_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    model: str = "spherical",
    min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> VariogramModel:
    """Fit an **isotropic** theoretical variogram (nugget, sill, range)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    return _fit_isotropic(coords, values, model, min_pairs)


def fit_isotropic_model(
    coords, values, model: str = "spherical", min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> VariogramModel:
    """Fit a purely **isotropic** model (alias of :func:`fit_variogram`)."""
    return fit_variogram(coords, values, model, min_pairs)


def krige_with_variogram(
    coords, values, targets, variogram: VariogramModel,
    clip_min: Optional[float] = None, clip_max: Optional[float] = None,
) -> "KrigingResult":
    """Ordinary kriging with a **pre-fitted** variogram model."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(values) == 0:
        return KrigingResult(
            np.full(len(targets), np.nan), np.full(len(targets), np.nan), variogram)
    ok = OrdinaryKriging(coords, values, variogram=variogram)
    res = ok.predict(targets)
    if clip_min is not None or clip_max is not None:
        res.values = np.clip(res.values, clip_min, clip_max)
    return res


@dataclass
class VariogramSuggestion:
    """Result of matching every candidate model against the data.

    ``rmse`` / ``mae`` / ``mape`` map each model name to its fit error against
    the experimental points (lower is better).  ``best`` is the *recommendation*
    — the model that is lowest on a majority (≥2) of the three metrics — and
    ``ranking`` lists the models with ``best`` first, the rest ordered by the
    combined Borda consensus.  The user remains free to pick any model.
    """

    lags: np.ndarray                       # experimental lag centres
    gamma: np.ndarray                      # experimental semivariance
    fits: Dict[str, "VariogramModel"]      # model name -> fitted semivariogram
    mse: Dict[str, float]                  # model name -> mean squared error
    rmse: Dict[str, float]                 # model name -> root-mean-square error
    mae: Dict[str, float]                  # model name -> mean absolute error
    mape: Dict[str, float]                 # model name -> mean abs % error (%)
    r2: Dict[str, float]                   # model name -> coefficient of determination
    ranking: List[str]                     # models ordered best -> worst
    best: str
    n_samples: int

    def describe_best(self) -> str:
        if not self.fits:
            return "Not enough well samples to match a semivariogram."
        b = self.best
        mape = self.mape.get(b, float("nan"))
        mape_txt = "n/a" if not np.isfinite(mape) else f"{mape:.1f}%"
        r2 = self.r2.get(b, float("nan"))
        r2_txt = "n/a" if not np.isfinite(r2) else f"{r2:.4f}"
        return (f"Recommended: {b.capitalize()} (RMSE "
                f"{self.rmse.get(b, float('nan')):.4g}, MAE "
                f"{self.mae.get(b, float('nan')):.4g}, MAPE {mape_txt}, "
                f"R2 {r2_txt}).")


def _rank_models(models: list, metrics: list) -> List[str]:
    """Order models best -> worst by a Borda consensus over several metrics.

    *metrics* is a list of ``(values, higher_is_better)`` pairs — e.g. MSE/RMSE/
    MAE/MAPE (lower is better) and R²/adjusted-R² (higher is better).  Each metric
    casts a vote by ranking the models from best (rank 0) to worst; the model with
    the smallest summed rank wins.  The consensus is immune to the different scales
    of the metrics.  A model missing a finite value for a metric is pushed to the
    back of that vote.  The first metric breaks ties.
    """
    if not metrics:
        return list(models)
    gate = metrics[0][0]                              # validity gate (e.g. MSE)
    rankable = [m for m in models if np.isfinite(gate.get(m, np.inf))]
    if not rankable:
        return list(models)
    score = {m: 0.0 for m in rankable}
    for values, higher_is_better in metrics:
        fin = [m for m in rankable if np.isfinite(values.get(m, np.inf))]
        order = sorted(fin, key=lambda x: values[x], reverse=higher_is_better)
        for rank, mm in enumerate(order):
            score[mm] += rank
        for mm in rankable:                          # penalise a missing metric
            if mm not in fin:
                score[mm] += len(rankable)
    return sorted(rankable, key=lambda mm: (score[mm], gate.get(mm, np.inf)))


def suggest_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    models: tuple = VARIOGRAM_MODELS,
    min_pairs: int = _MIN_PAIRS_PER_LAG,
) -> VariogramSuggestion:
    """Fit every candidate *model* and rank them by experimental-fit error.

    This is the data-matching step: it inspects the property's spatial
    distribution (its experimental semivariogram) and recommends the model that
    is collectively best across five goodness-of-fit metrics — MSE, RMSE, MAE and
    MAPE (lower is better) plus R² (higher is better) — via a Borda consensus.
    Purely advisory - the caller decides what to use.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    lags, gamma = experimental_variogram(coords, values, min_pairs)
    fits: Dict[str, VariogramModel] = {}
    mse: Dict[str, float] = {}
    rmse: Dict[str, float] = {}
    mae: Dict[str, float] = {}
    mape: Dict[str, float] = {}
    r2: Dict[str, float] = {}
    K = len(lags)
    # Total sum of squares of the experimental points (same for every model);
    # used for R².
    ss_tot = float(np.sum((gamma - gamma.mean()) ** 2)) if K else 0.0
    for m in models:
        vm = fit_variogram(coords, values, model=m, min_pairs=min_pairs)
        fits[m] = vm
        if K:
            err = vm.gamma(lags) - gamma
            ss_res = float(np.sum(err ** 2))
            mse[m] = ss_res / K
            rmse[m] = float(np.sqrt(mse[m]))
            mae[m] = float(np.mean(np.abs(err)))
            # MAPE relative to the experimental semivariance; skip lags where
            # gamma ≈ 0 so the percentage does not blow up.
            nz = np.abs(gamma) > 1e-12
            mape[m] = (float(np.mean(np.abs(err[nz] / gamma[nz])) * 100.0)
                       if np.any(nz) else float("inf"))
            # R² = 1 − SS_res/SS_tot.
            r2[m] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        else:
            mse[m] = rmse[m] = mae[m] = mape[m] = float("inf")
            r2[m] = float("nan")
    # Recommendation = Borda consensus over the five metrics (errors ascending,
    # R² descending); the recommended model heads the ranking.
    metrics = [(mse, False), (rmse, False), (mae, False), (mape, False),
               (r2, True)]
    ranking = _rank_models(list(models), metrics)
    best = ranking[0] if ranking else (models[0] if models else "spherical")
    return VariogramSuggestion(
        lags=lags, gamma=gamma, fits=fits, mse=mse, rmse=rmse, mae=mae,
        mape=mape, r2=r2, ranking=ranking, best=best,
        n_samples=len(values),
    )


@dataclass
class KrigingResult:
    values: np.ndarray             # (T,) estimates at the targets
    variance: np.ndarray           # (T,) kriging variance
    model: VariogramModel


class OrdinaryKriging:
    """Ordinary kriging estimator with a fitted/﻿supplied variogram."""

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        model: str = "spherical",
        variogram: Optional[VariogramModel] = None,
    ):
        self.coords = np.asarray(coords, dtype=float).reshape(-1, 2)
        self.values = np.asarray(values, dtype=float).ravel()
        if len(self.coords) != len(self.values):
            raise ValueError("coords and values length mismatch")
        self.n = len(self.values)
        self.variogram = variogram or fit_variogram(self.coords, self.values, model=model)
        self._prepare()

    def _prepare(self) -> None:
        n = self.n
        if n == 0:
            return
        # Build and LU-factorise the augmented sample matrix once.
        D = cdist(self.coords, self.coords)
        G = self.variogram.gamma(D)
        A = np.ones((n + 1, n + 1), dtype=float)
        A[:n, :n] = G
        A[n, n] = 0.0
        self._A = A
        try:
            from scipy.linalg import lu_factor

            self._lu = lu_factor(A)
            self._use_lu = True
        except Exception:  # noqa: BLE001
            self._use_lu = False

    def predict(self, targets: np.ndarray) -> KrigingResult:
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        T = len(targets)
        if self.n == 0:
            return KrigingResult(np.full(T, np.nan), np.full(T, np.nan), self.variogram)
        if self.n == 1:
            return KrigingResult(
                np.full(T, self.values[0]), np.full(T, np.nan), self.variogram
            )

        Dt = cdist(self.coords, targets)              # (n, T)
        G0 = self.variogram.gamma(Dt)                 # (n, T)
        rhs = np.ones((self.n + 1, T), dtype=float)
        rhs[:self.n, :] = G0

        if self._use_lu:
            from scipy.linalg import lu_solve

            sol = lu_solve(self._lu, rhs)             # (n+1, T)
        else:
            sol = np.linalg.lstsq(self._A, rhs, rcond=None)[0]

        weights = sol[:self.n, :]                     # (n, T)
        mu = sol[self.n, :]                           # (T,)
        est = weights.T @ self.values                 # (T,)
        var = np.einsum("it,it->t", weights, G0) + mu
        var = np.clip(var, 0.0, None)
        return KrigingResult(values=est, variance=var, model=self.variogram)


def krige_property(
    coords: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    model: str = "spherical",
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> KrigingResult:
    """Convenience wrapper: fit + predict, optionally clamping the estimates.

    With fewer than two samples it degrades gracefully to a constant field
    (the single value, or NaN when there are none).
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(values) == 0:
        return KrigingResult(
            np.full(len(targets), np.nan), np.full(len(targets), np.nan),
            VariogramModel(model, 0.0, 1.0, 1.0),
        )
    ok = OrdinaryKriging(coords, values, model=model)
    res = ok.predict(targets)
    if clip_min is not None or clip_max is not None:
        res.values = np.clip(res.values, clip_min, clip_max)
    return res
