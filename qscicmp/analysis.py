"""Post-processing of trial records: costs to reach an accuracy, reductions, and curves."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qscicmp.checkpoint import exact_points, reach_point

CHEMICAL_ACCURACY = 1.6e-3


@dataclass
class ReachStatistics:
    system: str
    method: str
    n_trials: int
    n_reached: int
    queries: np.ndarray
    t_counts: np.ndarray
    oracle_shares: np.ndarray

    @property
    def median_queries(self) -> float:
        return float(np.median(self.queries)) if self.queries.size else float("nan")

    @property
    def median_t(self) -> float:
        return float(np.median(self.t_counts)) if self.t_counts.size else float("nan")

    @property
    def mean_queries(self) -> float:
        return float(np.mean(self.queries)) if self.queries.size else float("nan")

    @property
    def mean_t(self) -> float:
        return float(np.mean(self.t_counts)) if self.t_counts.size else float("nan")


def reach_statistics(payloads: list[dict], epsilon: float = CHEMICAL_ACCURACY) -> ReachStatistics:
    queries, t_counts, shares = [], [], []
    for payload in payloads:
        point = reach_point(payload, epsilon)
        if point is None:
            continue
        queries.append(point["total_queries"])
        t_counts.append(point["total_t"])
        shares.append(point["total_oracle_t"] / point["total_t"] if point["total_t"] > 0 else 0.0)
    return ReachStatistics(
        system=payloads[0]["system"] if payloads else "", method=payloads[0]["method"] if payloads else "",
        n_trials=len(payloads), n_reached=len(queries),
        queries=np.asarray(queries, dtype=float), t_counts=np.asarray(t_counts, dtype=float), oracle_shares=np.asarray(shares, dtype=float),
    )


def reduction(reference: ReachStatistics, candidate: ReachStatistics, statistic: str = "median") -> tuple[float, float]:
    """Relative reduction of the query count and T-gate count of ``candidate`` with respect to ``reference``."""
    if statistic == "median":
        return 100.0 * (1.0 - candidate.median_queries / reference.median_queries), 100.0 * (1.0 - candidate.median_t / reference.median_t)
    return 100.0 * (1.0 - candidate.mean_queries / reference.mean_queries), 100.0 * (1.0 - candidate.mean_t / reference.mean_t)


def bootstrap_reduction_interval(reference: ReachStatistics, candidate: ReachStatistics, n_samples: int = 2000, seed: int = 0) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% bootstrap interval of the median reductions."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        i = rng.integers(0, reference.queries.size, reference.queries.size)
        j = rng.integers(0, candidate.queries.size, candidate.queries.size)
        samples.append((
            100.0 * (1.0 - np.median(candidate.queries[j]) / np.median(reference.queries[i])),
            100.0 * (1.0 - np.median(candidate.t_counts[j]) / np.median(reference.t_counts[i])),
        ))
    samples = np.asarray(samples)
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def trial_curve(payload: dict, x_key: str, up_to_epsilon: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Energy error at the exact-energy points of one trial, as a function of the cumulative cost ``x_key``."""
    points = exact_points(payload)
    xs, ys = [], []
    for point in points:
        xs.append(float(point[x_key]))
        ys.append(float(point["energy_error"]))
        if up_to_epsilon is not None and point["energy_error"] <= up_to_epsilon:
            break
    return np.asarray(xs), np.asarray(ys)


def median_curve(payloads: list[dict], x_key: str, up_to_epsilon: float | None = None, n_grid: int = 200,
                 interpolate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Median over trials of the energy error on a logarithmic cost grid.

    The energy is evaluated exactly only at some iterations.  With
    ``interpolate`` the curve of each trial connects these points by linear
    interpolation in log-log coordinates, as in the figures of the paper;
    otherwise each trial is a step function holding the last evaluated
    value.  A grid point is kept when at least half of the trials have
    data there.
    """
    curves = [trial_curve(payload, x_key, up_to_epsilon) for payload in payloads]
    curves = [(x[x > 0], np.minimum.accumulate(y)[x > 0]) for x, y in curves if np.any(x > 0)]
    curves = [(x, y) for x, y in curves if x.size]
    if not curves:
        return np.asarray([]), np.asarray([])
    x_min = min(float(x.min()) for x, _ in curves)
    x_max = max(float(x.max()) for x, _ in curves)
    grid = np.geomspace(x_min, x_max, n_grid)
    values = np.full((len(curves), n_grid), np.nan)
    for row, (x, y) in enumerate(curves):
        inside = (grid >= x[0]) & (grid <= x[-1])
        if interpolate:
            values[row, inside] = np.exp(np.interp(np.log(grid[inside]), np.log(x), np.log(np.maximum(y, 1e-300))))
        else:
            positions = np.searchsorted(x, grid[inside], side="right") - 1
            values[row, inside] = y[positions]
    enough = np.sum(~np.isnan(values), axis=0) >= max(1, (len(curves) + 1) // 2)
    medians = np.nanmedian(values[:, enough], axis=0)
    return grid[enough], medians


@dataclass
class OracleStatistics:
    """Weight of the oracle in the T-gate count, over the trials that reached the accuracy.

    ``cumulative_share`` is the oracle fraction of all T gates spent up to
    the reach point, which is the ratio of the averaged G_oracle / 2 to C in
    Eq. (20) of the paper.  ``reach_share`` is the same fraction within the
    reach iteration alone.  ``cumulative_ratio`` and ``reach_ratio`` give
    the oracle T gates divided by the state-preparation T gates instead.
    """

    system: str
    method: str
    n_reached: int
    cumulative_share: np.ndarray
    cumulative_ratio: np.ndarray
    reach_share: np.ndarray
    reach_ratio: np.ndarray


def oracle_statistics(payloads: list[dict], epsilon: float = CHEMICAL_ACCURACY) -> OracleStatistics:
    cumulative_share, cumulative_ratio, reach_share, reach_ratio = [], [], [], []
    for payload in payloads:
        point = reach_point(payload, epsilon)
        if point is None or point["iteration"] == 0:
            continue
        stateprep = float(point["total_stateprep_t"])
        oracle = float(point["total_oracle_t"])
        cumulative_share.append(oracle / (oracle + stateprep))
        cumulative_ratio.append(oracle / stateprep)
        entry = next(e for e in payload["history"] if e["iteration"] == point["iteration"])
        o, s = float(entry["oracle_increment"]), float(entry["stateprep_increment"])
        reach_share.append(o / (o + s))
        reach_ratio.append(o / s)
    return OracleStatistics(
        system=payloads[0]["system"] if payloads else "", method=payloads[0]["method"] if payloads else "",
        n_reached=len(cumulative_share),
        cumulative_share=np.asarray(cumulative_share), cumulative_ratio=np.asarray(cumulative_ratio),
        reach_share=np.asarray(reach_share), reach_ratio=np.asarray(reach_ratio),
    )
