"""
readout.py
==========
Response-domain read-out, decision rule, conditioning, and failure-mode
diagnostics for the shape-constrained I-spline active-learning loop.

Everything here is a reduction of the (S, G) posterior draw array

    draws[s, g] = f^(s)(x_g)

that the loop already forms to compute mu_*(x) and sigma_*(x), evaluated on
an ascending reporting grid x with x[-1] = x_max.
No new sampling and no new model.

Motivation.  The level-crossing read-out x_sat(alpha) inverts the fitted curve,
and the conditioning of that inversion is 1/f'(x_alpha), which is large by
definition at alpha near one.  The same fit error is therefore small measured
in the response and large measured in the input.  Group A reports it in the
response domain; group B turns the choice of input into a decision under
an explicit cost (the cost-indexed quantile); group C measures the amplification itself; group D flags the
two ways the plateau anchor can silently fail, and supplies the magnitude
metrics used for the unconstrained Gaussian-process arm.

Groups
------
A  attained_fraction, risk_curve, input_from_risk
B  input_from_quantile, quantile_sweep
C  level_crossing_median, conditioning, readout_report
D  window_sufficiency, plateau_leverage, negative_variation, upcrossings

Crossings are located by linear interpolation between grid nodes, matching
ISplineRegressor._crossings, so the read-out here and in the main campaign
agree to floating point.
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "attained_fraction", "risk_curve", "input_from_risk",
    "input_from_quantile", "quantile_sweep",
    "level_crossing_median", "conditioning", "readout_report",
    "window_sufficiency", "plateau_leverage",
    "negative_variation", "upcrossings",
]


def _interp_crossing(x, y, level):
    """First x at which the sampled curve y reaches `level` (linear interp).

    Returns np.inf if the level is never reached.  Mirrors
    ISplineRegressor._crossings for a single curve.
    """
    idx = np.flatnonzero(y >= level)
    if idx.size == 0:
        return np.inf
    i = int(idx[0])
    if i == 0:
        return float(x[0])
    y0, y1 = y[i - 1], y[i]
    if y1 <= y0:
        return float(x[i - 1])
    return float(x[i - 1] + (level - y0) * (x[i] - x[i - 1]) / (y1 - y0))


# ----------------------------------------------------------------------
# A. Read-out in the response domain
# ----------------------------------------------------------------------

def attained_fraction(draws):
    """theta^(s)(x) = f^(s)(x) / f^(s)(x_max), per draw, shape (S, G).

    Each draw is referenced to its own plateau, so plateau uncertainty stays
    inside the posterior instead of being collapsed to a point estimate first.
    """
    draws = np.asarray(draws, float)
    plateau = draws[:, -1][:, None]
    return draws / np.where(plateau > 0, plateau, np.nan)


def _below(theta, alpha):
    """theta < alpha as floats, keeping NaN (a draw whose plateau is not
    positive) as NaN so that nanmean excludes it instead of counting it as
    saturated. Unreachable for the I-spline, whose plateau is always positive,
    but the same functions are applied to unconstrained GP draws."""
    theta = np.asarray(theta, float)
    return np.where(np.isnan(theta), np.nan, (theta < alpha).astype(float))


def risk_curve(draws, alpha=0.95):
    """R(x) = P(theta(x) < alpha) at every grid point, shape (G,).

    For a candidate input x, the posterior probability that the response
    under-saturates there.  Non-increasing in x because every draw is monotone.
    """
    return np.nanmean(_below(attained_fraction(draws), alpha), axis=0)


def input_from_risk(draws, x, alpha=0.95, eps=0.05):
    """Smallest input whose under-saturation risk is at most eps.

    The benchmark-dose one-sided limit stated in the response domain.
    Returns np.inf if no grid point meets the tolerance, which is the
    correct answer: the window does not contain a defensible input.
    """
    x = np.asarray(x, float)
    ok = np.flatnonzero(risk_curve(draws, alpha) <= eps)
    return float(x[ok[0]]) if ok.size else float("inf")


# ----------------------------------------------------------------------
# B. Decision-theoretic read-out
# ----------------------------------------------------------------------

def input_from_quantile(draws, x, alpha=0.95, q=0.5):
    """The q-th percentile of the crossing posterior of x_sat(alpha).

    This is the Bayes act under pinball (quantile) loss on x_sat, and it is the
    single read-out family the method recommends: q = 0.5 is the median crossing,
    q = 0.95 the conservative one-sided limit, and q/(1-q) is the ratio of the
    cost of under-saturating by one second to the cost of over-dosing by one
    second. The survival function R (risk_curve) generates all of
    them: R(x_q) = 1 - q.
    """
    x = np.asarray(x, float)
    theta = attained_fraction(draws)
    xs = np.array([_interp_crossing(x, theta[s], alpha)
                   for s in range(theta.shape[0])])
    xs = xs[np.isfinite(xs)]
    return float(np.quantile(xs, q)) if xs.size else float("inf")


def quantile_sweep(draws, x, alpha=0.95,
                   qs=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    """input_from_quantile over a range of q, with the attained fraction and
    under-saturation risk each recommendation carries."""
    x = np.asarray(x, float)
    theta = attained_fraction(draws)
    out = {"q": np.asarray(qs, float), "x_q": [], "theta_mean": [], "risk": []}
    for q in out["q"]:
        xq = input_from_quantile(draws, x, alpha, q)
        j = int(np.argmin(np.abs(x - xq))) if np.isfinite(xq) else -1
        out["x_q"].append(xq)
        out["theta_mean"].append(float(np.nanmean(theta[:, j])) if j >= 0 else np.nan)
        out["risk"].append(float(np.nanmean(_below(theta[:, j], alpha))) if j >= 0 else np.nan)
    for k in ("x_q", "theta_mean", "risk"):
        out[k] = np.asarray(out[k], float)
    return out


# ----------------------------------------------------------------------
# C. Conditioning of the level-crossing inverse problem
# ----------------------------------------------------------------------

def level_crossing_median(draws, x, alpha=0.95):
    """Per-draw first crossing of alpha * f^(s)(x_max): median, sd, quantiles.

    Returns (median, sd, q025, q975, frac_no_crossing).  Draws that never
    cross are excluded from the summary and counted separately.
    """
    x = np.asarray(x, float)
    theta = attained_fraction(draws)
    xs = np.array([_interp_crossing(x, theta[s], alpha)
                   for s in range(theta.shape[0])])
    good = np.isfinite(xs)
    frac_none = float(1.0 - good.mean())
    xs = xs[good]
    if xs.size == 0:
        return (np.inf, np.nan, np.inf, np.inf, 1.0)
    return (float(np.median(xs)), float(xs.std(ddof=1)),
            float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975)),
            frac_none)


def conditioning(draws, x, x_hat):
    """1 / mu_*'(x_hat), the amplification in seconds per unit response.

    Computed from the posterior mean, so it is available on real data where
    the true derivative is unknown.  Large values mean the level crossing is
    ill conditioned and the read-out should be reported in the response
    domain instead.
    """
    x = np.asarray(x, float)
    mu = np.asarray(draws, float).mean(axis=0)
    if not np.isfinite(x_hat):
        return float("inf")
    slope = float(np.interp(x_hat, x, np.gradient(mu, x)))
    return float("inf") if slope <= 0 else 1.0 / slope


def readout_report(draws, x, alpha=0.95, eps=0.05):
    """Everything a practitioner should see about one recommendation."""
    med, sd, lo, hi, none = level_crossing_median(draws, x, alpha)
    theta = attained_fraction(draws)
    j = int(np.argmin(np.abs(x - med))) if np.isfinite(med) else 0
    return {
        "x_sat_median": med,
        "x_sat_sd": sd,
        "x_sat_95ci": (lo, hi),
        "frac_draws_no_crossing": none,
        "amplification_s_per_unit": conditioning(draws, x, med),
        "x_risk_bounded": input_from_risk(draws, x, alpha, eps),
        "theta_at_x_sat_median": float(np.nanmean(theta[:, j]))
        if np.isfinite(med) else np.nan,
        "window_sufficiency": window_sufficiency(draws, x),
        "plateau_leverage": plateau_leverage(draws, x, alpha),
    }


# ----------------------------------------------------------------------
# D. Failure-mode diagnostics
# ----------------------------------------------------------------------

def window_sufficiency(draws, x, tol=0.05, octaves=1.0):
    """P(the curve has actually plateaued inside the window).

    For each draw, the fraction of the plateau accumulated over the final
    `octaves` doublings of input,

        r^(s) = [f^(s)(x_max) - f^(s)(x_max / 2**octaves)] / f^(s)(x_max),

    and the statistic returned is P(r < tol).  A saturated curve adds almost
    nothing over its last doubling; an unsaturated one adds a large fraction.

    tol is a stated tolerance, not a universal constant: it is calibrated on
    ground truths known to saturate inside the window (robustness.py failure
    and window) and reported with that calibration.
    """
    x = np.asarray(x, float)
    draws = np.asarray(draws, float)
    j = int(np.argmin(np.abs(x - x[-1] / (2.0 ** octaves))))
    top = draws[:, -1]
    r = (top - draws[:, j]) / np.where(top > 0, top, np.nan)
    return float(np.nanmean(r < tol))


def plateau_leverage(draws, x, alpha=0.95):
    """d x_sat / d f(x_max), in seconds per one percent shift of the anchor.

    Re-reads the level crossing with the plateau reference perturbed by
    +/- 1 percent.  The whole read-out rests on the single measurement at
    x_max, so this is that observation's influence on the recommended input.
    """
    x = np.asarray(x, float)
    theta = attained_fraction(draws)

    def med_crossing(level):
        xs = np.array([_interp_crossing(x, theta[s], level)
                       for s in range(theta.shape[0])])
        xs = xs[np.isfinite(xs)]
        return np.median(xs) if xs.size else np.nan

    hi, lo = med_crossing(alpha * 1.01), med_crossing(alpha * 0.99)
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return float("nan")
    return float((hi - lo) / 2.0)


def negative_variation(mean_curve, x, sigma_y=None):
    """Magnitude of the monotonicity violation in one fitted mean curve.

    Replaces the binary "posterior mean is non-monotone" indicator, which
    saturates at 100 percent for a smooth interpolator and therefore has no
    dynamic range.

    Returns total_negative_variation (sum of downward steps, response units),
    max_dip_depth (largest single peak-to-trough descent),
    n_descending_intervals, frac_domain_descending, and, if sigma_y is given,
    dip_exceeds_noise.
    """
    mu = np.asarray(mean_curve, float)
    x = np.asarray(x, float)
    d = np.diff(mu)
    neg = d < 0
    depth = run = 0.0
    for step in d:
        run = run + step if step < 0 else 0.0
        depth = min(depth, run)
    out = {
        "total_negative_variation": float(-d[neg].sum()) if neg.any() else 0.0,
        "max_dip_depth": float(-depth),
        "n_descending_intervals": int(neg.sum()),
        "frac_domain_descending": float(np.diff(x)[neg].sum() / (x[-1] - x[0]))
        if neg.any() else 0.0,
    }
    if sigma_y is not None:
        out["dip_exceeds_noise"] = bool(out["max_dip_depth"] > sigma_y)
    return out


def upcrossings(mean_curve, x, alpha=0.95):
    """Upcrossing structure of the level alpha * f(x_max) for one mean curve.

    Replaces the binary "multi-valued read-out" indicator.  The count says
    whether the read-out is ambiguous; the span says how much that ambiguity
    costs in seconds, which is the operational number.
    """
    mu = np.asarray(mean_curve, float)
    x = np.asarray(x, float)
    above = mu >= alpha * mu[-1]
    up = np.flatnonzero((~above[:-1]) & above[1:]) + 1
    if up.size == 0:
        return {"count": 0, "first": np.inf, "last": np.inf, "span": 0.0}
    return {"count": int(up.size), "first": float(x[up[0]]),
            "last": float(x[up[-1]]), "span": float(x[up[-1]] - x[up[0]])}
