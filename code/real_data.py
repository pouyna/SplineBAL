"""
real_data.py
============
Apply the frozen surrogate to a MEASURED saturation curve.

Everything in the accompanying study is synthetic by design, so this
script is the bridge to real data. It takes a two-column table of
(exposure, response), fits the same I-spline surrogate with the same priors and
the same log-spaced knots used throughout the benchmark, and reports the full
read-out: the level crossing, the cost-indexed quantile family x_q that the
method recommends reporting (pick q from your cost ratio), the
one-sided upper limit, the response-domain risk curve, and the two diagnostics
that say whether the answer should be trusted at all.

Nothing here is fitted to the data except the surrogate: the knot geometry and
the priors are exactly those of the benchmark. The sampler uses the prior-median
initialization (see design_choices.py sampler) rather than the benchmark's default
uniform start -- the two give the same estimates, and the former cuts the
divergence rate from 11-22% to under 5%.

Usage
-----
    python3 real_data.py curve.csv [--alpha=0.95] [--out=prefix] [--robust]

--robust swaps in the Student-t(nu=4) likelihood recommended for field data
where an occasional bad metrology reading is plausible.

`curve.csv` has a header line and two columns, exposure and response, e.g.

    pulse_time_s,gpc_A_per_cycle
    0.020,0.041
    0.033,0.062
    ...

Replicate measurements at the same exposure are supported and encouraged:
repeat the row. The largest exposure in the file anchors the plateau, so it
should be a genuine plateau measurement, and replicating it is the cheapest
robustness available (see robustness.py failure).

Reducing in-situ ellipsometry to a saturation curve
---------------------------------------------------
If the raw measurement is a thickness-versus-time trace per run rather than a
growth per cycle, the reduction needs the CYCLE COUNT of each run, which is not
recoverable from the trace alone when runs differ in length. Compute

    GPC = (thickness at end of steady growth - thickness at start) / n_cycles

per run, over the steady-growth portion only (excluding any nucleation
transient), and write the resulting (pulse_time, GPC) pairs into the CSV above.
Do not infer n_cycles from the trace duration unless the recipe timing is known
to have been identical across runs.

Output
------
Console summary plus, with --out, a PNG of the fit and the risk curve and a
JSON of every reported quantity.
"""
import os
import sys
import csv
import json

import numpy as np

import ispline_model as M
import readout as RO

ALPHA_DEFAULT = 0.95
# The read-out family the method recommends reporting: x_q is the
# q-th percentile of the crossing posterior, and q/(1-q) is the ratio of the
# cost of under-saturating by one second to the cost of over-dosing by one.
QS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# Benchmark reference values quoted in the console report. They are printed for
# orientation only; recompute_stats.py recomputes all of them from the caches.
BAND_COV = 94                    # two-sided x_sat interval coverage (recompute_stats readout)
R_HEALTHY_LO, R_HEALTHY_HI = 0.5, 17.9  # r over the final octave, saturating windows
R_CALIBRATION = [               # (x_max / true x_sat, typical r, false-alarm rate)
    ("1.25x", 0.141, 32), ("1.75x", 0.095, 18),
    ("2.5x", 0.054, 2), ("4x", 0.034, 2)]


def read_csv(path):
    xs, ys = [], []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        rows = [r for r in rd if r and not r[0].lstrip().startswith("#")]
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1                      # header line
    for r in rows[start:]:
        xs.append(float(r[0]))
        ys.append(float(r[1]))
    return np.asarray(xs, float), np.asarray(ys, float)


def fit_measured(x, y, alpha=ALPHA_DEFAULT, seed=0, n_grid=M.EVAL_N, robust=False):
    """Fit the frozen surrogate to measured data and return the full read-out.

    The response is rescaled by the largest observed value so that the priors
    act on their intended scale; everything is reported
    back in the original units. The window is taken from the data.

    robust=True swaps the Gaussian likelihood for a Student-t (nu=4), which the
    study recommends for field data where an occasional bad reading is plausible
    (see refits.py studentt): it roughly halves a contaminated
    plateau anchor's pull on the read-out and costs almost nothing on clean
    data. Monotonicity and f(0)=0 are untouched, because they live in the prior.
    """
    if x.min() <= 0:
        raise ValueError("exposures must be positive; f(0)=0 is imposed by the "
                         "basis and the origin must not be an observation")
    scale = float(np.max(y))
    if scale <= 0:
        raise ValueError("responses must have a positive maximum")
    domain = (float(x.min()), float(x.max()))
    cls = M.StudentTRegressor if robust else M.ISplineRegressor
    # Prior-median initialization and a 0.95 target acceptance: the configuration
    # recommended by the sampler diagnostics (design_choices.py sampler).  The campaigns were run at
    # NumPyro's defaults, which give the same answers but pay a 11-22% divergence
    # rate escaping the prior's tail; there is no reason to inherit that here.
    reg = cls(domain=domain, n_init=len(x), n_iters=0,
              target_accept=M.TARGET_ACCEPT_RECOMMENDED, init_median=True)
    for xi, yi in zip(x, y / scale):
        reg.add(float(xi), float(yi))
    reg.fit(seed=seed)

    # Reporting grid: LOG-spaced, as in every campaign. The basis is smooth in
    # log, and a linear grid at this size is coarser than x_min and than the
    # innermost knot, so it resolves the near-onset crossing badly.
    tg = M.eval_points(0.0, domain[1], n_grid)
    draws = reg.sample(tg)
    rep = RO.readout_report(draws, tg, alpha=alpha, eps=0.05)
    theta = RO.attained_fraction(draws)
    xs = np.array([RO._interp_crossing(tg, theta[s], alpha)
                   for s in range(theta.shape[0])])
    xs = xs[np.isfinite(xs)]
    qsweep = RO.quantile_sweep(draws, tg, alpha, QS)

    out = dict(
        n_obs=int(len(x)), domain=list(domain), plateau_scale=scale,
        alpha=alpha,
        x_sat_median=rep["x_sat_median"],
        x_sat_upper95=float(np.quantile(xs, 0.95)) if xs.size else float("inf"),
        x_sat_95ci=list(rep["x_sat_95ci"]),
        x_risk_bounded=rep["x_risk_bounded"],
        amplification_s_per_unit=rep["amplification_s_per_unit"],
        f_plateau=float(draws[:, -1].mean() * scale),
        sigma_inferred=float(np.mean(reg._sigma) * scale),
        window_sufficiency=rep["window_sufficiency"],
        r_last_octave=float(np.nanmean(
            (draws[:, -1] - draws[:, int(np.argmin(np.abs(tg - tg[-1] / 2)))])
            / draws[:, -1])),
        plateau_leverage=rep["plateau_leverage"],
        n_at_xmax=int(np.sum(np.isclose(x, x.max()))),
        quantile_sweep=dict(q=list(qsweep["q"]), x_q=list(qsweep["x_q"]),
                            theta=list(qsweep["theta_mean"]),
                            risk=list(qsweep["risk"])),
        robust=bool(robust),
    )
    return reg, tg, draws, scale, out


def report(out):
    a = out["alpha"]
    print(f"\n=== frozen I-spline surrogate on {out['n_obs']} measured points "
          f"===")
    print(f"  window                       [{out['domain'][0]:g}, "
          f"{out['domain'][1]:g}]")
    print(f"  inferred plateau f(x_max)    {out['f_plateau']:.4g} "
          f"(response units)")
    print(f"  inferred noise scale sigma   {out['sigma_inferred']:.4g}")
    print(f"\n  --- recommendation at alpha = {a} ---")
    print(f"  median level crossing        {out['x_sat_median']:.3f}   "
          f"<-- q = 0.5 of the sweep below")
    print(f"  one-sided 95% upper limit    {out['x_sat_upper95']:.3f}   "
          f"(q = 0.95; deliberately conservative)")
    print(f"  risk-bounded (P(under)<=5%)  {out['x_risk_bounded']:.3f}")
    print(f"  two-sided 95% interval       [{out['x_sat_95ci'][0]:.3f}, "
          f"{out['x_sat_95ci'][1]:.3f}]   ({BAND_COV}% coverage on the benchmark)")
    print(f"  amplification 1/mu'(x_hat)   {out['amplification_s_per_unit']:.1f}"
          f"   (exposure units per unit response)")

    print(f"\n  --- what to report: the cost-indexed quantile ---")
    print(f"  x_q is the q-th percentile of the crossing posterior, and")
    print(f"      q/(1-q) = (cost of under-saturating by 1 s)/(cost of "
          f"over-dosing by 1 s),")
    print(f"  so pick q from a cost ratio you can judge, then quote x_q with the")
    print(f"  under-saturation risk beside it. There is no single right q.")
    qs = out["quantile_sweep"]
    print(f"  {'q':>10s}{'cost ratio':>12s}{'x_q':>10s}{'attained':>11s}"
          f"{'risk':>8s}")
    for i, q in enumerate(qs["q"]):
        print(f"  {q:>10g}{q / (1 - q):>12.1f}{qs['x_q'][i]:>10.3f}"
              f"{qs['theta'][i]:>11.3f}{qs['risk'][i]:>8.2f}")

    print(f"\n  --- should this answer be trusted? ---")
    r = out["r_last_octave"]
    flag = "OK" if r < 0.15 else "WARNING"
    print(f"  [{flag}] window sufficiency: the fit still gains {100*r:.1f}% of "
          f"its plateau over the")
    print(f"        final octave of exposure. On ground truths known to "
          f"saturate inside the")
    print(f"        window this is {R_HEALTHY_LO:.1f}-{R_HEALTHY_HI:.1f}%; a "
          f"truncated window gives far more (see robustness.py failure).")
    print(f"        The r>15% tolerance is NOT a fixed operating point: r is "
          f"itself inflated")
    print(f"        by a gradual plateau approach, so a tight (efficient) "
          f"window false-alarms")
    print(f"        more often. Calibration against window generosity "
          f"(robustness.py window):")
    print(f"          {'x_max/x_sat':>12s}{'typical r':>11s}"
          f"{'false alarms at r>0.15':>25s}")
    for ratio, r_typ, fa in R_CALIBRATION:
        print(f"          {ratio:>12s}{r_typ:>11.3f}{fa:>24.0f}%")
    n = out["n_at_xmax"]
    flag = "OK" if n >= 2 else "WARNING"
    print(f"  [{flag}] plateau anchor: {n} measurement(s) at x_max. The "
          f"read-out shifts by")
    print(f"        {out['plateau_leverage']:.3f} exposure units per 1% error "
          f"in that anchor, so a single")
    print(f"        bad reading moves the recommendation. Replicate it.")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    path = argv[0]
    alpha, prefix, robust = ALPHA_DEFAULT, None, False
    for a in argv[1:]:
        if a.startswith("--alpha="):
            alpha = float(a.split("=", 1)[1])
        if a == "--robust":
            robust = True
        if a.startswith("--out="):
            prefix = a.split("=", 1)[1]
    x, y = read_csv(path)
    reg, tg, draws, scale, out = fit_measured(x, y, alpha=alpha,
                                             robust=robust)
    report(out)

    if prefix:
        with open(prefix + ".json", "w") as fh:
            json.dump(out, fh, indent=2)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mu, sd = draws.mean(0) * scale, draws.std(0, ddof=1) * scale
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.0),
                                     layout="constrained")
        a1.fill_between(tg, mu - 1.96 * sd, mu + 1.96 * sd, alpha=0.2,
                        color="C0")
        a1.plot(tg, mu, "C0-", lw=1.8, label="posterior mean")
        a1.plot(x, y, "o", color="C3", ms=5, mec="k", mew=0.4,
                label="measurements")
        a1.axvline(out["x_sat_median"], color="C2", ls="-", lw=1.2,
                   label="median crossing (q=0.5)")
        a1.axvline(out["x_sat_upper95"], color="C2", ls="--", lw=1.2,
                   label="one-sided 95% limit (q=0.95)")
        a1.set_xlabel("exposure")
        a1.set_ylabel("response")
        a1.legend(fontsize=7)
        a1.set_title("(a) monotone fit", fontweight="bold")
        a2.plot(tg, RO.risk_curve(draws, alpha), "C1-", lw=1.8)
        a2.axhline(0.05, color="0.5", ls=":", lw=1.0)
        a2.set_xlabel("candidate exposure")
        a2.set_ylabel(f"P(under-saturates {alpha:g})")
        a2.set_title("(b) under-saturation risk", fontweight="bold")
        for ax in (a1, a2):
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(prefix + ".png", dpi=300, bbox_inches="tight")
        print(f"\nwrote {prefix}.json and {prefix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
