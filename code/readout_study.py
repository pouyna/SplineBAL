"""
readout_study.py -- the recommended input, the saturation input
x_sat(0.95), and whether its uncertainty is honest. Two subcommands:

  subcommand  what it produces                              cache / consumer
  ----------  ------------------------------------------    ------------------------------
  readout     refit the stored N=20 designs and compute     results_readout.json
              the I-spline x_sat read-out: CI coverage,     (recompute_stats readout, xsat)
              conditioning 1/f', response-domain gap, the
              quantile and risk-bounded read-outs, window
              sufficiency and plateau leverage
  gpxsat      the SAME x_sat read-out for the resolution-   results_gpxsat.json
              matched GP arm (GP-log-pin), so the           (recompute_stats readout, xsat)
              constrained and unconstrained read-outs
              are directly comparable

Usage:  python3 readout_study.py <readout|gpxsat> [--jobs=N] [--seeds=N] [--verify]
Both read results.json (the stored designs); 'gpxsat' also splices the I-spline
column from results_readout.json, so run 'readout' first.
"""
import os
import sys
import json

import numpy as np

import ispline_model as M
import isotherms as ISO
import readout as RO
import gp_arms as GA
import poolutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # result caches live in code/data
DOMAIN = ISO.DOMAIN
TG = M.eval_points(0.0, DOMAIN[1])
FAMILIES = list(ISO.STANDARD_SPECS)
ALPHA = 0.95


def _set_threads():
    poolutil.pin_threads()


# ===========================================================================
# readout -- the I-spline read-out: coverage, conditioning, the quantile family
# ===========================================================================


def _load():
    with open(os.path.join(DATA, "results.json")) as fh:
        return json.load(fh)


def _job_readout(job):
    nm, seed = job
    RES = _load()
    d = RES["mc"][nm]["uncertainty"][seed]
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)

    reg = M.ISplineRegressor(domain=DOMAIN)
    for x, y in zip(d["t_obs"], d["y_obs"]):
        reg.add(x, y)
    reg.fit(seed=seed)
    draws = reg.sample(TG)                       # (S, G)
    mu = draws.mean(0)
    agree = float(np.max(np.abs(mu - np.array(d["mu"]))))

    med, sd, lo, hi, nonec = RO.level_crossing_median(draws, TG, ALPHA)
    theta = RO.attained_fraction(draws)

    # true crossing and true conditioning
    x_true = ISO.true_level_time(f, ALPHA * 1.0)
    h = 1e-6
    fp = (float(f(np.array([x_true + h]))[0])
          - float(f(np.array([max(x_true - h, 0.0)]))[0])) / (2 * h)

    # the same error, read in the response domain, at the TRUE crossing
    jt = int(np.argmin(np.abs(TG - x_true)))
    theta_at_true = float(np.nanmean(theta[:, jt]))

    qsweep = RO.quantile_sweep(draws, TG, ALPHA)
    return job, dict(
        agree=agree,
        q=list(qsweep["q"]), x_q=list(qsweep["x_q"]),
        theta_q=list(qsweep["theta_mean"]), risk_q=list(qsweep["risk"]),
        x_sat_med=med, x_sat_sd=sd, ci_lo=lo, ci_hi=hi,
        frac_nocross=nonec,
        covered=bool(lo <= x_true <= hi),
        covered_onesided=bool(x_true <= float(np.quantile(
            np.array([RO._interp_crossing(TG, theta[s], ALPHA)
                      for s in range(theta.shape[0])]), 0.95))),
        x_true=float(x_true),
        amp_post=RO.conditioning(draws, TG, med),
        amp_true=float(1.0 / fp) if fp > 0 else np.inf,
        theta_at_true=theta_at_true,
        resp_gap=float(ALPHA - theta_at_true),
        x_risk=RO.input_from_risk(draws, TG, ALPHA, 0.05),
        window_suff=RO.window_sufficiency(draws, TG),
        plateau_lev=RO.plateau_leverage(draws, TG, ALPHA),
        r_last_octave=float(np.nanmean(
            (draws[:, -1] - draws[:, int(np.argmin(np.abs(TG - TG[-1] / 2)))])
            / draws[:, -1])),
    )


def run_readout(argv):
    jobs, nseeds, verify = None, 50, False
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a == "--verify":
            verify = True
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_readout.json")
    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))
    if jobs > 1:
        _set_threads()

    print(f"Refitting {len(joblist)} stored N=20 designs on {jobs} workers ...",
          flush=True)
    recs = poolutil.run_pool(_job_readout, joblist, jobs, tasks_per_child=8,
                             label="read-out refits")

    out = {nm: [recs[(nm, s)] for s in seeds] for nm in FAMILIES}
    out["_meta"] = dict(seeds=seeds, alpha=ALPHA)
    with open(cache, "w") as fh:
        json.dump(out, fh)

    def C(nm, k):
        return np.array([r[k] for r in out[nm]], float)

    if verify:
        worst = max(float(C(nm, "agree").max()) for nm in FAMILIES)
        print(f"\nrefit vs cached posterior mean: worst |delta| = {worst:.2e}")

    print("\n=== the recommended input, x_sat(0.95): does its interval cover? ===")
    print(f"  {'family':18s}{'x_true':>8s}{'median':>9s}{'bias':>8s}"
          f"{'post sd':>9s}{'seed sd':>9s}{'95% CI':>18s}{'cover %':>9s}"
          f"{'1-sided %':>11s}")
    for nm in FAMILIES:
        med, sd = C(nm, "x_sat_med"), C(nm, "x_sat_sd")
        print(f"  {nm:18s}{C(nm,'x_true')[0]:>8.2f}{med.mean():>9.2f}"
              f"{med.mean()-C(nm,'x_true')[0]:>+8.2f}{sd.mean():>9.2f}"
              f"{med.std(ddof=1):>9.2f}"
              f"{'['+format(C(nm,'ci_lo').mean(),'.2f')+','+format(C(nm,'ci_hi').mean(),'.2f')+']':>18s}"
              f"{100*C(nm,'covered').mean():>9.0f}"
              f"{100*C(nm,'covered_onesided').mean():>11.0f}")
    allc = np.concatenate([C(nm, "covered") for nm in FAMILIES])
    all1 = np.concatenate([C(nm, "covered_onesided") for nm in FAMILIES])
    print(f"  {'POOLED':18s}{'':>8s}{'':>9s}{'':>8s}{'':>9s}{'':>9s}{'':>18s}"
          f"{100*allc.mean():>9.0f}{100*all1.mean():>11.0f}")

    print("\n=== the same error in two domains, at the true crossing x=2.0 s ===")
    print(f"  {'family':18s}{'theta(2.0)':>12s}{'gap (resp.)':>13s}"
          f"{'gap (time)':>12s}{'amp true':>10s}{'amp post':>10s}")
    for nm in FAMILIES:
        gt = C(nm, "resp_gap").mean()
        dt = C(nm, "x_sat_med").mean() - C(nm, "x_true")[0]
        print(f"  {nm:18s}{C(nm,'theta_at_true').mean():>12.3f}{gt:>13.3f}"
              f"{100*dt/C(nm,'x_true')[0]:>11.0f}%"
              f"{C(nm,'amp_true')[0]:>10.1f}"
              f"{np.median(C(nm,'amp_post')):>10.1f}")

    print("\n=== response-domain and decision read-outs (mean over seeds) ===")
    print(f"  {'family':18s}{'median x':>10s}{'risk-bnd':>10s}")
    for nm in FAMILIES:
        print(f"  {nm:18s}{C(nm,'x_sat_med').mean():>10.2f}"
              f"{C(nm,'x_risk').mean():>10.2f}")

    print("\n=== window sufficiency and plateau leverage on saturating runs ===")
    print(f"  {'family':18s}{'r(last octave)':>16s}{'P(r<0.05)':>12s}"
          f"{'d x_sat/d f(xmax)':>20s}")
    for nm in FAMILIES:
        print(f"  {nm:18s}{C(nm,'r_last_octave').mean():>16.4f}"
              f"{C(nm,'window_suff').mean():>12.2f}"
              f"{C(nm,'plateau_lev').mean():>18.3f} s/%")
    print("\nwrote", cache)


# ===========================================================================
# gpxsat -- the SAME x_sat read-out off the resolution-matched GP arm
# ===========================================================================
# 500 exact draws from the closed-form GP posterior, matching the monotone
# arm's 500 MCMC draws, so every read-out summarizes the same number of
# posterior draws.
S = 500


def _gp_draws(xo, yo, grid, kind, pin, egrid, seed):
    """S posterior samples from the closed-form GP predictive on `grid`."""
    xo = np.asarray(xo, float)
    yo = np.asarray(yo, float)
    if pin:
        xo = np.concatenate([[0.0], xo])
        yo = np.concatenate([[0.0], yo])
    uo = GA.warp(xo, kind)
    ug = GA.warp(grid, kind)
    pin_var = np.zeros(len(uo))
    if pin:
        pin_var[0] = 1.0
    best = (-np.inf, None, None, None)
    for i, ell in enumerate(egrid):
        for j, sf in enumerate(GA.SF_GRID):
            for k, sn in enumerate(GA.SN_GRID):
                nvar = np.where(pin_var > 0, 1e-8, sn**2)
                lml = GA._lml(uo, yo, nvar, ell, sf)
                if lml > best[0]:
                    best = (lml, i, j, k)
    _, i, j, k = best
    ell, sf, sn = float(egrid[i]), float(GA.SF_GRID[j]), float(GA.SN_GRID[k])
    nvar = np.where(pin_var > 0, 1e-8, sn**2)
    ybar = yo.mean()
    K = GA._rbf(uo, uo, ell, sf) + np.diag(nvar) + 1e-10 * np.eye(len(uo))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yo - ybar))
    Ks = GA._rbf(ug, uo, ell, sf)
    mu = ybar + Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    cov = GA._rbf(ug, ug, ell, sf) - v.T @ v
    cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(len(ug))
    rng = np.random.default_rng(10_000 + seed)
    draws = rng.multivariate_normal(mu, cov, size=S)
    return draws                                      # (S, G)


def _job_gpxsat(job):
    nm, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    egrid = GA.ell_grid("log")
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(5, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    for _ in range(15):
        _, sd_a, _, _ = GA._gp_fit(xs, ys, GA.ACQ, "log", True, egrid)
        xn = float(GA.ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    draws = _gp_draws(xs, ys, TG, "log", True, egrid, seed)

    # SAME crossing convention as the I-spline read-out
    plateau = draws[:, -1][:, None]
    theta = draws / np.where(plateau > 0, plateau, np.nan)
    cross = np.array([RO._interp_crossing(TG, theta[s], ALPHA)
                      for s in range(theta.shape[0])])
    good = np.isfinite(cross)
    xg = cross[good]
    x_true = ISO.true_level_time(f, ALPHA)
    # multi-valued: mean curve crosses level upward more than once
    mu = draws.mean(0)
    uc = RO.upcrossings(mu, TG, ALPHA)
    if xg.size < 10:
        return job, dict(x_sat=float("nan"), covered=False,
                         covered_1s=False, multiv=bool(uc["count"] > 1),
                         x_true=float(x_true), bias=float("nan"),
                         frac_nocross=float(1 - good.mean()))
    lo, hi = np.quantile(xg, 0.025), np.quantile(xg, 0.975)
    q95 = np.quantile(xg, 0.95)
    med = float(np.median(xg))
    return job, dict(
        x_sat=med, bias=float(med - x_true), x_true=float(x_true),
        covered=bool(lo <= x_true <= hi),
        covered_1s=bool(x_true <= q95),
        multiv=bool(uc["count"] > 1),
        frac_nocross=float(1 - good.mean()))


def run_gpxsat(argv):
    jobs, nseeds = None, 50
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_gpxsat.json")
    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    recs = poolutil.run_pool(_job_gpxsat, joblist, jobs, tasks_per_child=40,
                             label="GP x_sat runs (no MCMC)")
    out = {nm: [recs[(nm, s)] for s in seeds] for nm in FAMILIES}
    out["_meta"] = dict(seeds=seeds, alpha=ALPHA, arm="gp_log_pin")
    with open(cache, "w") as fh:
        json.dump(out, fh)

    # I-spline reference from results_readout.json
    try:
        RD = json.load(open(os.path.join(DATA, "results_readout.json")))
    except (OSError, ValueError):
        RD = None

    def C(nm, k):
        return np.array([r[k] for r in out[nm]], float)

    print("\n=== the RECOMMENDED INPUT, x_sat(0.95): unconstrained vs constrained ===")
    print(f"  {'family':18s}{'arm':10s}{'bias (s)':>10s}{'2-sided cov':>13s}"
          f"{'1-sided cov':>13s}{'multi-val':>11s}")
    for nm in FAMILIES:
        b = np.nanmean(C(nm, "bias"))
        print(f"  {nm:18s}{'GP-pin':10s}{b:>+10.2f}"
              f"{100*C(nm,'covered').mean():>12.0f}%{100*C(nm,'covered_1s').mean():>12.0f}%"
              f"{100*C(nm,'multiv').mean():>10.0f}%")
        if RD and nm in RD:
            ib = np.mean([r["x_sat_med"] - r["x_true"] for r in RD[nm]])
            ic = np.mean([r["covered"] for r in RD[nm]])
            i1 = np.mean([r["covered_onesided"] for r in RD[nm]])
            print(f"  {'':18s}{'I-spline':10s}{ib:>+10.2f}{100*ic:>12.0f}%"
                  f"{100*i1:>12.0f}%{'0%':>10s}")   # monotone: never multi-valued
    gp_b = np.nanmean(np.concatenate([C(nm, "bias") for nm in FAMILIES]))
    gp_c = np.mean(np.concatenate([C(nm, "covered") for nm in FAMILIES]))
    gp_1 = np.mean(np.concatenate([C(nm, "covered_1s") for nm in FAMILIES]))
    gp_mv = np.mean(np.concatenate([C(nm, "multiv") for nm in FAMILIES]))
    print(f"\n  GP-log-pin POOLED: bias {gp_b:+.2f}s, 2-sided cov {100*gp_c:.0f}%, "
          f"1-sided cov {100*gp_1:.0f}%, multi-valued {100*gp_mv:.0f}%")
    if RD:
        ib = np.mean([r["x_sat_med"] - r["x_true"]
                      for nm in FAMILIES for r in RD[nm]])
        ic = np.mean([r["covered"] for nm in FAMILIES for r in RD[nm]])
        i1p = np.mean([r["covered_onesided"] for nm in FAMILIES for r in RD[nm]])
        print(f"  I-spline   POOLED: bias {ib:+.2f}s, 2-sided cov {100*ic:.0f}%, "
              f"1-sided cov {100*i1p:.0f}%, multi-valued 0% (monotone by construction)")
    print("\nwrote", cache)


RUNNERS = {"readout": run_readout, "gpxsat": run_gpxsat}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in RUNNERS:
        print(f"readout_study.py: unknown subcommand {cmd!r}; "
              f"choose from {list(RUNNERS)}", file=sys.stderr)
        sys.exit(2)
    RUNNERS[cmd](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
