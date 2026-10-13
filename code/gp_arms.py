"""
gp_arms.py
==========
The unconstrained Gaussian-process ablation: three arms that differ
from the monotone I-spline by one design axis at a time, so that reading
across them separates multi-scale resolution and the exact origin from the
monotonicity constraint.  A single-scale kernel on a linear axis is denied
the resolution the I-spline's log-spaced knots supply, so its gap alone says
nothing about the shape constraint; the log-axis and origin-pinned arms
restore those features one at a time.

Four arms share the identical survey design, budget, acquisition rule, noise
realizations, and seeds:

  gp_lin      RBF on x                       (unconstrained, single-scale, linear axis)
  gp_log      RBF on u = log(1 + x/x_min)    (unconstrained, log-resolved)
  gp_log_pin  as gp_log, plus a noiseless pseudo-observation at (x=0, y=0)
  (the monotone I-spline arm is the main campaign, results.json)

Hyperparameters are chosen by maximum marginal likelihood over a grid that is
deliberately wide and verified NON-binding (the script reports the fraction of
fits whose optimum lands on any grid edge; see --check), so no arm is hobbled
by a search-range choice rather than by its kernel.

Recorded per run, in addition to the saturation error at every budget --
magnitudes rather than binaries, so the failure modes stay discriminating:
  * negative_variation  -- total/max magnitude of the monotonicity violation
  * upcrossings         -- count AND span in seconds of the 95% level
  * acquisition harm    -- fraction of active selections placed where the GP
                           posterior mean was locally DECREASING, and within
                           one candidate spacing of a local minimum of it
                           (the failure-mode inventory in recompute_stats gp)

Usage: python3 gp_arms.py [--jobs=N] [--seeds=N] [--check]
Output: results_gparms.json + console summary.

       python3 gp_arms.py matched [--jobs=N]
The matched-design control (recompute_stats gpmatched): GP-log-pin and GP-log fitted
to the I-spline campaign's stored designs (same inputs, same readings, every
strategy and seed), so surrogate accuracy is compared on identical data, plus
the GP arms' own loops re-run to record how flat their bands were and how
their designs cluster.  Output: results_gp_matched.json.
"""
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import ispline_model as M
import isotherms as ISO
import readout as RO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # result caches live in code/data
CACHE = os.path.join(DATA, "results_gparms.json")
DOMAIN = ISO.DOMAIN
N_INIT, N_ITERS = 5, 15
FAMILIES = list(ISO.STANDARD_SPECS)
SIGMA_Y = ISO.SIGMA_Y

TG = M.eval_points(0.0, DOMAIN[1])             # reporting grid (log-spaced, as I-spline)
ACQ = M.acq_points(DOMAIN[0], DOMAIN[1])      # acquisition grid (linear 0.001 s, as I-spline)
ACQ_DX = float(ACQ[1] - ACQ[0])

ARMS = {
    "gp_lin":     dict(warp="linear", pin=False),
    "gp_log":     dict(warp="log",    pin=False),
    "gp_log_pin": dict(warp="log",    pin=True),
}

# Marginal-likelihood search grid, in WARPED units, deliberately wide enough
# that the optimum is interior.  Reported as such by --check.
SF_GRID = np.array([0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.2, 3.5])
SN_GRID = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35])


def warp(x, kind):
    """Input warping.  'log' is log(1 + x/x_min): log-like above x_min, and,
    unlike log(x), defined at the origin, so the pinned arm and the reporting
    grid (which starts at 0) are both representable."""
    x = np.asarray(x, float)
    if kind == "log":
        return np.log1p(x / DOMAIN[0])
    return x


def ell_grid(kind):
    span = float(warp(np.array([DOMAIN[1]]), kind)[0])
    return np.geomspace(span / 400.0, span * 2.0, 26)


def _rbf(a, b, ell, sf):
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :])**2 / ell**2)


def _lml(uo, yo, nvar, ell, sf):
    ybar = yo.mean()
    K = _rbf(uo, uo, ell, sf) + np.diag(nvar) + 1e-10 * np.eye(len(uo))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    a = np.linalg.solve(L, yo - ybar)
    return -0.5 * a @ a - np.log(np.diag(L)).sum()


def _gp_fit(xo, yo, grid, kind, pin, egrid):
    """Marginal-likelihood-optimal RBF GP posterior mean and sd on `grid`.

    Returns (mu, sd, (ell, sf, sn), on_edge).
    """
    xo = np.asarray(xo, float)
    yo = np.asarray(yo, float)
    if pin:                        # noiseless pseudo-observation at the origin
        xo = np.concatenate([[0.0], xo])
        yo = np.concatenate([[0.0], yo])
    uo = warp(xo, kind)
    ug = warp(grid, kind)
    pin_var = np.zeros(len(uo))
    if pin:
        pin_var[0] = 1.0           # marker row: fixed tiny noise, see below

    best = (-np.inf, None, None, None)
    for i, ell in enumerate(egrid):
        for j, sf in enumerate(SF_GRID):
            for k, sn in enumerate(SN_GRID):
                nvar = np.where(pin_var > 0, 1e-8, sn**2)
                lml = _lml(uo, yo, nvar, ell, sf)
                if lml > best[0]:
                    best = (lml, i, j, k)
    _, i, j, k = best
    ell, sf, sn = float(egrid[i]), float(SF_GRID[j]), float(SN_GRID[k])
    on_edge = bool(i in (0, len(egrid) - 1) or j in (0, len(SF_GRID) - 1)
                   or k in (0, len(SN_GRID) - 1))

    nvar = np.where(pin_var > 0, 1e-8, sn**2)
    ybar = yo.mean()
    K = _rbf(uo, uo, ell, sf) + np.diag(nvar) + 1e-10 * np.eye(len(uo))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yo - ybar))
    Ks = _rbf(ug, uo, ell, sf)
    mu = ybar + Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    sd = np.sqrt(np.clip(sf**2 - np.einsum("ij,ij->j", v, v), 1e-12, None))
    return mu, sd, (ell, sf, sn), on_edge


def _acq_harm(mu_acq, x_pick):
    """Did this active selection land on a descending stretch of the GP mean,
    or next to a spurious local minimum of it?"""
    j = int(np.argmin(np.abs(ACQ - x_pick)))
    slope = np.gradient(mu_acq, ACQ)
    descending = bool(slope[j] < 0)
    d = np.diff(mu_acq)
    locmin = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    near_min = bool(locmin.size and np.min(np.abs(ACQ[locmin] - x_pick)) <= ACQ_DX)
    return descending, near_min


def _run_job(job):
    arm, nm, seed = job
    cfg = ARMS[arm]
    kind, pin = cfg["warp"], cfg["pin"]
    egrid = ell_grid(kind)
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    rng = np.random.default_rng(seed)
    yt = f(TG)

    xs = list(M.survey_design(N_INIT, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    hist, ells, n_desc, n_nearmin = [], [], 0, 0
    for _ in range(N_ITERS + 1):
        mu, sd, hyp, edge = _gp_fit(xs, ys, TG, kind, pin, egrid)
        hist.append(M.saturation_error(TG, mu, yt))
        ells.append(hyp[0])
        if len(xs) < N_INIT + N_ITERS:
            mu_a, sd_a, _, _ = _gp_fit(xs, ys, ACQ, kind, pin, egrid)
            xn = float(ACQ[int(np.argmax(sd_a))])
            desc, near = _acq_harm(mu_a, xn)
            n_desc += int(desc)
            n_nearmin += int(near)
            xs.append(xn)
            ys.append(observe(xn, rng))

    # the last pass of the loop (len(xs) == N, nothing added) fitted the
    # complete design, so mu, sd, hyp and edge already describe the final fit
    nv = RO.negative_variation(mu, TG, sigma_y=SIGMA_Y)
    uc = RO.upcrossings(mu, TG, alpha=0.95)
    return job, dict(
        hist=[float(h) for h in hist], sat_err=float(hist[-1]),
        dips=bool(nv["n_descending_intervals"] > 0),
        tot_negvar=nv["total_negative_variation"],
        max_dip=nv["max_dip_depth"],
        dip_gt_noise=nv["dip_exceeds_noise"],
        frac_desc=nv["frac_domain_descending"],
        ncross95=uc["count"], multivalued=bool(uc["count"] > 1),
        span95=uc["span"],
        f0=float(mu[0]),
        ell=float(hyp[0]), sf=float(hyp[1]), sn=float(hyp[2]),
        on_edge=bool(edge), ell_hist=[float(e) for e in ells],
        n_desc=int(n_desc), n_nearmin=int(n_nearmin), n_active=int(N_ITERS),
    )



def _ispline_reference():
    """Pooled I-spline saturation error per family, read from results.json.

    Returns ({family: mean}, pooled_mean); ({}, None) if the cache is absent, so
    the summary still prints before the main campaign has been run.
    """
    path = os.path.join(DATA, "results.json")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path) as fh:
            R = json.load(fh)["mc"]
    except (ValueError, KeyError, OSError):
        return {}, None
    per, allv = {}, []
    for nm in FAMILIES:
        rs = R.get(nm, {}).get("uncertainty")
        if not rs:
            continue
        v = [r["sat_err"] for r in rs]
        per[nm] = float(np.mean(v))
        allv += v
    return per, (float(np.mean(allv)) if allv else None)


MATCHED_CACHE = os.path.join(DATA, "results_gp_matched.json")
STRATEGIES = ["uncertainty", "grid", "random_log", "random"]   # designs stored in results.json


def _matched_fit_job(job):
    """GP arm fitted to one stored I-spline design (same inputs, same readings).
    job = (arm, strat, nm, seed, t_obs, y_obs, ispline_sat_err); the design is passed
    in so the workers do not each reload results.json."""
    arm, strat, nm, seed, t_obs, y_obs, isp_err = job
    cfg = ARMS[arm]
    fam, kw = ISO.STANDARD_SPECS[nm]
    yt = ISO.make_isotherm(fam, **kw)(TG)
    mu, sd, hyp, edge = _gp_fit(np.asarray(t_obs, float), np.asarray(y_obs, float),
                                TG, cfg["warp"], cfg["pin"], ell_grid(cfg["warp"]))
    return job[:4], dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                         ispline_sat_err=float(isp_err), ell=float(hyp[0]))


def _design_stats(xs, flat=None):
    xs = np.sort(np.asarray(xs, float))
    # same conventions as recompute_stats acq / robustness: inputs closer than
    # 0.02 s count as one, and a reading within 0.02 s of x_max is "at x_max"
    d = dict(n_distinct=int(1 + np.sum(np.diff(xs) > 0.02)),
             n_at_xmax=int(np.sum(np.abs(xs - DOMAIN[1]) < 0.02)))
    if flat is not None:
        f = np.asarray(flat, float)          # rows: (flat share, sd_min/sd_max) per acquisition step
        d.update(flat_share=float(f[:, 0].mean()), sd_range=float(f[:, 1].mean()))
    return d


def _own_loop_job(job):
    """Re-run a GP arm's own uncertainty-sampling loop (bitwise the campaign's, same
    seed) and record the design and how flat its band was at each step."""
    arm, nm, seed = job
    cfg = ARMS[arm]
    kind, pin = cfg["warp"], cfg["pin"]
    egrid = ell_grid(kind)
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(N_INIT, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    flat, ells = [], []
    for _ in range(N_ITERS):
        mu_a, sd_a, hyp, _ = _gp_fit(xs, ys, ACQ, kind, pin, egrid)
        # share of the candidate grid whose band is within 10% of the widest, and
        # the narrowest band relative to the widest: both near 1 mean a flat band
        flat.append((float(np.mean(sd_a >= 0.9 * sd_a.max())), float(sd_a.min() / sd_a.max())))
        ells.append(float(hyp[0]))
        xn = float(ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    rec = _design_stats(xs, flat)
    rec["ell_median"] = float(np.median(ells))
    return job, rec


def _matched_truth_job(job):
    """GP arm fitted to one stored design of a curve given by (spec_kind, spec);
    spec_kind 'family' -> (name, sigma) is only bookkeeping, the truth is the family
    curve; 'heldout' -> the generalization.py spec dict (rebuilt with build())."""
    arm, tag, spec_kind, spec, t_obs, y_obs, isp_err = job
    cfg = ARMS[arm]
    if spec_kind == "family":
        fam, kw = ISO.STANDARD_SPECS[spec]
        yt = ISO.make_isotherm(fam, **kw)(TG)
    else:
        import generalization as GEN
        yt = GEN.build(spec)(TG)
    mu, sd, hyp, edge = _gp_fit(np.asarray(t_obs, float), np.asarray(y_obs, float),
                                TG, cfg["warp"], cfg["pin"], ell_grid(cfg["warp"]))
    return (arm, tag), dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                            ispline_sat_err=float(isp_err), ell=float(hyp[0]))


def _matched_noise_jobs():
    """Jobs for the noise sweep: the I-spline designs stored by recovery.py noise."""
    path = os.path.join(DATA, "results_noise.json")
    if not os.path.exists(path):
        return None, "results_noise.json absent"
    with open(path) as fh:
        NZ = json.load(fh)
    jobs, skipped = [], set()
    for key, runs in NZ.items():
        if not key.startswith("ispline|"):
            continue
        _, nm, sy = key.split("|")
        if not runs or "t_obs" not in runs[0]:
            skipped.add(sy)          # a level whose designs were not stored (e.g. 0.01)
            continue
        for s, r in enumerate(runs):
            for arm in ("gp_log_pin", "gp_log"):
                jobs.append((arm, (nm, float(sy), s), "family", nm, r["t_obs"], r["y_obs"], r["sat_err"]))
    if skipped:
        print(f"  noise: no designs stored at sigma_y in {sorted(skipped)}; those levels are skipped")
    if not jobs:
        return None, "results_noise.json stores no designs (rerun recovery.py noise --isp-sigmas=...)"
    return jobs, None


def _matched_heldout_jobs():
    """Jobs for the held-out benchmark, both x_sat measures."""
    import generalization as GEN
    jobs = []
    for measure, fname in (("loguniform", "results_heldout.json"), ("uniform", "results_heldout_uniform.json")):
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            return None, f"{fname} absent"
        with open(path) as fh:
            H = json.load(fh)
        runs = H["ispline"]
        if not runs or "t_obs" not in runs[0]:
            return None, f"{fname} stores no designs (rerun generalization.py heldout)"
        GEN.XSAT_DIST = measure            # draw_spec reads the module global at call time
        for idx, r in enumerate(runs):
            spec = GEN.draw_spec(idx)
            assert abs(spec["x_sat"] - r["x_sat"]) < 1e-9 and spec["kind"] == r["gkind"], (measure, idx)
            for arm in ("gp_log_pin", "gp_log"):
                jobs.append((arm, (measure, idx), "heldout", spec, r["t_obs"], r["y_obs"], r["sat_err"]))
    return jobs, None


def _print_extra(out, parts):
    """Console summary of the noise and held-out matched-design fits, against the
    own-loop GP caches where they exist."""
    from scipy.stats import wilcoxon

    def own_loop(arm, key_fn, path):
        p = os.path.join(DATA, path)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            d = json.load(fh)
        return key_fn(d)

    if "noise" in parts and "noise" in out:
        print("\n=== matched-design control, noise sweep: GP arms on the I-spline designs at each sigma_y")
        print(f"  {'sigma_y':8s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}{'paired p':>11s}"
              f"{'pin own loop':>14s}{'log own loop':>14s}")
        pin_own = own_loop("gp_log_pin", lambda d: d, "results_gppin_noise.json")
        log_own = own_loop("gp_log", lambda d: d, "results_noise.json")
        sigmas = sorted({float(k.split("|")[1]) for k in out["noise"]["gp_log_pin"]})
        for sy in sigmas:
            keys = [k for k in out["noise"]["gp_log_pin"] if abs(float(k.split("|")[1]) - sy) < 1e-12]
            isp = np.array([r["ispline_sat_err"] for k in keys for r in out["noise"]["gp_log_pin"][k].values()])
            pin = np.array([r["sat_err"] for k in keys for r in out["noise"]["gp_log_pin"][k].values()])
            log = np.array([r["sat_err"] for k in keys for r in out["noise"]["gp_log"][k].values()])
            po = (np.mean([r["sat_err"] for nm in FAMILIES for r in pin_own.get(f"gp_log_pin|{nm}|{sy:g}", [])])
                  if pin_own and any(f"gp_log_pin|{nm}|{sy:g}" in pin_own for nm in FAMILIES) else float("nan"))
            lo = (np.mean([r["sat_err"] for nm in FAMILIES for r in log_own.get(f"gp_log|{nm}|{sy:g}", [])])
                  if log_own and any(f"gp_log|{nm}|{sy:g}" in log_own for nm in FAMILIES) else float("nan"))
            print(f"  {sy:<8g}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}{pin.mean()/isp.mean():>9.2f}"
                  f"{wilcoxon(pin - isp).pvalue:>11.0e}{po:>14.4f}{lo:>14.4f}")
    if "heldout" in parts and "heldout" in out:
        print("\n=== matched-design control, held-out benchmark: GP arms on the I-spline designs")
        print(f"  {'x_sat draw':12s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}{'paired p':>11s}"
              f"{'pin own loop':>14s}{'>sig_y isp/pin/pin-own':>26s}")
        for measure, own in (("loguniform", "results_heldout_gppin.json"), ("uniform", "results_heldout_gppin_uniform.json")):
            if measure not in out["heldout"]["gp_log_pin"]:
                continue
            isp = np.array([r["ispline_sat_err"] for r in out["heldout"]["gp_log_pin"][measure].values()])
            pin = np.array([r["sat_err"] for r in out["heldout"]["gp_log_pin"][measure].values()])
            log = np.array([r["sat_err"] for r in out["heldout"]["gp_log"][measure].values()])
            po = own_loop("gp_log_pin", lambda d: np.array([r["sat_err"] for r in d["gp_log_pin"]]), own)
            po_m = po.mean() if po is not None else float("nan")
            po_t = np.mean(po > SIGMA_Y) if po is not None else float("nan")
            print(f"  {measure:12s}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}{pin.mean()/isp.mean():>9.2f}"
                  f"{wilcoxon(pin - isp).pvalue:>11.0e}{po_m:>14.4f}"
                  f"{100*np.mean(isp > SIGMA_Y):>12.0f}%/{100*np.mean(pin > SIGMA_Y):.0f}%/{100*po_t:.0f}%")


def run_matched(argv):
    """`gp_arms.py matched [--parts=main,noise,heldout]`: the matched-design control.
    main:    (a) GP-log-pin and GP-log fitted to the I-spline campaign's stored designs
             (every strategy, every seed), so surrogate accuracy is compared on identical
             data; (b) the GP arms' own loops re-run to record how flat their bands were
             and how their designs cluster, against the I-spline's own designs.
    noise:   the same fits on the I-spline designs of the output-noise sweep.
    heldout: the same fits on the I-spline designs of the held-out benchmark (both
             x_sat measures).  Parts are merged into results_gp_matched.json."""
    jobs = None
    parts = {"main", "noise", "heldout"}
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--parts="):
            parts = set(a.split("=", 1)[1].split(","))
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    out = {}
    if os.path.exists(MATCHED_CACHE):
        with open(MATCHED_CACHE) as fh:
            out = json.load(fh)

    if parts - {"main"}:
        extra_jobs = []
        for part, mk in (("noise", _matched_noise_jobs), ("heldout", _matched_heldout_jobs)):
            if part not in parts:
                continue
            jb, why = mk()
            if jb is None:
                print(f"  skipping {part}: {why}")
                continue
            extra_jobs += [(part,) + j for j in jb]
        if extra_jobs:
            print(f"matched-design fits on the {sorted(parts - {'main'})} designs: {len(extra_jobs)}; "
                  f"{jobs} workers ...", flush=True)
            got = {}
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_matched_truth_job, j[1:]): j[0] for j in extra_jobs}
                for n, fut in enumerate(as_completed(futs), 1):
                    key, rec = fut.result()
                    got[(futs[fut],) + key] = rec
                    if n % 500 == 0:
                        print(f"  [{n}/{len(futs)}]", flush=True)
            for part in ("noise", "heldout"):
                items = {k: v for k, v in got.items() if k[0] == part}
                if not items:
                    continue
                block = {}
                for (_, arm, tag), rec in items.items():
                    block.setdefault(arm, {}).setdefault("|".join(str(t) for t in tag[:-1]), {})[str(tag[-1])] = rec
                out[part] = block
            with open(MATCHED_CACHE, "w") as fh:
                json.dump(out, fh)
            _print_extra(out, parts)
        if "main" not in parts:
            print("\nwrote", MATCHED_CACHE)
            return

    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)["mc"]
    seeds = list(range(len(R[FAMILIES[0]]["uncertainty"])))
    arms = ["gp_log_pin", "gp_log", "gp_lin"]

    fit_jobs = [(arm, st, nm, s, R[nm][st][s]["t_obs"], R[nm][st][s]["y_obs"], R[nm][st][s]["sat_err"])
                for arm in arms for st in STRATEGIES for nm in FAMILIES for s in seeds]
    loop_jobs = [(arm, nm, s) for arm in arms for nm in FAMILIES for s in seeds]
    print(f"matched-design fits: {len(fit_jobs)}; own-loop re-runs: {len(loop_jobs)}; "
          f"{jobs} workers ...", flush=True)
    recs, loops = {}, {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_matched_fit_job, j): "fit" for j in fit_jobs}
        futs.update({ex.submit(_own_loop_job, j): "loop" for j in loop_jobs})
        for n, fut in enumerate(as_completed(futs), 1):
            job, rec = fut.result()
            (recs if futs[fut] == "fit" else loops)[job] = rec
            if n % 200 == 0:
                print(f"  [{n}/{len(futs)}]", flush=True)

    # merge into the cache so the noise/heldout parts written earlier are kept
    out["matched"] = {arm: {st: {nm: [recs[(arm, st, nm, s)] for s in seeds]
                                 for nm in FAMILIES} for st in STRATEGIES} for arm in arms}
    out["own_loop"] = {arm: {nm: [loops[(arm, nm, s)] for s in seeds] for nm in FAMILIES}
                       for arm in arms}
    # the I-spline's own designs, for the same statistics (no band record: its band is
    # the posterior sd of the MCMC draws and is not re-run here)
    out["own_loop"]["ispline"] = {nm: [_design_stats(R[nm]["uncertainty"][s]["t_obs"]) for s in seeds]
                                  for nm in FAMILIES}
    out["_meta"] = dict(seeds=seeds, strategies=STRATEGIES, arms=arms, n_init=N_INIT,
                        n_iters=N_ITERS, distinct_tol_s=0.02, flat_tol=0.10,
                        warped_span=float(warp(np.array([DOMAIN[1]]), "log")[0]))
    with open(MATCHED_CACHE, "w") as fh:
        json.dump(out, fh)

    from scipy.stats import wilcoxon
    print("\n=== matched-design control: saturation error at N=20 on the I-spline's stored designs")
    print(f"  {'design':14s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}{'paired p':>11s}")
    for st in STRATEGIES:
        isp = np.array([r["ispline_sat_err"] for nm in FAMILIES for r in out["matched"]["gp_log_pin"][st][nm]])
        pin = np.array([r["sat_err"] for nm in FAMILIES for r in out["matched"]["gp_log_pin"][st][nm]])
        log = np.array([r["sat_err"] for nm in FAMILIES for r in out["matched"]["gp_log"][st][nm]])
        print(f"  {st:14s}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}"
              f"{pin.mean()/isp.mean():>9.2f}{wilcoxon(pin - isp).pvalue:>11.0e}")
    print("\n=== own loops: design clustering and band flatness (mean over 250 runs)")
    print(f"  {'arm':11s}{'distinct/20':>12s}{'at xmax':>9s}{'flat share':>12s}{'sdmin/sdmax':>13s}{'ell med/span':>14s}")
    for arm in ["ispline"] + arms:
        rs = [r for nm in FAMILIES for r in out["own_loop"][arm][nm]]
        extra = (f"{np.mean([r['flat_share'] for r in rs]):>12.2f}{np.mean([r['sd_range'] for r in rs]):>13.2f}"
                 f"{np.median([r['ell_median'] for r in rs]) / out['_meta']['warped_span']:>14.2f}"
                 if arm != "ispline" else f"{'':>12s}{'':>13s}{'':>14s}")
        print(f"  {arm:11s}{np.mean([r['n_distinct'] for r in rs]):>12.1f}{np.mean([r['n_at_xmax'] for r in rs]):>9.1f}{extra}")
    print("\nwrote", MATCHED_CACHE)


def main(argv):
    if argv and argv[0] == "matched":
        return run_matched(argv[1:])
    jobs, nseeds, check = None, 50, False
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a == "--check":
            check = True
    seeds = list(range(nseeds))
    joblist = [(arm, nm, s) for arm in ARMS for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))

    recs = {}
    print(f"Running {len(joblist)} GP runs ({len(ARMS)} arms x {len(FAMILIES)} "
          f"families x {len(seeds)} seeds) on {jobs} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_run_job, j) for j in joblist]
        for n, fut in enumerate(as_completed(futs), 1):
            job, rec = fut.result()
            recs[job] = rec
            if n % 100 == 0:
                print(f"  [{n}/{len(joblist)}]", flush=True)

    out = {arm: {nm: [recs[(arm, nm, s)] for s in seeds] for nm in FAMILIES}
           for arm in ARMS}
    out["_meta"] = dict(seeds=seeds, n_init=N_INIT, n_iters=N_ITERS,
                        arms={k: dict(v) for k, v in ARMS.items()},
                        sigma_y=SIGMA_Y)
    with open(CACHE, "w") as fh:
        json.dump(out, fh)

    def col(arm, key, nm=None):
        rs = (out[arm][nm] if nm else
              [r for f in FAMILIES for r in out[arm][f]])
        return np.array([r[key] for r in rs], float)

    print("\n=== GP ablation, N=20, saturation error across levels ===")
    print(f"  {'family':18s}" + "".join(f"{a:>14s}" for a in ARMS)
          + f"{'I-spline':>12s}")
    # Read the I-spline reference from the main cache rather than hardcoding it;
    # the hardcoded values here were from the first-order-penalty era and drifted.
    ISP, ISP_POOL = _ispline_reference()
    for nm in FAMILIES:
        cells = "".join(f"{col(a,'sat_err',nm).mean():>14.3f}" for a in ARMS)
        ref = f"{ISP[nm]:>12.3f}" if nm in ISP else f"{'n/a':>12s}"
        print(f"  {nm:18s}{cells}{ref}")
    pool = f"{ISP_POOL:>12.3f}" if ISP_POOL is not None else f"{'n/a':>12s}"
    print(f"  {'POOLED':18s}" + "".join(f"{col(a,'sat_err').mean():>14.3f}"
                                        for a in ARMS) + pool)

    print("\n=== failure-mode magnitudes at N=20 (pooled over families) ===")
    print(f"  {'arm':12s}{'nonmono %':>11s}{'tot|negvar|':>13s}{'max dip':>10s}"
          f"{'dip>sig_y %':>13s}{'multival %':>12s}{'span (s)*':>11s}"
          f"{'f(0)':>9s}{'edge %':>8s}")
    for a in ARMS:
        mv = col(a, "multivalued").astype(bool)
        span = col(a, "span95")
        print(f"  {a:12s}{100*col(a,'dips').mean():>11.1f}"
              f"{col(a,'tot_negvar').mean():>13.4f}{col(a,'max_dip').mean():>10.4f}"
              f"{100*col(a,'dip_gt_noise').mean():>13.1f}"
              f"{100*mv.mean():>12.1f}"
              f"{(span[mv].mean() if mv.any() else 0.0):>11.2f}"
              f"{col(a,'f0').mean():>9.4f}{100*col(a,'on_edge').mean():>8.1f}")
    print("  * mean upcrossing span over the multi-valued fits only")

    print("\n=== acquisition harm: where the unconstrained loop sent the tool ===")
    print(f"  {'arm':12s}{'on descending mean %':>23s}{'near local min %':>19s}")
    for a in ARMS:
        nd = col(a, "n_desc").sum()
        nn = col(a, "n_nearmin").sum()
        tot = col(a, "n_active").sum()
        print(f"  {a:12s}{100*nd/tot:>23.1f}{100*nn/tot:>19.1f}"
              f"   ({int(tot)} selections)")

    if check:
        print("\n=== hyperparameter grid diagnostics ===")
        for a in ARMS:
            e = col(a, "ell")
            g = ell_grid(ARMS[a]["warp"])
            print(f"  {a:12s} ell grid [{g[0]:.4f},{g[-1]:.3f}]  "
                  f"fitted min={e.min():.4f} med={np.median(e):.3f} max={e.max():.3f}"
                  f"  any-edge={100*col(a,'on_edge').mean():.1f}%")

    print("\nwrote", CACHE)


# ===========================================================================
# Shared GP posterior machinery. gp_arms' own arms use _gp_fit above; these
# two are the pieces controls.py monoproj reuses to project an
# unconstrained GP posterior onto the monotone cone.
# ===========================================================================

def _warp_fwd(x, kind):
    return warp(x, "log" if kind == "log" else "linear")


def _post(u_o, r_o, u_g, egrid, pin_var):
    """MML-optimal RBF posterior on u_g: returns (mu, posterior covariance)."""
    best = (-np.inf, None)
    for ell in egrid:
        for sf in SF_GRID:
            for sn in SN_GRID:
                nvar = np.where(pin_var > 0, 1e-8, sn ** 2)
                lml = _lml(u_o, r_o, nvar, ell, sf)
                if lml > best[0]:
                    best = (lml, (ell, sf, sn))
    ell, sf, sn = best[1]
    nvar = np.where(pin_var > 0, 1e-8, sn ** 2)
    ybar = r_o.mean()
    K = _rbf(u_o, u_o, ell, sf) + np.diag(nvar) + 1e-10 * np.eye(len(u_o))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r_o - ybar))
    Ks = _rbf(u_g, u_o, ell, sf)
    mu = ybar + Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    Kpost = _rbf(u_g, u_g, ell, sf) - v.T @ v
    return mu, Kpost


if __name__ == "__main__":
    main(sys.argv[1:])
