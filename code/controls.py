"""controls.py -- the three one-off controls, each asking whether a design
choice is load-bearing:

  controls.py monoproj   post-hoc monotonization of an unconstrained GP
                         (PAVA and Lin-Dunson) -> results_mono_projection.json
  controls.py band       credible-band calibration and the inferred scales
                         -> results_band.json  (--sigma-only refreshes only the
                         inferred-scale block, 50 refits; --rho-only recomputes
                         the correlation blocks from the raw arrays, no refits)
  controls.py decoupled  acquire with an unconstrained spline, clean up with
                         the monotone one -> results_decoupled.json

Usage:  python3 controls.py <monoproj|band|decoupled> [--jobs=N] [--seeds=N]
"""
import os
import sys
import json
import time
import gc

import numpy as np
import jax
from scipy.stats import spearmanr

import gp_arms as GPA
import ispline_model as M
import isotherms as ISO
import poolutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOMAIN = ISO.DOMAIN
# GPA.TG is eval_points(0, x_max); verified identical to M.eval_points here.
TG = M.eval_points(0.0, DOMAIN[1])
ACQ = GPA.ACQ
FAMILIES = list(ISO.STANDARD_SPECS)
SIGMA_Y = ISO.SIGMA_Y
TRUE_XSAT = 2.0
ALPHA = 0.95
N_INIT, N_ITERS = 5, 15
# Lin-Dunson posterior draws: 500, matching the I-spline arm it is compared
# against, so every read-out summarizes 500 posterior draws.
NDRAW = 500

# ===========================================================================
# monoproj -- can an unconstrained GP be repaired after the fact?
# ===========================================================================


def _pava(y):
    """Monotone (non-decreasing) L2 isotonic projection via pool-adjacent-violators."""
    y = np.asarray(y, float).copy()
    # simple O(n) PAVA: push each value, then pool back while the last block
    # violates monotonicity, carrying block weights so the pooled value is the
    # weighted mean (the L2 projection).
    vals = list(y)
    out_vals = []
    out_w = []
    for v in vals:
        out_vals.append(v)
        out_w.append(1.0)
        while len(out_vals) > 1 and out_vals[-2] > out_vals[-1]:
            v2 = out_vals.pop(); w2 = out_w.pop()
            v1 = out_vals.pop(); w1 = out_w.pop()
            out_vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            out_w.append(w1 + w2)
    res = []
    for v, ww in zip(out_vals, out_w):
        res.extend([v] * int(ww))
    return np.array(res)


def _xsat_from_curve(d, alpha):
    """First up-crossing of alpha*plateau on a monotone-enough curve."""
    tgt = alpha * d[-1]
    idx = np.where(d >= tgt)[0]
    if idx.size == 0 or idx[0] == 0:
        return None
    i = idx[0]
    y0, y1 = d[i - 1], d[i]
    if y1 <= y0:
        return None
    return TG[i - 1] + (tgt - y0) * (TG[i] - TG[i - 1]) / (y1 - y0)


def _gp_logpin_designs(nm, seed):
    """Run the GP-log-pin loop; return the collected design and the true curve."""
    kind, pin = "log", True
    egrid = GPA.ell_grid(kind)
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(GPA.N_INIT, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    for _ in range(GPA.N_ITERS):
        _, sd_a, _, _ = GPA._gp_fit(xs, ys, ACQ, kind, pin, egrid)
        xn = float(ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    return xs, ys, f(TG)


def _mono_job(job):
    nm, seed = job
    kind, pin = "log", True
    xs, ys, yt = _gp_logpin_designs(nm, seed)
    xo = np.concatenate([[0.0], np.asarray(xs, float)])   # pin origin
    yo = np.concatenate([[0.0], np.asarray(ys, float)])
    pin_var = np.zeros(len(xo)); pin_var[0] = 1.0
    egrid = GPA.ell_grid(kind)
    mu, Kpost = GPA._post(GPA._warp_fwd(xo, kind), yo,
                          GPA._warp_fwd(TG, kind), egrid, pin_var)
    L = np.linalg.cholesky(Kpost + 1e-9 * np.eye(len(TG)))
    rng = np.random.default_rng(seed + 10_000)
    draws = mu[None, :] + (L @ rng.standard_normal((len(TG), NDRAW))).T

    # raw (unconstrained) GP
    raw_err = float(M.saturation_error(TG, mu, yt))
    raw_dips = int(np.any(np.diff(mu) < -1e-9))

    # PAVA of the mean
    mu_p = _pava(mu)
    pava_err = float(M.saturation_error(TG, mu_p, yt))
    x_pava = _xsat_from_curve(mu_p, ALPHA)

    # Lin-Dunson projection of every draw
    proj = np.array([_pava(d) for d in draws])
    ld_mean = proj.mean(0)
    ld_err = float(M.saturation_error(TG, ld_mean, yt))
    xs_ld = [x for x in (_xsat_from_curve(d, ALPHA) for d in proj) if x is not None]
    if xs_ld:
        med = float(np.median(xs_ld))
        lo, hi = np.percentile(xs_ld, [2.5, 97.5])
        ld_bias = med - TRUE_XSAT
        ld_cov = bool(lo <= TRUE_XSAT <= hi)
    else:
        ld_bias, ld_cov = float("nan"), False
    return job, dict(raw_err=raw_err, raw_dips=raw_dips,
                     pava_err=pava_err,
                     pava_bias=(x_pava - TRUE_XSAT) if x_pava else float("nan"),
                     ld_err=ld_err, ld_bias=ld_bias, ld_cov=ld_cov,
                     n_cross_ok=len(xs_ld))


def run_monoproj(argv):
    jobs, seeds = 4, list(range(50))
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    if smoke:
        for nm in (FAMILIES[0], FAMILIES[2]):
            _, r = _mono_job((nm, 0))
            print(f"  smoke {nm:12s} raw {r['raw_err']:.3f} (dips={r['raw_dips']}) "
                  f"PAVA {r['pava_err']:.3f} LD {r['ld_err']:.3f} "
                  f"LD x_sat bias {r['ld_bias']:+.2f} cov {r['ld_cov']}")
        print("SMOKE OK")
        return
    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_mono_job, joblist, jobs, tasks_per_child=20,
                             label="mono-projection")
    out = {f"{nm}|{s}": recs[(nm, s)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, alpha=ALPHA,
                        true_xsat=TRUE_XSAT, ndraw=NDRAW, domain=list(DOMAIN),
                        arm="GP-log-pin post-hoc monotonized (PAVA mean / Lin-Dunson draws)")
    with open(os.path.join(DATA, "results_mono_projection.json"), "w") as fh:
        json.dump(out, fh)

    def pool(k):
        return np.array([recs[(nm, s)][k] for nm in FAMILIES for s in seeds], float)
    print(f"MONO-PROJECTION COMPLETE in {time.time()-t0:.0f}s "
          f"({len(FAMILIES)} families x {len(seeds)} seeds)")
    print(f"  raw GP-log-pin curve err      : {np.nanmean(pool('raw_err')):.3f} "
          f"(mean dips in {100*pool('raw_dips').mean():.0f}% of fits)")
    print(f"  PAVA-of-the-mean curve err    : {np.nanmean(pool('pava_err')):.3f}")
    print(f"  Lin-Dunson draws curve err    : {np.nanmean(pool('ld_err')):.3f}")
    print(f"  Lin-Dunson x_sat(0.95) bias   : {np.nanmedian(pool('ld_bias')):+.2f} s, "
          f"coverage {100*np.nanmean(pool('ld_cov')):.0f}%")
    print(f"  (I-spline monotone-in-prior reference: 0.039 curve, +0.3 s / 94%)")



# ===========================================================================
# band -- is the credible band calibrated, and is its width the problem?
# ===========================================================================

# ---------------------------------------------------------------------------
# statistics (shared by both arms, so the two are compared like for like)
# ---------------------------------------------------------------------------
def rho_all(sd, err, x):
    """Spearman(band width, realized error) under four conventions.

    The convention matters more than the number: rho ranges over roughly 0.06
    to 0.35 on the same fits depending on which one is used, so any report
    has to name the one it quotes.  `controls.py band --rho-only` recomputes
    every rho block of an existing cache from results_band_raw.npz without
    refitting, for when a convention changes.

      pointwise      over every (grid point, seed) pair on the full grid.
                     Inflated by the x-profile: ~27% of the log grid lies below
                     x_min, where band and error are both ~0, so this measures
                     "band and error are both small near the origin" as much as
                     "the band tracks the error".
      pointwise_in   the same, restricted to x >= x_min (data-supported region).
      across_seeds   at each x >= x_min, correlate across seeds, then average
                     over those x. Removes the x-profile confound: it asks
                     whether a wider band at this input on this seed predicts
                     a larger error there. This is the one that answers the
                     question the acquisition argument actually needs. It is
                     restricted to the data-supported region because below
                     x_min band and error are both ~0 and their ranks across
                     seeds carry no information (averaging over the whole grid
                     would give 0.20 instead of 0.08 for the penalized arm).
      run_level      mean band width vs mean error, one point per run.
    """
    sd, err, x = np.asarray(sd), np.asarray(err), np.asarray(x)
    out = {"pointwise": float(spearmanr(sd.ravel(), err.ravel()).statistic)}
    k = x >= ISO.DOMAIN[0]
    out["pointwise_in"] = float(spearmanr(sd[:, k].ravel(),
                                          err[:, k].ravel()).statistic)
    per_x = []
    for j in np.flatnonzero(k):
        if sd[:, j].std() > 0 and err[:, j].std() > 0:
            per_x.append(spearmanr(sd[:, j], err[:, j]).statistic)
    out["across_seeds"] = float(np.nanmean(per_x)) if per_x else float("nan")
    out["run_level"] = float(spearmanr(sd.mean(1), err.mean(1)).statistic)
    return out


def band_stats(mu, sd, yt, x=TG):
    """mu, sd: (n_seeds, n_grid). yt: (n_grid,) ground truth."""
    mu, sd, yt = np.asarray(mu), np.asarray(sd), np.asarray(yt)
    err = np.abs(mu - yt)
    inside = err <= 1.96 * sd
    # is the band under-dispersed, or mis-centred?  compare its half-width to
    # the seed-to-seed spread of the posterior mean at the same x.
    spread = mu.std(0, ddof=1)
    ok = spread > 0
    d = dict(
        coverage=float(inside.mean()),
        coverage_in=float(inside[:, np.asarray(x) >= ISO.DOMAIN[0]].mean()),
        band_over_spread=float(np.mean(sd[:, ok].mean(0) / spread[ok])),
        mean_halfwidth=float(np.mean(1.96 * sd)),
    )
    d["rho"] = rho_all(sd, err, x)
    # kept for backward compatibility with the first version of this cache
    d["rho_pointwise"] = d["rho"]["pointwise"]
    d["rho_run"] = d["rho"]["run_level"]
    return d


def pooled_stats(per_family):
    """Pool the (grid point, seed) pairs of every family into one statistic."""
    mu = np.concatenate([f["mu"] for f in per_family])
    sd = np.concatenate([f["sd"] for f in per_family])
    yt = np.concatenate([np.tile(f["yt"], (len(f["mu"]), 1)) for f in per_family])
    err = np.abs(mu - yt)
    inside = err <= 1.96 * sd
    k = TG >= ISO.DOMAIN[0]
    d = dict(
        coverage=float(inside.mean()),
        coverage_in=float(inside[:, k].mean()),
    )
    d["rho"] = rho_all(sd, err, TG)
    d["rho_pointwise"] = d["rho"]["pointwise"]
    d["rho_run"] = d["rho"]["run_level"]
    return d


# ---------------------------------------------------------------------------
# arm 1: the penalized surrogate, straight out of the run_band campaign cache
# ---------------------------------------------------------------------------
def penalized_arm():
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)
    per_fam, out, raw = [], {}, {}
    for nm in FAMILIES:
        runs = R["mc"][nm]["uncertainty"]
        mu = np.array([r["mu"] for r in runs])
        sd = np.array([r["sd"] for r in runs])
        yt = np.array(runs[0]["yt"])
        per_fam.append(dict(mu=mu, sd=sd, yt=yt))
        raw[nm] = dict(mu=mu, sd=sd, yt=yt)
        out[nm] = band_stats(mu, sd, yt)
    out["pooled"] = pooled_stats(per_fam)
    out["_n_seeds"] = len(R["mc"][FAMILIES[0]]["uncertainty"])
    return out, raw


# ---------------------------------------------------------------------------
# arm 2: the unpenalized P=11 basis, driving its own loop (needs the fits)
# ---------------------------------------------------------------------------
def _job_unpen(job):
    nm, seed = job
    fname, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fname, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_y=SIGMA_Y)
    rng = np.random.default_rng(seed)
    reg = M.UnpenalizedRegressor(domain=DOMAIN, n_init=N_INIT, n_iters=N_ITERS)
    for x in M.survey_design(N_INIT, DOMAIN, rng):
        reg.add(x, observe(x, rng))
    reg.fit(seed=seed)
    for _ in range(N_ITERS):
        xn = reg.next_x()
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)
    mu, sd = reg.predict(TG)
    return job, dict(mu=[float(v) for v in mu], sd=[float(v) for v in sd])


def unpenalized_arm(seeds, jobs):
    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    recs = poolutil.run_pool(_job_unpen, joblist, jobs, tasks_per_child=10,
                             label="unpenalized P=11 band")
    per_fam, out, raw = [], {}, {}
    for nm in FAMILIES:
        fname, kw = ISO.STANDARD_SPECS[nm]
        yt = ISO.make_isotherm(fname, **kw)(TG)
        mu = np.array([recs[(nm, s)]["mu"] for s in seeds])
        sd = np.array([recs[(nm, s)]["sd"] for s in seeds])
        per_fam.append(dict(mu=mu, sd=sd, yt=yt))
        raw[nm] = dict(mu=mu, sd=sd, yt=yt)
        out[nm] = band_stats(mu, sd, yt)
    out["pooled"] = pooled_stats(per_fam)
    out["_n_seeds"] = len(seeds)
    return out, raw


# ---------------------------------------------------------------------------
# arm 3: the inferred scales sigma and tau_eta at N=20
# ---------------------------------------------------------------------------
def _job_sigma(job):
    nm, seed = job
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)
    d = R["mc"][nm]["uncertainty"][seed]
    reg = M.ISplineRegressor(domain=DOMAIN, n_init=N_INIT, n_iters=N_ITERS)
    for x, y in zip(d["t_obs"], d["y_obs"]):
        reg.add(x, y)
    reg.fit(seed=seed)
    # posterior mean of each scale, and the posterior sd of tau_eta within this
    # one fit (how well twenty points pin the roughness)
    return job, dict(sigma_post=float(np.mean(reg._sigma)),
                     tau_post=float(np.mean(reg._tau)),
                     tau_psd=float(np.std(reg._tau)))


def sigma_arm(seeds, jobs):
    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    recs = poolutil.run_pool(_job_sigma, joblist, jobs, tasks_per_child=10,
                             label="inferred scale refits")
    out = {}
    for nm in FAMILIES:
        v = np.array([recs[(nm, s)]["sigma_post"] for s in seeds])
        t = np.array([recs[(nm, s)]["tau_post"] for s in seeds])
        tp = np.array([recs[(nm, s)]["tau_psd"] for s in seeds])
        out[nm] = dict(sigma_mean=float(v.mean()), sigma_sd=float(v.std(ddof=1)),
                       excess_pct=float(100 * (v.mean() - SIGMA_Y) / SIGMA_Y),
                       tau_mean=float(t.mean()), tau_sd=float(t.std(ddof=1)),
                       tau_psd_mean=float(tp.mean()))
    out["_n_seeds"] = len(seeds)
    out["_tau_prior"] = dict(scale=M.ISplineRegressor().tau_scale,
                             median=float(M.ISplineRegressor().tau_scale * 0.6745))
    return out


def _print_scales(sig):
    print(f"\n  inferred scales at N=20 (generative sigma_y={SIGMA_Y}; "
          f"tau_eta prior median {sig['_tau_prior']['median']:.3f}):")
    for nm in FAMILIES:
        d = sig[nm]
        print(f"    {nm:14s} sigma {d['sigma_mean']:.3f} ({d['excess_pct']:+.0f}%)   "
              f"tau_eta {d['tau_mean']:.3f} +/- {d['tau_sd']:.3f}, "
              f"within-fit sd {d['tau_psd_mean']:.3f}")


def run_band(argv):
    jobs, nseeds = 3, 50
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a == "--smoke":
            nseeds = 2
    seeds = list(range(nseeds))
    t0 = time.time()

    if "--rho-only" in argv:
        # recompute the rho blocks (per family and pooled, both arms) from the
        # saved raw posterior means and bands; nothing is refit.
        path = os.path.join(DATA, "results_band.json")
        with open(path) as fh:
            out = json.load(fh)
        Z = np.load(os.path.join(DATA, "results_band_raw.npz"))
        for arm, key in (("pen", "penalized"), ("unp", "unpenalized_p11")):
            per_fam = []
            for nm in FAMILIES:
                mu, sd, yt = Z[f"{arm}|{nm}|mu"], Z[f"{arm}|{nm}|sd"], Z[f"{arm}|{nm}|yt"]
                per_fam.append(dict(mu=mu, sd=sd, yt=yt))
                new = band_stats(mu, sd, yt)
                for k_ in ("rho", "rho_pointwise", "rho_run"):
                    out[key][nm][k_] = new[k_]
            new = pooled_stats(per_fam)
            for k_ in ("rho", "rho_pointwise", "rho_run"):
                out[key]["pooled"][k_] = new[k_]
            r = out[key]["pooled"]["rho"]
            print(f"  {key:16s} " + "  ".join(f"{c} {v:.3f}" for c, v in r.items()))
        if smoke:
            print("  SMOKE: caches not written")
            return 0
        with open(path, "w") as fh:
            json.dump(out, fh)
        print("rho blocks refreshed")
        return 0

    if "--sigma-only" in argv:
        # refresh only the inferred-scale block of an existing cache; the band
        # arms (250 refits) are untouched.  50 refits, a few minutes.
        path = os.path.join(DATA, "results_band.json")
        with open(path) as fh:
            out = json.load(fh)
        sig_seeds = seeds[:10]
        print(f"inferred-scale refits: {len(FAMILIES) * len(sig_seeds)} fits ...",
              flush=True)
        out["sigma"] = sigma_arm(sig_seeds, jobs)
        if smoke:
            print("  SMOKE: caches not written")
            return 0
        with open(path, "w") as fh:
            json.dump(out, fh)
        _print_scales(out["sigma"])
        return 0

    print("penalized arm (from results.json, no refit) ...", flush=True)
    pen, pen_raw = penalized_arm()
    print(f"unpenalized P=11 arm: {len(FAMILIES) * len(seeds)} runs on "
          f"{jobs} workers ...", flush=True)
    unp, unp_raw = unpenalized_arm(seeds, jobs)

    sig_seeds = seeds[:10]
    print(f"inferred-scale refits: {len(FAMILIES) * len(sig_seeds)} fits ...",
          flush=True)
    sig = sigma_arm(sig_seeds, jobs)

    out = {"penalized": pen, "unpenalized_p11": unp, "sigma": sig,
           "_meta": dict(families=FAMILIES, seeds=seeds, sigma_y=SIGMA_Y,
                         domain=list(DOMAIN), n_init=N_INIT, n_iters=N_ITERS,
                         grid="eval_points(0, x_max)", n_grid=len(TG),
                         nominal=0.95, z=1.96)}
    if smoke:
        # A 2-seed smoke run must never land in the production cache.
        print("  SMOKE: caches not written")
        return 0
    with open(os.path.join(DATA, "results_band.json"), "w") as fh:
        json.dump(out, fh)
    # the summary above discards mu/sd, which cost 30 minutes of sampling to
    # produce; keep them so another rho convention can be tested for free.
    np.savez_compressed(
        os.path.join(DATA, "results_band_raw.npz"), x=TG,
        **{f"{arm}|{nm}|{k}": np.asarray(v)
           for arm, fams in (("pen", pen_raw), ("unp", unp_raw))
           for nm, d in fams.items() for k, v in d.items()})

    print(f"\nBAND CALIBRATION COMPLETE in {time.time() - t0:.0f}s")
    print(f"  {'arm':22s}{'coverage':>10s}{'rho_pt':>9s}{'rho_run':>9s}"
          f"{'band/spread':>13s}")
    for lab, d in (("penalized (P=25)", pen), ("unpenalized (P=11)", unp)):
        p = d["pooled"]
        bs = np.mean([d[nm]["band_over_spread"] for nm in FAMILIES])
        print(f"  {lab:22s}{100 * p['coverage']:>9.1f}%{p['rho_pointwise']:>9.3f}"
              f"{p['rho_run']:>9.3f}{bs:>13.2f}")

    print(f"\n  rho_s by convention (a report must name the one it quotes):")
    convs = list(pen["pooled"]["rho"])
    print(f"  {'convention':16s}{'penalized':>12s}{'unpenalized':>14s}{'':>4s}")
    for c in convs:
        a, b = pen["pooled"]["rho"][c], unp["pooled"]["rho"][c]
        flag = "" if a > b else "   <-- unpenalized HIGHER"
        print(f"  {c:16s}{a:>12.3f}{b:>14.3f}{flag}")
    print(f"\n  per-family coverage (penalized): " + ", ".join(
        f"{nm.split()[0]} {100 * pen[nm]['coverage']:.0f}%" for nm in FAMILIES))
    _print_scales(sig)
    return 0


# ===========================================================================
# decoupled -- does acquiring with an unconstrained spline buy anything?
# ===========================================================================
WARMUP, DRAWS = 500, 500                     # production; the smoke overrides to 50/50


def _run_one(nm, seed, warmup=WARMUP, draws=DRAWS, n_iters=N_ITERS):
    """One (family, seed): the signed driver picks the points, the production
    penalized I-spline cleans up at each N."""
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    yt = f(TG)
    rng = np.random.default_rng(seed)

    xs, ys = [], []
    for x in M.survey_design(N_INIT, DOMAIN, rng):
        xs.append(float(x))
        ys.append(float(observe(x, rng)))

    curve = []          # (N, driver_err, cleanup_err)
    cleanup_nonmono = 0
    for step in range(n_iters + 1):
        N = N_INIT + step
        driver = M.SignedRegressor(domain=DOMAIN, n_init=N_INIT, n_iters=n_iters,
                                   warmup=warmup, draws=draws)
        for x, y in zip(xs, ys):
            driver.add(x, y)
        driver.fit(seed=seed)

        cleanup = M.ISplineRegressor(domain=DOMAIN, n_init=N_INIT, n_iters=n_iters,
                                     warmup=warmup, draws=draws)
        for x, y in zip(xs, ys):
            cleanup.add(x, y)
        cleanup.fit(seed=seed)

        d_err = float(M.saturation_error(TG, driver.predict(TG)[0], yt))
        c_err = float(M.saturation_error(TG, cleanup.predict(TG)[0], yt))
        cleanup_nonmono += cleanup.monotonicity_report()["non_monotone"]
        curve.append((N, d_err, c_err))

        if step < n_iters:                  # signed driver chooses the next input
            xn = driver.next_x()
            xs.append(xn)
            ys.append(float(observe(xn, rng)))

    jax.clear_caches()                       # bound memory once per run_decoupled (not per fit)
    gc.collect()
    return dict(curve=curve, cleanup_nonmono=int(cleanup_nonmono),
                driver_final=curve[-1][1], cleanup_final=curve[-1][2])


def _dec_job(job):
    nm, seed = job
    return job, _run_one(nm, seed)


def run_decoupled(argv):
    jobs, seeds = 3, list(range(50))
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))

    if smoke:
        for nm in FAMILIES[:1]:
            for s in (0, 1):
                r = _run_one(nm, s, warmup=50, draws=50, n_iters=3)
                assert r["cleanup_nonmono"] == 0, "cleanup fit must be monotone"
                assert len(r["curve"]) == 4, "expected N in {5,6,7,8}"
                print(f"  smoke {nm} seed{s}: driver_final={r['driver_final']:.3f} "
                      f"cleanup_final={r['cleanup_final']:.3f}  cleanup_mono_ok")
        print("SMOKE OK")
        return

    joblist = [(nm, s) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_dec_job, joblist, jobs, tasks_per_child=4, label="decoupled")
    out = {f"{nm}|{s}": recs[(nm, s)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, n_init=N_INIT, n_iters=N_ITERS,
                        sigma_x=ISO.SIGMA_X, sigma_y=ISO.SIGMA_Y, domain=list(DOMAIN),
                        driver="SignedRegressor (Eilers-Marx penalized, unconstrained)",
                        cleanup="ISplineRegressor (production penalized)", metric="saturation_error")
    with open(os.path.join(DATA, "results_decoupled.json"), "w") as fh:
        json.dump(out, fh)
    nonmono = sum(recs[(nm, s)]["cleanup_nonmono"] for nm in FAMILIES for s in seeds)
    dpool = np.mean([recs[(nm, s)]["driver_final"] for nm in FAMILIES for s in seeds])
    cpool = np.mean([recs[(nm, s)]["cleanup_final"] for nm in FAMILIES for s in seeds])
    print(f"DECOUPLED COMPLETE in {time.time()-t0:.0f}s. "
          f"pooled N=20: signed-driver {dpool:.3f}, cleanup {cpool:.3f}. "
          f"cleanup non-monotone draws (must be 0): {nonmono}.", flush=True)



RUNNERS = {"monoproj": run_monoproj, "band": run_band,
           "decoupled": run_decoupled}


def main(argv):
    if not argv or argv[0] not in RUNNERS:
        print(__doc__)
        return 2
    RUNNERS[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
