"""
design_choices.py -- the configuration justifications: knot geometry, prior
robustness, acquisition grid, and sampler convergence. Six subcommands:
knots | priors | acqgrid | diag | sampler | targets. Every subcommand writes a
cache in data/, which recompute_stats.py and make_figures.py read instead of
recomputing.

  subcommand  what it justifies                            output / consumer
  ----------  ------------------------------------------    ------------------------------
  knots       does the log-spaced grid earn its keep vs a   results_knots.json
              uniform grid? best monotone approximation     (recompute_stats knots,
              (NNLS, noise-free) across 4 configs           make_figures knot figure)
  priors      prior sensitivity: largest posterior-mean     results_priors.json
              shift under 5x changes to the roughness       (recompute_stats prior)
              (s_tau) and noise (s_sigma) scales, N in {5,10,20}
  acqgrid     acquisition control: linear vs geometric      results_acqgrid.json
              candidate grid, and grid-located vs           (recompute_stats acqgrid)
              root-found level error
  diag        MCMC convergence for the production model     results_diag.json
              (non-centred log-RW): 4-chain Rhat/ESS/       (recompute_stats diag)
              divergences + single-chain campaign divergence/ESS
  sampler     production vs fixed sampler settings refit    results_sampler_settings.json
              on identical stored designs                   (recompute_stats sampler)
  targets     target-acceptance / initialization probe      results_target_accept.json
              on the seed-0 designs                         (recompute_stats sampler)

Usage:  python3 design_choices.py <knots|priors|acqgrid|diag|sampler|targets> [--jobs=N]
'priors', 'diag', 'sampler' and 'targets' read results.json (the stored designs). Import is side-effect
free, so make_figures can import the knot helpers below.
"""
import os
import poolutil
import sys
import json
import time

import numpy as np
from scipy.optimize import nnls
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary

import ispline_model as M
import isotherms as ISO

numpyro.set_host_device_count(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # result caches live in code/data
DOMAIN = ISO.DOMAIN
TG = M.eval_points(0.0, DOMAIN[1])
FAMILIES = list(ISO.STANDARD_SPECS)
KNOTS = M.make_knots(DOMAIN, 21, 3)          # production penalized basis, P=25
DEG = 3

# The prior sweep must measure the PRODUCTION surrogate, so it takes the model
# from ISplineRegressor._MODEL rather than re-implementing it here: a private
# copy could drift silently if the production penalty ever changed.
# test_surrogates.py asserts this really is the second-order walk.
PRODUCTION_MODEL = M.ISplineRegressor._MODEL


def _load_results():
    with open(os.path.join(DATA, "results.json")) as fh:
        return json.load(fh)


def _set_threads():
    poolutil.pin_threads()


# ===========================================================================
# knots -- does the log-spaced knot grid earn its keep vs a uniform grid?
# (best monotone approximation, NNLS of the noise-free curve onto the basis;
#  no MCMC, no noise -- isolates the basis. Helpers reused by make_figures.)
# ===========================================================================

def knots_log(domain, n_interior, degree=3):
    return M.make_knots(domain, n_interior, degree)


def knots_uniform(domain, n_interior, degree=3):
    t_lo, t_hi = domain
    interior = np.linspace(t_lo, t_hi, n_interior + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), interior, np.full(degree + 1, t_hi)])


KNOT_CONFIGS = {
    "P=25 log (default)": (knots_log, 21),
    "P=25 uniform":       (knots_uniform, 21),
    "P=11 log":           (knots_log, 7),
    "P=11 uniform":       (knots_uniform, 7),
}

# Representative curves: the five headlines, plus the extreme fast-rising
# regime variants (near-origin resolution matters most) and the late, step-like
# Sips n=4 (where it should hurt).
KNOT_CURVES = {
    "Langmuir (headline)":   ISO.make_isotherm("Langmuir"),
    "Langmuir tau=0.15":     ISO.make_isotherm("Langmuir", tau=0.15),
    "KWW beta=0.35 (fast)":  ISO.make_isotherm("DR-KWW", beta=0.35),
    "KWW (headline)":        ISO.make_isotherm("DR-KWW", beta=0.55),
    "MM K=0.03 (steep)":     ISO.make_isotherm("Dissociative", K=0.03),
    "Sips n=2 (headline)":   ISO.make_isotherm("Sips", n=2.0),
    "Sips n=4 (late step)":  ISO.make_isotherm("Sips", n=4.0),
    "Elovich (headline)":    ISO.make_isotherm("Elovich"),
}

# Fit on a log-spaced grid (dense where curves bend); score with the same
# rise-weighted saturation-error metric on a fine linear grid.
_t_fit = np.concatenate([[0.0], np.geomspace(DOMAIN[0], DOMAIN[1], 1500)])
_t_eval = np.linspace(0.0, DOMAIN[1], 2000)


def best_approx_err(f, knots):
    B_fit = M.ispline_basis(_t_fit, knots, 3)
    w, _ = nnls(B_fit, f(_t_fit))
    mu = M.ispline_basis(_t_eval, knots, 3) @ w
    return M.saturation_error(_t_eval, mu, f(_t_eval))


def run_knots(argv):
    hdr = "  {:24s}".format("curve") + "".join(f"{c:>20s}" for c in KNOT_CONFIGS)
    print("=== Best monotone approximation error (noise-free, saturation-error "
          "across levels) ===\n")
    print(hdr)
    print("  " + "-" * (24 + 20 * len(KNOT_CONFIGS)))
    out = {}
    for name, f in KNOT_CURVES.items():
        cells = []
        out[name] = {}
        for cfg, (kfun, ni) in KNOT_CONFIGS.items():
            err = best_approx_err(f, kfun(DOMAIN, ni))
            out[name][cfg] = float(err)
            cells.append(f"{err:.4f}")
        print("  {:24s}".format(name) + "".join(f"{c:>20s}" for c in cells))

    # Cache the table so the knot numbers stay recomputable rather than living
    # only in this console output; recompute_stats and make_figures read it.
    out[chr(95) + 'meta'] = dict(configs=list(KNOT_CONFIGS),
                                curves=list(KNOT_CURVES), domain=list(DOMAIN),
                                n_fit=len(_t_fit), n_eval=len(_t_eval))
    with open(os.path.join(DATA, 'results_knots.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', os.path.join(DATA, 'results_knots.json'))


# ===========================================================================
# priors -- prior sensitivity of the production penalized model: largest posterior-mean
# shift under 5x changes to the roughness scale s_tau and noise scale s_sigma,
# swept over budgets N in {5,10,20} (where the prior matters most)
# ===========================================================================
BUDGETS = [5, 10, 20]
PRIORS_SEEDS = [0, 1, 2, 3, 4]
BASE_PRIOR = (0.3, 0.3)   # production defaults: roughness scale s_tau, noise scale s_sigma
# one-at-a-time (4) + joint corners (4), each scale x5 / /5 about the production base
PRIORS_CONFIGS = {
    "baseline":        (0.3, 0.3),
    "s_tau x5":        (1.5, 0.3),
    "s_tau /5":        (0.06, 0.3),
    "s_sig x5":        (0.3, 1.5),
    "s_sig /5":        (0.3, 0.06),
    "both x5":         (1.5, 1.5),
    "both /5":         (0.06, 0.06),
    "tau x5, sig /5":  (1.5, 0.06),
    "tau /5, sig x5":  (0.06, 1.5),
}
_BG = M.ispline_basis(TG, KNOTS, 3)


def _priors_fit(t_obs, y_obs, s_tau, s_sigma, seed):
    Xb = M.ispline_basis(np.asarray(t_obs, float), KNOTS, 3)
    mask = np.ones(len(t_obs), bool)
    mcmc = MCMC(NUTS(PRODUCTION_MODEL), num_warmup=500, num_samples=500,
                num_chains=1, chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), Xb=jnp.array(Xb), mask=jnp.array(mask),
             tau_scale=float(s_tau), sigma_prior=float(s_sigma),
             y=jnp.array(np.asarray(y_obs, float)))
    w = np.array(mcmc.get_samples()["w"])
    jax.clear_caches()
    return (w @ _BG.T).mean(0)


def _job_priors(job):
    nm, N, seed = job
    RES = _load_results()
    d = RES["mc"][nm]["uncertainty"][seed]
    # the first N points of the recorded design (survey first, then active)
    t_obs, y_obs = d["t_obs"][:N], d["y_obs"][:N]
    curves = {}
    for cfg, (st, ss) in PRIORS_CONFIGS.items():
        curves[cfg] = _priors_fit(t_obs, y_obs, st, ss, seed)
    base = curves["baseline"]
    dev = {cfg: float(np.max(np.abs(c - base)))
           for cfg, c in curves.items() if cfg != "baseline"}
    return job, dev


def run_priors(argv):
    jobs = None
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    joblist = [(nm, N, s) for nm in FAMILIES for N in BUDGETS for s in PRIORS_SEEDS]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
    jobs = min(jobs, len(joblist))

    # recycling pool (poolutil.run_pool), as every other MCMC campaign uses:
    # each cell runs nine NUTS fits, and a persistent worker would keep every
    # compiled executable it has ever built
    print(f"Prior-sensitivity cells: {len(PRIORS_CONFIGS)-1} perturbations each",
          flush=True)
    acc = poolutil.run_pool(_job_priors, joblist, jobs, tasks_per_child=4,
                            every=20, label="prior-sensitivity cells")

    print("\n=== Largest pointwise shift of the posterior-mean curve under a "
          "5x prior change (response units, A=1) ===\n")
    print(f"  {'family':18s}" + "".join(f"{'N='+str(N):>12s}" for N in BUDGETS))
    worst_overall = 0.0
    for nm in FAMILIES:
        cells = []
        for N in BUDGETS:
            vals = [max(acc[(nm, N, s)].values()) for s in PRIORS_SEEDS]
            worst_overall = max(worst_overall, max(vals))
            cells.append(f"{np.mean(vals):.3f}/{max(vals):.3f}")
        print(f"  {nm:18s}" + "".join(f"{c:>12s}" for c in cells))
    print("  (mean over seeds / worst seed)")

    print("\n  worst perturbation type, pooled over families and seeds:")
    for cfg in PRIORS_CONFIGS:
        if cfg == "baseline":
            continue
        for N in BUDGETS:
            vals = [acc[(nm, N, s)][cfg] for nm in FAMILIES for s in PRIORS_SEEDS]
            if N == BUDGETS[0]:
                line = f"    {cfg:16s}"
            line += f"  N={N}: {np.mean(vals):.3f}"
        print(line)
    print(f"\n  >> WORST shift anywhere (any family, seed, budget, corner): "
          f"{worst_overall:.3f}")

    out = {f"{nm}|{N}|{s}": acc[(nm, N, s)]
           for nm in FAMILIES for N in BUDGETS for s in PRIORS_SEEDS}
    out["_meta"] = dict(families=FAMILIES, budgets=BUDGETS, seeds=PRIORS_SEEDS,
                        configs={k: list(v) for k, v in PRIORS_CONFIGS.items()},
                        base=list(BASE_PRIOR), model="ISplineRegressor._MODEL "
                        "(production second-order log-random-walk penalty)")
    with open(os.path.join(DATA, "results_priors.json"), "w") as fh:
        json.dump(out, fh)
    print("  wrote data/results_priors.json")


# ===========================================================================
# acqgrid -- acquisition control: linear vs geometric candidate grid, and
# grid-located vs root-found level error (measures the aliasing effect)
# ===========================================================================
ACQ_SEEDS = list(range(50))
N_INIT, N_ITERS = 5, 15
ACQ_ARMS = ["uncertainty", "uncertainty_geom"]
LEVELS19 = np.arange(0.05, 0.96, 0.05)


def exact_level_error(f, t, mu):
    """Metric (i) with the true crossings ROOT-FOUND instead of grid-located."""
    err = []
    for a in LEVELS19:
        xa = ISO.true_level_time(f, a * ISO.A_DEFAULT)
        err.append(float(np.interp(xa, t, mu)) - a * ISO.A_DEFAULT)
    return float(np.sqrt(np.mean(np.square(err))))


def _job_acq(job):
    nm, arm, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    reg, _ = M.run_active_learning(observe, domain=DOMAIN, seed=seed,
                                   n_init=N_INIT, n_iters=N_ITERS,
                                   acquisition=arm)
    t = M.eval_points(0.0, DOMAIN[1])
    mu, _ = reg.predict(t)
    yt = f(t)
    xs = np.asarray(reg.t_obs, float)
    return job, dict(grid_err=M.saturation_error(t, mu, yt),
                     exact_err=exact_level_error(f, t, mu),
                     n_below=int((xs[N_INIT:] < 0.0175).sum()),
                     min_active=float(xs[N_INIT:].min()))


def run_acq(argv):
    jobs = None
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    cache = os.path.join(DATA, "results_acqgrid.json")
    joblist = [(nm, arm, s) for nm in FAMILIES for arm in ACQ_ARMS for s in ACQ_SEEDS]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
        try:
            import psutil
            jobs = max(1, min(jobs, int(psutil.virtual_memory().available / 1e9 / 1.5)))
        except ImportError:
            jobs = min(jobs, 4)
    jobs = min(jobs, len(joblist))

    res = {nm: {arm: [None] * len(ACQ_SEEDS) for arm in ACQ_ARMS} for nm in FAMILIES}
    # recycling pool (poolutil.run_pool), as every other MCMC campaign uses
    recs = poolutil.run_pool(_job_acq, joblist, jobs, tasks_per_child=8,
                             label="acquisition-grid runs")
    for (nm, arm, seed), rec in recs.items():
        res[nm][arm][ACQ_SEEDS.index(seed)] = rec
    with open(cache, "w") as fh:
        json.dump(res, fh)

    print("\n=== 3.1  LINEAR vs GEOMETRIC acquisition candidate grid "
          f"({len(ACQ_SEEDS)} seeds, N=20) ===\n")
    print(f"  {'family':18s}{'linear':>16s}{'geometric':>16s}{'ratio':>9s}"
          f"{'pts<0.0175s (lin/geo)':>24s}")
    for nm in FAMILIES:
        a = np.array([r["grid_err"] for r in res[nm]["uncertainty"]])
        b = np.array([r["grid_err"] for r in res[nm]["uncertainty_geom"]])
        na = np.mean([r["n_below"] for r in res[nm]["uncertainty"]])
        nb = np.mean([r["n_below"] for r in res[nm]["uncertainty_geom"]])
        print(f"  {nm:18s}{a.mean():>10.3f}+/-{a.std():.3f}"
              f"{b.mean():>10.3f}+/-{b.std():.3f}{a.mean()/b.mean():>9.2f}"
              f"{na:>12.1f}{nb:>12.1f}")

    print("\n=== 3.2  metric (i): grid-located vs ROOT-FOUND crossings "
          "(linear-acquisition arm) ===\n")
    print(f"  {'family':18s}{'as reported (grid)':>22s}{'root-found (exact)':>22s}{'diff':>9s}")
    for nm in FAMILIES:
        g = np.array([r["grid_err"] for r in res[nm]["uncertainty"]])
        e = np.array([r["exact_err"] for r in res[nm]["uncertainty"]])
        print(f"  {nm:18s}{g.mean():>16.3f}+/-{g.std():.3f}"
              f"{e.mean():>16.3f}+/-{e.std():.3f}{e.mean()-g.mean():>+9.3f}")
    print("\nwrote", cache)


# ===========================================================================
# diag -- MCMC convergence diagnostics for the production penalized model
# ===========================================================================

def _diag_fit(t_obs, y_obs, seed, chains):
    Xb = M.ispline_basis(np.asarray(t_obs, float), KNOTS, DEG)
    mask = np.ones(len(t_obs), bool)
    mcmc = MCMC(NUTS(PRODUCTION_MODEL), num_warmup=500, num_samples=500, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), Xb=jnp.array(Xb), mask=jnp.array(mask),
             sigma_prior=0.3, tau_scale=0.3, y=jnp.array(np.asarray(y_obs, float)),
             extra_fields=("diverging",))
    return mcmc


def _diag_rhat_ess(mcmc):
    g = mcmc.get_samples(group_by_chain=True)
    g = {k: v for k, v in g.items() if k != "w"}          # exclude deterministic
    s = summary(g, prob=0.9)
    rhat = max(float(np.nanmax(s[k]["r_hat"])) for k in s)
    ess = min(float(np.nanmin(s[k]["n_eff"])) for k in s)
    return rhat, ess


def run_diag(argv):
    RES = _load_results()
    print("=== (1) 4-chain diagnostics, seed 0 ===")
    print(f"  {'family':18s}{'maxRhat':>9s}{'minESS':>8s}{'div':>5s}")
    per_fam = {}
    for fam in FAMILIES:
        r = RES["mc"][fam]["uncertainty"][0]
        mcmc = _diag_fit(r["t_obs"], r["y_obs"], 0, 4)
        rhat, ess = _diag_rhat_ess(mcmc)
        div = int(np.sum(mcmc.get_extra_fields()["diverging"]))
        jax.clear_caches()
        per_fam[fam] = dict(rhat=float(rhat), ess=float(ess), div=div)
        print(f"  {fam:18s}{rhat:>9.3f}{ess:>8.0f}{div:>5d}", flush=True)

    print("\n=== (2) single-chain across 250 N=20 uncertainty designs ===")
    tot_div = 0; any_div = 0; esss = []; per_fam_frac = {f: 0 for f in FAMILIES}
    n = 0
    for fam in FAMILIES:
        # Use each run's own sampler seed (the list is indexed by seed, see
        # make_figures.MC_SEEDS). Fixing the seed at 0 across all 250 designs
        # measured the divergence rate at a single initialization draw, which is
        # the one design that cannot see the initialization variability this
        # section's argument is about.
        for seed, r in enumerate(RES["mc"][fam]["uncertainty"]):
            mcmc = _diag_fit(r["t_obs"], r["y_obs"], seed, 1)
            div = int(np.sum(mcmc.get_extra_fields()["diverging"]))
            _, ess = _diag_rhat_ess(mcmc)
            tot_div += div; any_div += (div > 0); per_fam_frac[fam] += (div > 0)
            esss.append(ess); n += 1
            jax.clear_caches()
        print(f"  {fam} done", flush=True)
    esss = np.array(esss)
    print(f"\n  fits={n}  total divergences={tot_div}  "
          f"fits with any={any_div} ({100*any_div/n:.0f}%)")
    print(f"  per-family % with any: "
          + ", ".join(f"{f.split()[0]} {100*per_fam_frac[f]/len(RES['mc'][f]['uncertainty']):.0f}%" for f in FAMILIES))
    print(f"  single-chain ESS: min={esss.min():.0f}  median={np.median(esss):.0f}  (of 500 draws)")

    out = dict(per_family=per_fam,
               single_chain=dict(fits=n, total_div=tot_div, any_div=any_div,
                                 ess_min=float(esss.min()),
                                 ess_median=float(np.median(esss))),
               _meta=dict(families=FAMILIES, chains=4, warmup=500, draws=500,
                          seed=0, model="ISplineRegressor._MODEL (production "
                          "second-order log-random-walk penalty)"))
    with open(os.path.join(DATA, "results_diag.json"), "w") as fh:
        json.dump(out, fh)
    print("  wrote data/results_diag.json")


# ===========================================================================
# sampler -- production NUTS settings vs a fixed-step alternative
# (writes results_sampler_settings.json)
# ===========================================================================
CONFIGS = [("production", 0.8, 500, False), ("fixed", 0.95, 500, True)]


def _sampler_job(job):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_median
    from numpyro.diagnostics import summary
    import ispline_model as M

    nm, seed, chains = job
    KN = M.make_knots(ISO.DOMAIN, 21, 3)
    TG = M.eval_points(0.0, ISO.DOMAIN[1])
    BG = M.ispline_basis(TG, KN, 3)
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)
    r = R["mc"][nm]["uncertainty"][seed]
    Xb = M.ispline_basis(np.array(r["t_obs"]), KN, 3)
    mask = np.ones(len(r["t_obs"]), bool)
    yt = np.array(r["yt"])

    out = {}
    for label, tgt, warm, med in CONFIGS:
        kw = dict(target_accept_prob=tgt)
        if med:
            kw["init_strategy"] = init_to_median
        m = MCMC(NUTS(M.ISplineRegressor._MODEL, **kw), num_warmup=warm,
                 num_samples=500, num_chains=chains, chain_method="sequential",
                 progress_bar=False)
        m.run(jax.random.PRNGKey(seed), Xb=jnp.array(Xb), mask=jnp.array(mask),
              sigma_prior=0.3, tau_scale=0.3,
              y=jnp.array(np.array(r["y_obs"])), extra_fields=("diverging",))
        w = np.array(m.get_samples()["w"])
        mu = w @ BG.T
        rec = dict(mu=[float(v) for v in mu.mean(0)],
                   sat_err=float(M.saturation_error(TG, mu.mean(0), yt)),
                   div=int(np.sum(m.get_extra_fields()["diverging"])),
                   ndraw=int(chains * 500))
        if chains > 1:
            g = {k: v for k, v in m.get_samples(group_by_chain=True).items()
                 if k != "w"}
            s = summary(g, prob=0.9)
            rec["rhat"] = max(float(np.nanmax(s[k]["r_hat"])) for k in s)
            rec["ess"] = min(float(np.nanmin(s[k]["n_eff"])) for k in s)
        out[label] = rec
        jax.clear_caches()
    a = np.array(out["production"]["mu"]); b = np.array(out["fixed"]["mu"])
    out["max_dmu"] = float(np.max(np.abs(a - b)))
    return job, out


def run_sampler(argv):
    jobs, nseeds, chains = 5, 10, 1
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a.startswith("--chains="):
            chains = int(a.split("=", 1)[1])
    seeds = list(range(nseeds))
    joblist = [(nm, s, chains) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_sampler_job, joblist, jobs, tasks_per_child=6,
                             label="sampler-settings pairs")

    out = {f"{nm}|{s}": recs[(nm, s, chains)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, chains=chains,
                        configs=[list(c) for c in CONFIGS],
                        sigma_y=ISO.SIGMA_Y)
    with open(os.path.join(DATA, "results_sampler_settings.json"), "w") as fh:
        json.dump(out, fh)

    print(f"\nSAMPLER-SETTINGS COMPARISON COMPLETE in {time.time()-t0:.0f}s")
    print(f"  {'family':18s}{'sat err prod':>14s}{'sat err fixed':>15s}"
          f"{'div prod':>10s}{'div fixed':>11s}{'max|dmu|':>11s}")
    D = []
    for nm in FAMILIES:
        p = np.array([out[f"{nm}|{s}"]["production"]["sat_err"] for s in seeds])
        f = np.array([out[f"{nm}|{s}"]["fixed"]["sat_err"] for s in seeds])
        dp = np.mean([out[f"{nm}|{s}"]["production"]["div"] for s in seeds])
        df = np.mean([out[f"{nm}|{s}"]["fixed"]["div"] for s in seeds])
        d = np.array([out[f"{nm}|{s}"]["max_dmu"] for s in seeds]); D.append(d)
        print(f"  {nm:18s}{p.mean():>14.4f}{f.mean():>15.4f}"
              f"{dp:>10.1f}{df:>11.1f}{d.mean():>11.4f}")
    D = np.concatenate(D)
    print(f"\n  pooled max|dmu|: mean {D.mean():.4f}  worst {D.max():.4f}  "
          f"against sigma_y={ISO.SIGMA_Y}")
    print("  -> the cached campaigns stand if this is well under sigma_y and "
          "well under the\n     reported saturation errors (~0.039).")
    return 0




# ===========================================================================
# targets -- the target-acceptance / initialization probe
# (writes results_target_accept.json)
# ===========================================================================
TA_BG = M.ispline_basis(TG, KNOTS, 3)


def _ta_fit(t_obs, y_obs, target, chains, seed=0, warmup=500, init_median=False):
    Xb = M.ispline_basis(np.asarray(t_obs, float), KNOTS, 3)
    mask = np.ones(len(t_obs), bool)
    kw = dict(target_accept_prob=float(target))
    if init_median:
        from numpyro.infer.initialization import init_to_median
        kw["init_strategy"] = init_to_median
    kernel = NUTS(PRODUCTION_MODEL, **kw)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=500, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), Xb=jnp.array(Xb), mask=jnp.array(mask),
             sigma_prior=0.3, tau_scale=0.3,
             y=jnp.array(np.asarray(y_obs, float)), extra_fields=("diverging",))
    g = {k: v for k, v in mcmc.get_samples(group_by_chain=True).items() if k != "w"}
    s = summary(g, prob=0.9)
    rhat = max(float(np.nanmax(s[k]["r_hat"])) for k in s)
    ess = min(float(np.nanmin(s[k]["n_eff"])) for k in s)
    div = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    w = np.array(mcmc.get_samples()["w"])
    mu = (w @ TA_BG.T).mean(0)
    jax.clear_caches()
    return dict(rhat=rhat, ess=ess, div=div, mu=[float(v) for v in mu])


# (label, target, warmup, init_to_median).  Each row backs a sentence of
# Sampler configurations: the production configuration, the two that show raising the
# target alone strands the chain, the warmup control, and the fix.
TA_GRID = [
    ("production",        0.80,  500, False),
    ("target 0.95",       0.95,  500, False),
    ("target 0.95, 4x warmup", 0.95, 2000, False),
    ("target 0.99",       0.99,  500, False),
    ("fix: 0.95 + median", 0.95,  500, True),
    ("0.99 + median",     0.99,  500, True),
]


def run_targets(argv):
    chains = 4
    for a in argv:
        if a.startswith("--chains="):
            chains = int(a.split("=", 1)[1])
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)

    t0, out = time.time(), {}
    print(f"{'family':18s}{'configuration':24s}{'maxRhat':>10s}{'minESS':>9s}"
          f"{'div':>7s}{'div%':>8s}{'max|mu|':>11s}", flush=True)
    for fam in FAMILIES:
        r = R["mc"][fam]["uncertainty"][0]
        for label, tgt, warm, med in TA_GRID:
            d = _ta_fit(r["t_obs"], r["y_obs"], tgt, chains, warmup=warm,
                    init_median=med)
            mu = np.array(d["mu"])
            d.update(label=label, target=tgt, warmup=warm, init_median=med,
                     max_abs_mu=float(np.max(np.abs(mu))))
            out[f"{fam}|{label}"] = d
            print(f"{fam:18s}{label:24s}{d['rhat']:>10.3f}{d['ess']:>9.0f}"
                  f"{d['div']:>7d}{100*d['div']/(chains*500):>7.1f}%"
                  f"{d['max_abs_mu']:>11.3f}", flush=True)

    out["_meta"] = dict(families=FAMILIES, grid=[list(g) for g in TA_GRID],
                        chains=chains, draws=500, seed=0,
                        model="ISplineRegressor._MODEL (production, second-order)")
    with open(os.path.join(DATA, "results_target_accept.json"), "w") as fh:
        json.dump(out, fh)
    print(f"\nSAMPLER-CONFIGURATION PROBE COMPLETE in {time.time()-t0:.0f}s "
          f"-> data/results_target_accept.json")
    print("  a max|mu| far above 1 means the chain never reached the typical set")
    print("  (the response is normalised to a plateau of one).")
    return 0




RUNNERS = {"knots": run_knots, "priors": run_priors,
           "acqgrid": run_acq, "diag": run_diag,
           "sampler": run_sampler, "targets": run_targets}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in RUNNERS:
        print(f"design_choices.py: unknown subcommand {cmd!r}; "
              f"choose from {list(RUNNERS)}", file=sys.stderr)
        sys.exit(2)
    RUNNERS[cmd](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
