"""
generalization.py -- does the frozen configuration hold beyond the development
set: on curves it never saw, at randomized rise locations, and across survey
fractions. Four subcommands, each writing its own cache:

  subcommand  what it shows                                 cache / consumer
  ----------  ------------------------------------------    ------------------------------
  heldout     benchmark on a DISTRIBUTION of saturation     results_heldout.json
              curves the config never saw (mixtures +       (+ _uniform variant via
              Hill), I-spline vs the fair GP-log arm        HELDOUT_XSAT=uniform)
                                                            (recompute_stats heldout)
  randxsat    the acquisition comparison with a RANDOMIZED  results_randxsat.json
              x_sat per seed (fair adaptive-vs-grid test):  (+ _uniform via
              learning curves, N=20 error, noise crossing   RANDXSAT_XDIST=uniform)
                                                            (recompute_stats acq)
  survey      is the near-tie a property of the problem or  results_survey.json
              of the N0=5 survey init? sweep N0 in {3,5,8}  (recompute_stats survey)
  heldoutgppin  the origin-matched GP-log-pin arm on the    results_heldout_gppin.json and
              held-out set, both x_sat measures in one run  results_heldout_gppin_uniform.json
                                                            (recompute_stats heldout)

Usage:  python3 generalization.py <heldout|randxsat|survey|heldoutgppin> [--jobs=N] [--seeds=N]
        python3 generalization.py randxsat --fresh|--summary
'survey' reads the randxsat loguniform cache (run 'randxsat' first). The two
"uniform" change-of-measure controls are selected by the env vars above.
"""
import os
import sys
import json

import numpy as np
from scipy.optimize import brentq

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


def _set_threads():
    poolutil.pin_threads()


# ===========================================================================
# heldout -- benchmark on a distribution of curves the configuration never saw
# ===========================================================================
N_MIX, N_HILL = 200, 50
XSAT_RANGE = (0.2, 4.0)

# The default generator draws x_sat log-uniformly, the same measure the
# log-spaced basis is matched to, which is a confound for the generalization
# claim exactly as the randomized-location study identifies for the acquisition
# comparison. HELDOUT_XSAT=uniform draws x_sat uniformly on the same interval,
# the change-of-measure control, and writes a separate cache.
XSAT_DIST = os.environ.get("HELDOUT_XSAT", "loguniform")
HELDOUT_CACHE = os.path.join(DATA, "results_heldout.json" if XSAT_DIST != "uniform"
                             else "results_heldout_uniform.json")


def draw_spec(idx):
    """Picklable ground-truth spec for held-out curve `idx`.

    Uses a generator seeded on idx alone, independent of the run seed, so the
    curve set is fixed and reproducible while the noise is not.
    """
    g = np.random.default_rng(100000 + idx)
    if XSAT_DIST == "uniform":
        x_sat = float(g.uniform(XSAT_RANGE[0], XSAT_RANGE[1]))
    else:
        x_sat = float(np.exp(g.uniform(np.log(XSAT_RANGE[0]),
                                       np.log(XSAT_RANGE[1]))))
    if idx < N_MIX:
        K = int(g.integers(1, 9))
        a = g.dirichlet(np.ones(K))
        tau = np.exp(g.uniform(np.log(0.01), np.log(3.0), K))
        return dict(kind="mix", a=a.tolist(), tau=tau.tolist(), x_sat=x_sat)
    n = float(np.exp(g.uniform(np.log(1.0), np.log(4.0))))
    return dict(kind="hill", n=n, x_sat=x_sat)


def build(spec):
    """Rebuild the callable ground truth from a spec, rescaled so that
    f(x_sat) = 0.95 exactly and f(inf) = 1."""
    if spec["kind"] == "hill":
        n = spec["n"]
        ts = spec["x_sat"] / (0.95 / 0.05) ** (1.0 / n)

        def f(x):
            u = (np.maximum(np.asarray(x, float), 0.0) / ts) ** n
            return u / (1.0 + u)
        return f

    a = np.asarray(spec["a"], float)
    tau = np.asarray(spec["tau"], float)

    def base(x):
        x = np.maximum(np.asarray(x, float), 0.0)[..., None]
        return (a * (1.0 - np.exp(-x / tau))).sum(-1)

    # locate the unscaled 95% point, then stretch x so it lands on x_sat
    hi = 1.0
    while base(np.array([hi]))[0] < 0.95 and hi < 1e6:
        hi *= 2.0
    if base(np.array([hi]))[0] < 0.95:
        s = 1.0
    else:
        x95 = brentq(lambda z: float(base(np.array([z]))[0]) - 0.95, 0.0, hi)
        s = spec["x_sat"] / x95

    def f(x):
        return base(np.asarray(x, float) / s)
    return f


def _job_heldout(job):
    kind, idx, seed = job
    spec = draw_spec(idx)
    f = build(spec)
    observe = ISO.make_observe(f, domain=DOMAIN)
    yt = f(TG)

    if kind == "ispline":
        reg, _ = M.run_active_learning(observe, domain=DOMAIN, seed=seed,
                                       truth=f, acquisition="uncertainty")
        mu, _ = reg.predict(TG)
        # Measure monotonicity rather than assume it, so the cache records
        # evidence rather than an assumption.
        rep = reg.monotonicity_report()
        return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                         dips=0.0, multivalued=0.0, max_dip=0.0,
                         nonmono=int(rep["non_monotone"]),
                         n_draws=int(rep["n_draws"]),
                         x_sat=spec["x_sat"], gkind=spec["kind"],
                         # the design is kept so the GP arms can be fitted to
                         # identical data (gp_arms.py matched)
                         t_obs=[float(v) for v in reg.t_obs],
                         y_obs=[float(v) for v in reg.y_obs])

    egrid = GA.ell_grid("log")
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(5, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    for _ in range(15):
        _, sd_a, _, _ = GA._gp_fit(xs, ys, GA.ACQ, "log", False, egrid)
        xn = float(GA.ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    mu, _, _, _ = GA._gp_fit(xs, ys, TG, "log", False, egrid)
    nv = RO.negative_variation(mu, TG, sigma_y=ISO.SIGMA_Y)
    uc = RO.upcrossings(mu, TG, 0.95)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                     dips=float(nv["n_descending_intervals"] > 0),
                     multivalued=float(uc["count"] > 1),
                     max_dip=nv["max_dip_depth"],
                     x_sat=spec["x_sat"], gkind=spec["kind"])



def _devset_reference():
    """Pooled development-set I-spline saturation error, read from results.json.

    Read from the cache rather than hardcoded, so it cannot drift from the
    campaign; None if the cache is absent.
    """
    path = os.path.join(DATA, "results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            R = json.load(fh)["mc"]
    except (ValueError, KeyError, OSError):
        return None
    v = [r["sat_err"] for nm in R for r in R[nm].get("uncertainty", [])]
    return float(np.mean(v)) if v else None


def run_heldout(argv):
    jobs, ncurves = None, N_MIX + N_HILL
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--curves="):
            ncurves = int(a.split("=", 1)[1])
    joblist = [(k, i, i) for k in ("ispline", "gp_log") for i in range(ncurves)]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))
    _set_threads()

    print(f"Running {len(joblist)} held-out runs on {jobs} workers ...", flush=True)
    # recycling pool: 250 held-out curves x the heavier MCMC loop would balloon the
    # Windows page file in a non-recycled pool.
    recs = poolutil.run_pool(_job_heldout, joblist, jobs, tasks_per_child=8,
                             label="held-out runs")

    out = {k: [recs[(k, i, i)] for i in range(ncurves)]
           for k in ("ispline", "gp_log")}
    out["_meta"] = dict(n_mix=N_MIX, n_hill=N_HILL, ncurves=ncurves,
                        xsat_range=list(XSAT_RANGE))
    with open(HELDOUT_CACHE, "w") as fh:
        json.dump(out, fh)

    def C(k, key="sat_err", gk=None):
        rs = out[k] if gk is None else [r for r in out[k] if r["gkind"] == gk]
        return np.array([r[key] for r in rs], float)

    print(f"\n=== HELD-OUT benchmark: {ncurves} curves the configuration "
          f"never saw ===")
    print(f"  {'generator':12s}{'n':>5s}{'surrogate':>12s}{'mean':>9s}"
          f"{'median':>9s}{'p90':>9s}{'max':>9s}{'> sigma_y %':>13s}")
    for gk, lab in (("mix", "mixture"), ("hill", "Hill")):
        for k in ("ispline", "gp_log"):
            v = C(k, "sat_err", gk)
            if not v.size:
                continue
            print(f"  {lab:12s}{v.size:>5d}{k:>12s}{v.mean():>9.3f}"
                  f"{np.median(v):>9.3f}{np.quantile(v,0.9):>9.3f}"
                  f"{v.max():>9.3f}{100*np.mean(v>ISO.SIGMA_Y):>13.0f}")
    print()
    for k in ("ispline", "gp_log"):
        v = C(k)
        print(f"  {'ALL':12s}{v.size:>5d}{k:>12s}{v.mean():>9.3f}"
              f"{np.median(v):>9.3f}{np.quantile(v,0.9):>9.3f}"
              f"{v.max():>9.3f}{100*np.mean(v>ISO.SIGMA_Y):>13.0f}")
    a, b = C("ispline"), C("gp_log")
    print(f"\n  paired: I-spline better on {100*np.mean(a<b):.0f}% of curves; "
          f"mean ratio GP/I-spline = {np.mean(b/a):.2f}, median = {np.median(b/a):.2f}")
    ref = _devset_reference()
    print("  development-set mean for reference: I-spline "
          + (f"{ref:.3f}" if ref is not None else "n/a (results.json absent)"))
    print(f"  GP-log unconstrained: non-monotone mean {100*C('gp_log','dips').mean():.0f}%, "
          f"multi-valued read-out {100*C('gp_log','multivalued').mean():.0f}%")
    print("\nwrote", HELDOUT_CACHE)


# ===========================================================================
# randxsat -- acquisition comparison with a randomized saturation location per
# seed (holds curve SHAPE fixed, re-solves each family's SCALE from x_sat)
# ===========================================================================
MC_SEEDS = list(range(50))
N_INIT = 5
N_ITERS = 15
METHODS = ["uncertainty", "random", "random_log", "grid"]
SIGMA_Y = ISO.SIGMA_Y
XSAT_LO, XSAT_HI = 0.2, 4.0

# Distribution of the drawn x_sat.  "loguniform" (default) matches the log
# domain; "uniform" puts more mass at large x_sat, where a log-spaced grid is
# sparse, to check whether the grid's structural match to a log-uniform ensemble
# is what produced the near-tie.
XDIST = os.environ.get("RANDXSAT_XDIST", "loguniform")
RANDXSAT_CACHE = os.path.join(DATA, "results_randxsat.json" if XDIST == "loguniform"
                              else f"results_randxsat_{XDIST}.json")

# Fixed SHAPE per display name; the builders re-solve the SCALE from x_sat.
SHAPE = {
    "Langmuir":     ("Langmuir", {}),
    "Dissociative": ("Dissociative", {}),
    "Sips (n=2)":   ("Sips", {"n": 2.0}),
    "Elovich":      ("Elovich", {"gamma": 1.69415}),
    "D-R KWW":      ("DR-KWW", {"beta": 0.55}),
}
ORDER = list(SHAPE)


def xsat_for_seed(seed):
    """Deterministic per-seed x_sat, shared across families so that at each
    seed all five curves sit at the same (random) saturation location."""
    rng = np.random.default_rng(1_000_000 + seed)
    u = rng.random()
    if XDIST == "uniform":
        return float(XSAT_LO + (XSAT_HI - XSAT_LO) * u)
    return float(XSAT_LO * (XSAT_HI / XSAT_LO) ** u)


def _job_randxsat(job):
    """One (family, method, seed) run; returns the learning-curve history."""
    name, method, seed = job
    fam, shape_kw = SHAPE[name]
    xs = xsat_for_seed(seed)
    f = ISO.make_isotherm(fam, x_sat=xs, **shape_kw)
    observe = ISO.make_observe(f, domain=DOMAIN,
                               sigma_x=ISO.SIGMA_X, sigma_y=ISO.SIGMA_Y)
    reg, hist = M.run_active_learning(
        observe, domain=DOMAIN, seed=seed, n_init=N_INIT, n_iters=N_ITERS,
        record_history=True, truth=f, acquisition=method)
    return job, dict(x_sat=xs, history=hist,
                     mono_bad=getattr(reg, "mono_bad_total", None),
                     mono_draws=getattr(reg, "mono_draws_total", None))


def _randxsat_job_list():
    return [(name, method, seed)
            for name in ORDER for method in METHODS for seed in MC_SEEDS]


def _randxsat_run(fresh=False, jobs=None):
    tag = {"seeds": MC_SEEDS, "n_init": N_INIT, "n_iters": N_ITERS,
           "methods": METHODS, "xsat_range": [XSAT_LO, XSAT_HI],
           "design": "randomized-x_sat acquisition comparison"}
    if not fresh and os.path.exists(RANDXSAT_CACHE):
        with open(RANDXSAT_CACHE) as fh:
            cached = json.load(fh)
        if cached.get("tag") == tag:
            return cached

    res = {"tag": tag,
           "mc": {nm: {m: [None] * len(MC_SEEDS) for m in METHODS}
                  for nm in ORDER}}
    job_list = _randxsat_job_list()

    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            jobs = max(1, min(jobs, int(avail_gb / 1.5)))
        except ImportError:
            jobs = min(jobs, 4)
    jobs = min(jobs, len(job_list))

    print(f"Running {len(job_list)} randomized-x_sat runs on {jobs} workers ...",
          flush=True)
    # recycling pool: this campaign is as heavy as the main one (MCMC loops over
    # many seeds), so a non-recycled pool would balloon the Windows page file.
    recs = poolutil.run_pool(_job_randxsat, job_list, jobs, tasks_per_child=8,
                             label="randomized-x_sat runs")
    for job, rec in recs.items():
        nm, method, seed = job
        res["mc"][nm][method][MC_SEEDS.index(seed)] = rec

    with open(RANDXSAT_CACHE, "w") as fh:
        json.dump(res, fh)
    return res


def _randxsat_summarize(res):
    Ns = np.arange(N_INIT, N_INIT + N_ITERS + 1)
    j20 = list(Ns).index(20)

    def cross_per_seed(H):
        out = []
        for row in H:
            idx = np.where(np.array(row) < SIGMA_Y)[0]
            out.append(Ns[idx[0]] if len(idx) else np.nan)
        return np.array(out, float)

    xs_all = [res["mc"][ORDER[0]]["uncertainty"][i]["x_sat"] for i in range(len(MC_SEEDS))]
    print(f"\n=== RANDOMIZED x_sat in [{XSAT_LO},{XSAT_HI}] s, "
          f"drawn {XDIST} per seed (median {np.median(xs_all):.2f}, "
          f"range [{min(xs_all):.2f},{max(xs_all):.2f}]) ===")
    print(f"=== {len(MC_SEEDS)} seeds, methods={METHODS} ===\n")

    bad = draws = 0
    for nm in ORDER:
        print(f"  {nm}:")
        Hu = np.array([r["history"] for r in res["mc"][nm]["uncertainty"]])
        for method in METHODS:
            H = np.array([r["history"] for r in res["mc"][nm][method]])
            e20 = H[:, j20]
            cr = cross_per_seed(H)
            frac = np.mean(~np.isnan(cr))
            med = np.nanmedian(cr) if frac > 0 else np.nan
            ratio = e20.mean() / Hu[:, j20].mean()
            medstr = "NA" if np.isnan(med) else str(int(med))
            print(f"    {method:12s} N20={e20.mean():.3f}+/-{e20.std():.3f}  "
                  f"cross median N={medstr} in {frac*100:.0f}% of seeds   "
                  f"(err/active={ratio:.2f}x)")
        for method in METHODS:
            for r in res["mc"][nm][method]:
                bad += r["mono_bad"] or 0
                draws += r["mono_draws"] or 0
    print(f"\n=== MONOTONICITY (randomized study): {bad}/{draws} non-monotone ===")


def run_randxsat(argv):
    if "--summary" in argv:
        with open(RANDXSAT_CACHE) as fh:
            _randxsat_summarize(json.load(fh))
        return
    fresh = "--fresh" in argv or not os.path.exists(RANDXSAT_CACHE)
    jobs = None
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    res = _randxsat_run(fresh=fresh, jobs=jobs)
    _randxsat_summarize(res)


# ===========================================================================
# survey -- is the near-tie a property of the problem or of the N0=5 survey
# init? sweep N0 in {3,5,8} at fixed budget N=20, at the randomized location
# ===========================================================================
N0_GRID = [3, 5, 8]
N_TOTAL = 20


def _job_survey(job):
    n0, nm, seed = job
    fam, shape_kw = SHAPE[nm]
    f = ISO.make_isotherm(fam, x_sat=xsat_for_seed(seed), **shape_kw)
    observe = ISO.make_observe(f, domain=DOMAIN)
    yt = f(TG)
    reg, _ = M.run_active_learning(observe, domain=DOMAIN, seed=seed,
                                   n_init=n0, n_iters=N_TOTAL - n0,
                                   truth=f, acquisition="uncertainty")
    mu, _ = reg.predict(TG)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)))


def run_survey(argv):
    jobs, nseeds = None, 50
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_survey.json")
    # N0 = 5 is the cached randomized-location campaign under the identical
    # protocol (randxsat, uncertainty arm); only 3 and 8 are new.
    joblist = [(n0, nm, s) for n0 in N0_GRID if n0 != 5
               for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))
    _set_threads()

    print(f"Running {len(joblist)} survey-fraction runs on {jobs} workers ...",
          flush=True)
    recs = poolutil.run_pool(_job_survey, joblist, jobs, tasks_per_child=8,
                             label="survey-fraction runs")

    # log-grid reference and the cached N0 = 5 arm
    with open(os.path.join(DATA, "results_randxsat.json")) as fh:
        RXJ = json.load(fh)
    for nm in FAMILIES:
        for s in seeds:
            recs[(5, nm, s)] = dict(
                sat_err=float(RXJ["mc"][nm]["uncertainty"][s]["history"][-1]))

    out = {f"{n0}|{nm}": [recs[(n0, nm, s)] for s in seeds]
           for n0 in N0_GRID for nm in FAMILIES}
    out["_meta"] = dict(seeds=seeds, n0_grid=N0_GRID, n_total=N_TOTAL,
                        design="randomized x_sat, log-uniform")
    with open(cache, "w") as fh:
        json.dump(out, fh)

    def C(n0, nm):
        return np.array([r["sat_err"] for r in out[f"{n0}|{nm}"]], float)

    print("\n=== saturation error at N=20 vs survey fraction N0 "
          "(randomized rise location) ===")
    print(f"  {'family':18s}" + "".join(f"{'N0='+str(n):>10s}" for n in N0_GRID)
          + f"{'log grid':>11s}")

    def gridref(nm):
        return float(np.mean([r["history"][-1] for r in RXJ["mc"][nm]["grid"]]))

    for nm in FAMILIES:
        print(f"  {nm:18s}" + "".join(f"{C(n,nm).mean():>10.3f}" for n in N0_GRID)
              + f"{gridref(nm):>11.3f}")
    grids = float(np.mean([gridref(nm) for nm in FAMILIES]))
    print(f"  {'POOLED':18s}"
          + "".join(f"{np.mean([C(n,nm).mean() for nm in FAMILIES]):>10.3f}"
                    for n in N0_GRID) + f"{grids:>11.3f}")
    print("\nwrote", cache)



# ===========================================================================
# heldoutgppin -- the origin-matched GP-log-pin arm on the held-out set. Was
# heldout_gppin.py; writes BOTH results_heldout_gppin.json and
# results_heldout_gppin_uniform.json in one invocation, which is why it keeps
# its own measure-parameterised spec builder instead of using draw_spec (which
# takes the measure from the environment).
# ===========================================================================
GPPIN_NCURVES = N_MIX + N_HILL


def _gppin_spec_for(measure, idx):
    """Replicate generalization.draw_spec but with the measure passed in, so a
    single process can build both the log-uniform and uniform curve sets."""
    g = np.random.default_rng(100000 + idx)
    if measure == "uniform":
        x_sat = float(g.uniform(XSAT_RANGE[0], XSAT_RANGE[1]))
    else:
        x_sat = float(np.exp(g.uniform(np.log(XSAT_RANGE[0]),
                                       np.log(XSAT_RANGE[1]))))
    if idx < N_MIX:
        K = int(g.integers(1, 9))
        a = g.dirichlet(np.ones(K))
        tau = np.exp(g.uniform(np.log(0.01), np.log(3.0), K))
        return dict(kind="mix", a=a.tolist(), tau=tau.tolist(), x_sat=x_sat)
    n = float(np.exp(g.uniform(np.log(1.0), np.log(4.0))))
    return dict(kind="hill", n=n, x_sat=x_sat)


def _gppin_heldout_job(args):
    measure, idx, seed = args
    spec = _gppin_spec_for(measure, idx)
    f = build(spec)
    observe = ISO.make_observe(f, domain=DOMAIN)
    yt = f(TG)
    egrid = GA.ell_grid("log")
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(5, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    for _ in range(15):
        _, sd_a, _, _ = GA._gp_fit(xs, ys, GA.ACQ, "log", True, egrid)  # pin=True
        xn = float(GA.ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    mu, _, _, _ = GA._gp_fit(xs, ys, TG, "log", True, egrid)
    return args, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                      x_sat=spec["x_sat"], gkind=spec["kind"])


def run_heldout_gppin(argv):
    jobs = max(1, (os.cpu_count() or 2) - 2)
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    _set_threads()
    for measure, cache in (("loguniform", "results_heldout_gppin.json"),
                           ("uniform", "results_heldout_gppin_uniform.json")):
        joblist = [(measure, i, i) for i in range(GPPIN_NCURVES)]
        print(f"[{measure}] {len(joblist)} GP-log-pin held-out runs on "
              f"{min(jobs, len(joblist))} workers ...", flush=True)
        recs = poolutil.run_pool(_gppin_heldout_job, joblist, min(jobs, len(joblist)),
                                 tasks_per_child=16, label=f"gppin/{measure}")
        out = {"gp_log_pin": [recs[(measure, i, i)] for i in range(GPPIN_NCURVES)],
               "_meta": dict(n_mix=N_MIX, n_hill=N_HILL, ncurves=GPPIN_NCURVES,
                             measure=measure)}
        with open(os.path.join(DATA, cache), "w") as fh:
            json.dump(out, fh)
        v = np.array([r["sat_err"] for r in out["gp_log_pin"]])
        print(f"[{measure}] wrote {cache}: mean {v.mean():.3f} "
              f"median {np.median(v):.3f} p90 {np.quantile(v,0.9):.3f} "
              f"max {v.max():.3f} >sy {100*np.mean(v>ISO.SIGMA_Y):.0f}%", flush=True)
    return 0




RUNNERS = {"heldout": run_heldout, "randxsat": run_randxsat,
           "survey": run_survey, "heldoutgppin": run_heldout_gppin}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in RUNNERS:
        print(f"generalization.py: unknown subcommand {cmd!r}; "
              f"choose from {list(RUNNERS)}", file=sys.stderr)
        sys.exit(2)
    RUNNERS[cmd](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
