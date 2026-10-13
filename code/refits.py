"""refits.py -- refit the stored N=20 designs under a different model,
likelihood, or functional form. Four subcommands, each keeping the cache
filename its predecessor wrote, so every reader is unaffected:

  refits.py penalty     penalty-order + model-variant ablation
                        -> results_model_ablation.json
  refits.py studentt    Gaussian vs Student-t likelihood, clean and contaminated
                        -> results_studentt.json
  refits.py parametric  least-squares fits of the five closed forms
                        -> results_parametric.json
  refits.py unpen       the two unpenalized monotone bases (P=25, P=11) refit on
                        the stored designs of every acquisition strategy, the
                        matched-design control for those arms (scorecard figure)
                        -> results_unpen_matched.json

The first three read the same stored uncertainty designs from results.json, which
is why they share _designs() below; it was copied verbatim into three files before.
`unpen` reads the designs of all four strategies.

Usage:  python3 refits.py <penalty|studentt|parametric|unpen> [--jobs=N] [--seeds=N]
        add --smoke for a single-fit sanity check.
"""
import os
import sys
import json
import time
import gc
import warnings

import numpy as np
import jax
from scipy.optimize import curve_fit
from scipy.optimize import brentq

import ispline_model as M
import isotherms as ISO
import poolutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOMAIN = ISO.DOMAIN
TG = M.eval_points(0.0, DOMAIN[1])
FAMILIES = list(ISO.STANDARD_SPECS)
TRUE_XSAT = 2.0
ALPHA = 0.95
XM = DOMAIN[1]


def _designs(seeds):
    """The production arm's stored uncertainty designs (t_obs, y_obs) per
    (family, seed), read from the main-campaign cache."""
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)
    d = {}
    for nm in FAMILIES:
        recs = R["mc"][nm]["uncertainty"]
        for s in seeds:
            rec = recs[s]
            d[(nm, s)] = (list(rec["t_obs"]), list(rec["y_obs"]))
    return d


# ===========================================================================
# penalty -- penalty order and the model variants
# ===========================================================================
PEN_WARMUP, PEN_DRAWS = 500, 500

# The penalty-order ablation refits the same stored designs under four model
# variants: the first-order penalty (the pre-adoption configuration), the
# heteroscedastic-likelihood and plateau-parameter variants of the production
# second-order prior (each differing from production in ONE feature), and the
# production second-order penalty itself. The keys below are also the arm keys
# in the shipped cache (results_model_ablation.json).
PEN_ARMS = {"first_order": M.FirstOrderRegressor, "hetero": M.HeteroRegressor,
        "plateau": M.PlateauRegressor, "second_order": M.SecondOrderRegressor}


def _pen_fit_eval(cls, t_obs, y_obs, yt, seed, warmup, draws):
    reg = cls(domain=DOMAIN, n_init=5, n_iters=15, warmup=warmup, draws=draws)
    for x, y in zip(t_obs, y_obs):
        reg.add(float(x), float(y))
    reg.fit(seed=seed)
    sat = float(M.saturation_error(TG, reg.predict(TG)[0], yt))
    cr = reg.tstar(ALPHA)
    med = float(np.median(cr))
    lo, hi = np.percentile(cr, [2.5, 97.5])
    return dict(sat_err=sat, bias=med - TRUE_XSAT,
                covered=bool(lo <= TRUE_XSAT <= hi),
                nonmono=int(reg.monotonicity_report()["non_monotone"]))


def _pen_job(job):
    nm, seed, t_obs, y_obs = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    yt = f(TG)
    out = {arm: _pen_fit_eval(cls, t_obs, y_obs, yt, seed, PEN_WARMUP, PEN_DRAWS)
           for arm, cls in PEN_ARMS.items()}
    jax.clear_caches()
    gc.collect()
    return (nm, seed), out


def run_penalty(argv):
    global PEN_WARMUP, PEN_DRAWS
    jobs, seeds = 3, list(range(50))
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    designs = _designs(seeds)

    if smoke:
        PEN_WARMUP, PEN_DRAWS = 50, 50
        nm = FAMILIES[0]
        _, out = _pen_job((nm, 0, *designs[(nm, 0)]))
        for arm, r in out.items():
            assert r["nonmono"] == 0, f"{arm} not monotone"
            print(f"  smoke {arm:12s} sat={r['sat_err']:.3f} bias={r['bias']:+.2f} cov={r['covered']}")
        print("SMOKE OK")
        return

    joblist = [(nm, s, *designs[(nm, s)]) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_pen_job, joblist, jobs, tasks_per_child=6, label="model-ablation")
    out = {f"{nm}|{s}": recs[(nm, s)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, arms=list(PEN_ARMS),
                        true_xsat=TRUE_XSAT, alpha=ALPHA, domain=list(DOMAIN),
                        method="paired refit on the production arm's stored uncertainty designs")
    with open(os.path.join(DATA, "results_model_ablation.json"), "w") as fh:
        json.dump(out, fh)

    print(f"MODEL-ABLATION COMPLETE in {time.time()-t0:.0f}s. "
          f"(pooled over {len(FAMILIES)} families x {len(seeds)} seeds)")
    for arm in PEN_ARMS:
        se = np.mean([recs[(nm, s)][arm]["sat_err"] for nm in FAMILIES for s in seeds])
        bi = np.mean([recs[(nm, s)][arm]["bias"] for nm in FAMILIES for s in seeds])
        co = 100 * np.mean([recs[(nm, s)][arm]["covered"] for nm in FAMILIES for s in seeds])
        nz = sum(recs[(nm, s)][arm]["nonmono"] for nm in FAMILIES for s in seeds)
        print(f"  {arm:12s} sat_err {se:.3f}  x_sat bias {bi:+.2f}  cov {co:.0f}%  nonmono {nz}")



# ===========================================================================
# unpen -- the two unpenalized monotone bases refit on the stored designs of
# every acquisition strategy (fills the unpenalized rows of the scorecard
# figure; the matched-design control for those arms) -> results_unpen_matched.json
# ===========================================================================
UNPEN_ARMS = {"unpen25": 21, "unpen11": 7}          # arm -> n_interior (P = n + 4)
UNPEN_STRATEGIES = ["uncertainty", "grid", "random_log", "random"]


def _all_designs(seeds):
    """Stored designs of every strategy, keyed (strategy, family, seed)."""
    with open(os.path.join(DATA, "results.json")) as fh:
        R = json.load(fh)
    return {(st, nm, s): (list(R["mc"][nm][st][s]["t_obs"]), list(R["mc"][nm][st][s]["y_obs"]))
            for st in UNPEN_STRATEGIES for nm in FAMILIES for s in seeds}


def _unpen_job(job):
    st, nm, seed, t_obs, y_obs = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    yt = ISO.make_isotherm(fam, **kw)(TG)
    out = {}
    for arm, n_int in UNPEN_ARMS.items():
        reg = M.UnpenalizedRegressor(domain=DOMAIN, n_interior=n_int, n_init=5, n_iters=15)
        for x, y in zip(t_obs, y_obs):
            reg.add(float(x), float(y))
        reg.fit(seed=seed)
        out[arm] = dict(sat_err=float(M.saturation_error(TG, reg.predict(TG)[0], yt)))
    jax.clear_caches()
    gc.collect()
    return (st, nm, seed), out


def run_unpen(argv):
    jobs, seeds = 2, list(range(50))
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    designs = _all_designs(seeds)
    joblist = [(st, nm, s, *designs[(st, nm, s)]) for st in UNPEN_STRATEGIES for nm in FAMILIES for s in seeds]
    t0 = time.time()
    print(f"unpenalized bases on the stored designs: {len(joblist)} designs x {len(UNPEN_ARMS)} arms "
          f"on {jobs} workers ...", flush=True)
    recs = poolutil.run_pool(_unpen_job, joblist, jobs, tasks_per_child=6, label="unpen refits")
    out = {arm: {st: {nm: [recs[(st, nm, s)][arm]["sat_err"] for s in seeds] for nm in FAMILIES}
                 for st in UNPEN_STRATEGIES} for arm in UNPEN_ARMS}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, strategies=UNPEN_STRATEGIES,
                        arms={a: dict(n_interior=n, P=n + 4) for a, n in UNPEN_ARMS.items()},
                        method="unpenalized monotone bases refit on the stored designs of every strategy")
    with open(os.path.join(DATA, "results_unpen_matched.json"), "w") as fh:
        json.dump(out, fh)
    print(f"UNPEN REFITS COMPLETE in {time.time()-t0:.0f}s")
    print(f"  {'design':14s}" + "".join(f"{a:>10s}" for a in UNPEN_ARMS))
    for st in UNPEN_STRATEGIES:
        print(f"  {st:14s}" + "".join(f"{np.mean([v for nm in FAMILIES for v in out[a][st][nm]]):>10.4f}" for a in UNPEN_ARMS))


# ===========================================================================
# studentt -- Gaussian vs Student-t likelihood under a corrupted anchor
# ===========================================================================
ST_WARMUP, ST_DRAWS = 500, 500
ST_ARMS = {"gauss": M.ISplineRegressor, "studentt": M.StudentTRegressor}


def _st_fit_eval(cls, t_obs, y_obs, yt, seed):
    reg = cls(domain=DOMAIN, n_init=5, n_iters=15, warmup=ST_WARMUP, draws=ST_DRAWS)
    for x, y in zip(t_obs, y_obs):
        reg.add(float(x), float(y))
    reg.fit(seed=seed)
    mu = reg.predict(TG)[0]
    cr = reg.tstar(ALPHA)
    return dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                f_xmax=float(mu[-1]),
                x_sat=float(np.median(cr)),
                nonmono=int(reg.monotonicity_report()["non_monotone"]))


def _contaminate(t_obs, y_obs):
    """Bump the single reading nearest x_max by +3 sigma_y (robustness.py scenario)."""
    y = list(y_obs)
    i = int(np.argmin(np.abs(np.asarray(t_obs, float) - DOMAIN[1])))
    y[i] = y[i] + 3.0 * ISO.SIGMA_Y
    return y


def _st_job(job):
    nm, seed, t_obs, y_obs = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    yt = f(TG)
    y_bad = _contaminate(t_obs, y_obs)
    out = {}
    for arm, cls in ST_ARMS.items():
        out[f"{arm}|clean"] = _st_fit_eval(cls, t_obs, y_obs, yt, seed)
        out[f"{arm}|contam"] = _st_fit_eval(cls, t_obs, y_bad, yt, seed)
    jax.clear_caches()
    gc.collect()
    return (nm, seed), out


def run_studentt(argv):
    global ST_WARMUP, ST_DRAWS
    jobs, seeds = 3, list(range(50))
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    designs = _designs(seeds)

    if smoke:
        ST_WARMUP, ST_DRAWS = 60, 60
        nm = FAMILIES[0]
        _, out = _st_job((nm, 0, *designs[(nm, 0)]))
        for k, r in out.items():
            assert r["nonmono"] == 0, f"{k} not monotone"
            print(f"  smoke {k:18s} sat={r['sat_err']:.3f} f_xmax={r['f_xmax']:.3f} x_sat={r['x_sat']:.2f}")
        print("SMOKE OK")
        return

    joblist = [(nm, s, *designs[(nm, s)]) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_st_job, joblist, jobs, tasks_per_child=6, label="studentt")
    out = {f"{nm}|{s}": recs[(nm, s)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, true_xsat=TRUE_XSAT,
                        alpha=ALPHA, domain=list(DOMAIN), nu=M._STUDENTT_NU,
                        contamination="+3 sigma_y on the x_max reading",
                        method="refit results.json uncertainty designs, likelihood-only")
    with open(os.path.join(DATA, "results_studentt.json"), "w") as fh:
        json.dump(out, fh)

    def pool(arm, cond, k):
        return np.array([recs[(nm, s)][f"{arm}|{cond}"][k]
                         for nm in FAMILIES for s in seeds], float)
    print(f"STUDENTT-ROBUST COMPLETE in {time.time()-t0:.0f}s "
          f"({len(FAMILIES)} families x {len(seeds)} seeds)")
    print(f"  {'':20s}{'sat_err':>9s}{'f_xmax':>9s}{'x_sat':>8s}")
    for arm in ST_ARMS:
        for cond in ("clean", "contam"):
            se, fx, xs = (pool(arm, cond, "sat_err").mean(),
                          pool(arm, cond, "f_xmax").mean(),
                          np.median(pool(arm, cond, "x_sat")))
            print(f"  {arm+' '+cond:20s}{se:>9.3f}{fx:>9.3f}{xs:>8.2f}")
    gm = pool("gauss", "contam", "sat_err").mean()
    tm = pool("studentt", "contam", "sat_err").mean()
    print(f"  -> under contamination Student-t/Gaussian curve error {tm/gm:.2f}x; "
          f"plateau f_xmax {pool('studentt','contam','f_xmax').mean():.3f} vs "
          f"{pool('gauss','contam','f_xmax').mean():.3f} (true 1.0)")



# ===========================================================================
# parametric -- least-squares fits of the five closed forms (the oracle bound)
# ===========================================================================

def _relu(x):
    return np.maximum(x, 0.0)


# parametric forms: f(x; A, *shape). Fit A and the shape parameter(s) freely.
def _langmuir(x, A, tau):
    return A * (1.0 - np.exp(-_relu(x) / tau))


def _mm(x, A, K):
    return A * _relu(x) / (K + _relu(x))


def _sips(x, A, n, ts):
    u = (_relu(x) / ts) ** n
    return A * u / (1.0 + u)


def _elovich(x, A, g, tE):
    return np.minimum(A, (A / g) * np.log1p(_relu(x) / tE))


def _kww(x, A, beta, tp):
    return A * (1.0 - np.exp(-(_relu(x) / tp) ** beta))


FORMS = {
    "Langmuir": (_langmuir, [1.0, 1.0], ([1e-3, 1e-3], [10, 100])),
    "Dissociative": (_mm, [1.0, 0.5], ([1e-3, 1e-3], [10, 100])),
    "Sips": (_sips, [1.0, 2.0, 1.0], ([1e-3, 0.3, 1e-3], [10, 12, 100])),
    "Elovich": (_elovich, [1.0, 3.0, 0.5], ([1e-3, 0.2, 1e-3], [10, 30, 100])),
    "DR-KWW": (_kww, [1.0, 0.6, 1.0], ([1e-3, 0.2, 1e-3], [10, 3, 100])),
}
# map display family -> its true form key
TRUEFORM = {"Langmuir": "Langmuir", "Dissociative": "Dissociative",
            "Sips (n=2)": "Sips", "Elovich": "Elovich", "D-R KWW": "DR-KWW"}


def _fit_and_eval(form_key, xo, yo, yt):
    fn, p0, bounds = FORMS[form_key]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fn, xo, yo, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    mu = fn(TG, *popt)
    sat = float(M.saturation_error(TG, mu, yt))
    A = mu[-1]
    tgt = ALPHA * A
    try:
        if fn(np.array([XM]), *popt)[0] < tgt or fn(np.array([1e-9]), *popt)[0] > tgt:
            xs = float("nan")
        else:
            xs = brentq(lambda z: fn(np.array([z]), *popt)[0] - tgt, 1e-9, XM)
    except Exception:
        xs = float("nan")
    return dict(sat_err=sat, x_sat=float(xs))


def _par_job(job):
    nm, seed, t_obs, y_obs = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    yt = f(TG)
    xo, yo = np.asarray(t_obs, float), np.asarray(y_obs, float)
    out = {}
    for fk in FORMS:
        out[fk] = _fit_and_eval(fk, xo, yo, yt)
    return (nm, seed), out


def run_parametric(argv):
    jobs, seeds = 4, list(range(50))
    smoke = "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    designs = _designs(seeds)

    if smoke:
        for nm in (FAMILIES[0], FAMILIES[2]):
            _, out = _par_job((nm, 0, *designs[(nm, 0)]))
            tf = TRUEFORM[nm]
            orc = out[tf]
            wrong = [out[fk]["sat_err"] for fk in FORMS if fk != tf and out[fk]]
            print(f"  smoke {nm:12s} oracle({tf}) sat={orc['sat_err']:.3f} "
                  f"x_sat={orc['x_sat']:.2f} | best-wrong sat={min(wrong):.3f}")
        print("SMOKE OK")
        return

    joblist = [(nm, s, *designs[(nm, s)]) for nm in FAMILIES for s in seeds]
    t0 = time.time()
    recs = poolutil.run_pool(_par_job, joblist, jobs, tasks_per_child=25, label="parametric")
    out = {f"{nm}|{s}": recs[(nm, s)] for nm in FAMILIES for s in seeds}
    out["_meta"] = dict(families=FAMILIES, seeds=seeds, forms=list(FORMS),
                        trueform=TRUEFORM, true_xsat=TRUE_XSAT, alpha=ALPHA,
                        method="least-squares parametric fit on results.json designs")
    with open(os.path.join(DATA, "results_parametric.json"), "w") as fh:
        json.dump(out, fh)

    def orc(nm, k):
        tf = TRUEFORM[nm]
        return np.array([recs[(nm, s)][tf][k] for s in seeds
                         if recs[(nm, s)][tf]], float)

    def wrong(nm, k):
        tf = TRUEFORM[nm]
        v = []
        for s in seeds:
            cand = [recs[(nm, s)][fk][k] for fk in FORMS
                    if fk != tf and recs[(nm, s)][fk]]
            if cand:
                v.append(min(cand, key=abs) if k == "x_sat"
                         else min(cand))
        return np.array(v, float)

    print(f"PARAMETRIC-BASELINE COMPLETE in {time.time()-t0:.0f}s")
    print(f"  {'family':14s}{'oracle sat':>11s}{'oracle xbias':>13s}"
          f"{'best-wrong sat':>15s}")
    for nm in FAMILIES:
        ob = np.nanmedian(orc(nm, "x_sat")) - TRUE_XSAT
        print(f"  {nm:14s}{np.nanmean(orc(nm,'sat_err')):>11.3f}{ob:>+13.2f}"
              f"{np.nanmean(wrong(nm,'sat_err')):>15.3f}")
    allorc = np.concatenate([orc(nm, "sat_err") for nm in FAMILIES])
    allwr = np.concatenate([wrong(nm, "sat_err") for nm in FAMILIES])
    obias = np.concatenate([orc(nm, "x_sat") for nm in FAMILIES]) - TRUE_XSAT
    print(f"  POOLED  oracle sat {np.nanmean(allorc):.3f} (x_sat bias "
          f"{np.nanmedian(obias):+.2f}s)  best-wrong sat {np.nanmean(allwr):.3f}")
    print(f"  (nonparametric I-spline reference: 0.039 curve, +0.3 s read-out)")



RUNNERS = {"penalty": run_penalty, "studentt": run_studentt,
           "parametric": run_parametric, "unpen": run_unpen}


def main(argv):
    if not argv or argv[0] not in RUNNERS:
        print(__doc__)
        return 2
    RUNNERS[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
