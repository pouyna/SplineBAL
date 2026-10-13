"""
recovery.py -- the core accuracy story: curve recovery and how it degrades with
detector noise, plus the penalty-vs-monotonicity ablation. Five subcommands,
each writing its own cache:

  subcommand  what it produces                              cache / consumer
  ----------  ------------------------------------------    ------------------------------
  main        the main production active-learning campaign  results.json
              (resumable, checkpointed) -- every family,    (the recovery and read-out sections;
              method, seed at sigma_y=0.1                    the base cache the rest reuse)
  noise       saturation error vs sigma_y in {.01,.05,.1,   results_noise.json
              .15}: penalized I-spline vs the fair GP-log   (recompute_stats noise)
              arm, and GP failure rates
  ablation    unpenalized monotone I-spline (P=11) own      results_unpen11.json
              loop across sigma_y:                          (recompute_stats gp, noise)
                penalized vs unpenalized = the PENALTY,
                unpenalized vs GP-log-pin = MONOTONICITY
  unpen25     the unpenalized monotone basis at production  results_unpen25.json
              resolution (P=25), isolating the penalty at   (recompute_stats gp, noise)
              fixed resolution
  gppin       the origin-matched GP-log-pin arm across the  results_gppin_noise.json
              noise sweep (no MCMC)                         (recompute_stats noise)

The 'main' campaign reuses make_figures' job list / worker / store logic verbatim
(loaded lazily so the noise/ablation workers stay light) and checkpoints after
every chunk, so a crash loses at most one chunk rather than the whole run.

Usage:  python3 recovery.py <main|noise|ablation|unpen25|gppin> [--jobs=N] [--seeds=N]
        noise --isp-sigmas=0.05,0.15   regenerate only those I-spline arms (with their
                                       designs stored) and splice the rest from the cache
        python3 recovery.py main --fresh --jobs=3   # ignore any stale results.json
'noise' splices the cached base-sigma I-spline arm from results.json and
'ablation' compares against it in its summary, so run 'main' first.
"""
import os
import sys
import json
import time

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

# make_figures pulls in matplotlib; load it lazily (only the 'main' subcommand
# needs it) so the noise/ablation workers do not import pyplot on spawn.
MF = None


def _load_mf():
    global MF
    if MF is None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import make_figures as _MF
        MF = _MF
    return MF


# ===========================================================================
# main -- resumable, checkpointing rebuild of results.json (the base campaign)
# ===========================================================================

def _tag():
    return {"seeds": MF.MC_SEEDS, "n_init": MF.N_INIT, "n_iters": MF.N_ITERS,
            "methods": MF.METHODS, "metric": "saturation_error",
            "design": "per-isotherm-mc + noise-free-regimes",
            "levels": list(MF.LEVELS5)}


def _fresh_res(tag):
    return {"tag": tag,
            "mc": {nm: {m: [None] * len(MF.MC_SEEDS) for m in MF.METHODS}
                   for nm in ISO.STANDARD_SPECS},
            "regimes": {f: {} for f in ISO._FAMILIES}}


def _done(res, job):
    kind, keys = job[0], job[1]
    if kind == "mc":
        name, method, seed = keys
        return res["mc"][name][method][MF.MC_SEEDS.index(seed)] is not None
    fam, lab = keys
    return res["regimes"][fam].get(lab) is not None


def _order_regimes(res):
    res["regimes"] = {fam: {lab: res["regimes"][fam][lab]
                            for lab in ISO.family_variant_specs(fam)
                            if lab in res["regimes"][fam]}
                      for fam in ISO._FAMILIES}


def run_main(argv):
    _load_mf()
    cache = MF.CACHE                     # already an absolute path under ../data
    fresh = "--fresh" in argv
    jobs, chunk = 3, 48
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--chunk="):
            chunk = int(a.split("=", 1)[1])

    tag = _tag()
    if os.path.exists(cache) and not fresh:
        with open(cache) as fh:
            res = json.load(fh)
        if res.get("tag") != tag:
            print("cache tag mismatch -> starting fresh", flush=True)
            res = _fresh_res(tag)
    else:
        res = _fresh_res(tag)

    job_list = MF._job_list()
    pending = [j for j in job_list if not _done(res, j)]
    print(f"MAIN CAMPAIGN (production penalized I-spline): {len(job_list)} jobs, {len(pending)} pending  "
          f"[sigma_x={ISO.SIGMA_X}, sigma_y={ISO.SIGMA_Y}, jobs={jobs}, chunk={chunk}]",
          flush=True)
    t0 = time.time()
    for start in range(0, len(pending), chunk):
        sub = pending[start:start + chunk]
        recs = poolutil.run_pool(MF._pool_run, list(enumerate(sub)), jobs,
                                 label=f"runs {start}-{start + len(sub)}")
        for idx, rec in recs.items():
            MF._store(res, sub[idx], rec)
        _order_regimes(res)
        with open(cache, "w") as fh:
            json.dump(res, fh)
        ndone = sum(1 for j in job_list if _done(res, j))
        print(f"  CHECKPOINT -> results.json  ({ndone}/{len(job_list)} done, "
              f"{time.time() - t0:.0f}s)", flush=True)
    print(f"MAIN CAMPAIGN COMPLETE in {time.time() - t0:.0f}s.", flush=True)


# ===========================================================================
# noise -- saturation error vs sigma_y: penalized I-spline vs the fair GP-log arm
# ===========================================================================
NOISE_SIGMAS = [0.01, 0.05, 0.1, 0.15]
BASE_SIGMA = 0.1


def _job_noise(job):
    kind, nm, sy, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_y=sy)
    yt = f(TG)

    if kind == "ispline":
        reg, _ = M.run_active_learning(observe, domain=DOMAIN, seed=seed,
                                       truth=f, acquisition="uncertainty")
        mu, _ = reg.predict(TG)
        # the design is kept so the GP arms can be fitted to identical data
        # (the matched-design control, gp_arms.py matched)
        return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                         dips=0.0, max_dip=0.0, multivalued=0.0,
                         t_obs=[float(v) for v in reg.t_obs],
                         y_obs=[float(v) for v in reg.y_obs])

    # fair unconstrained baseline: GP on the log-warped axis
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
    nv = RO.negative_variation(mu, TG, sigma_y=sy)
    uc = RO.upcrossings(mu, TG, 0.95)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                     dips=float(nv["n_descending_intervals"] > 0),
                     max_dip=nv["max_dip_depth"],
                     multivalued=float(uc["count"] > 1))


def run_noise(argv):
    jobs, nseeds, only = None, 50, None
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a.startswith("--isp-sigmas="):
            # regenerate only the I-spline arms at these sigma levels and splice every
            # other entry (GP-log arms, the other I-spline levels) from the existing
            # cache: used to add the stored designs at the reported noise levels
            # without re-running the whole sweep
            only = [float(v) for v in a.split("=", 1)[1].split(",")]
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_noise.json")

    if only is None:
        joblist = [("gp_log", nm, sy, s)
                   for sy in NOISE_SIGMAS for nm in FAMILIES for s in seeds]
        joblist += [("ispline", nm, sy, s)
                    for sy in NOISE_SIGMAS if sy != BASE_SIGMA
                    for nm in FAMILIES for s in seeds]
    else:
        joblist = [("ispline", nm, sy, s)
                   for sy in NOISE_SIGMAS if sy in only and sy != BASE_SIGMA
                   for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))
    poolutil.pin_threads()

    print(f"Running {len(joblist)} noise-sweep runs on {jobs} workers ...",
          flush=True)
    # recycling pool: the ispline arm runs the heavier MCMC loop, and a
    # non-recycled pool over hundreds of such fits balloons the Windows page file.
    recs = poolutil.run_pool(_job_noise, joblist, jobs, tasks_per_child=8,
                             label="noise-sweep runs")

    if only is not None:
        # everything not regenerated comes from the existing cache
        with open(cache) as fh:
            OLD = json.load(fh)
        for k in ("ispline", "gp_log"):
            for nm in FAMILIES:
                for sy in NOISE_SIGMAS:
                    if k == "ispline" and (sy in only or sy == BASE_SIGMA):
                        continue
                    for s in seeds:
                        recs[(k, nm, sy, s)] = OLD[f"{k}|{nm}|{sy}"][s]

    # splice in the cached base-sigma I-spline arm (the main campaign)
    with open(os.path.join(DATA, "results.json")) as fh:
        MAIN = json.load(fh)
    for nm in FAMILIES:
        for s in seeds:
            m_ = MAIN["mc"][nm]["uncertainty"][s]
            recs[("ispline", nm, BASE_SIGMA, s)] = dict(
                sat_err=float(m_["sat_err"]),
                dips=0.0, max_dip=0.0, multivalued=0.0,
                t_obs=list(m_["t_obs"]), y_obs=list(m_["y_obs"]))

    out = {f"{k}|{nm}|{sy}": [recs[(k, nm, sy, s)] for s in seeds]
           for k in ("ispline", "gp_log") for nm in FAMILIES for sy in NOISE_SIGMAS}
    out["_meta"] = dict(seeds=seeds, sigmas=NOISE_SIGMAS, families=FAMILIES)
    with open(cache, "w") as fh:
        json.dump(out, fh)

    def C(k, nm, sy, key="sat_err"):
        return np.array([r[key] for r in out[f"{k}|{nm}|{sy}"]], float)

    print("\n=== saturation error at N=20 vs detector noise ===")
    print(f"  {'family':18s}" + "".join(
        f"{'sy='+str(sy):>26s}" for sy in NOISE_SIGMAS))
    print(f"  {'':18s}" + "".join(f"{'I-spl':>12s}{'GP-log':>9s}{'ratio':>5s}"
                                  for _ in NOISE_SIGMAS))
    for nm in FAMILIES:
        row = ""
        for sy in NOISE_SIGMAS:
            a, b = C("ispline", nm, sy).mean(), C("gp_log", nm, sy).mean()
            row += f"{a:>12.3f}{b:>9.3f}{b/a:>5.1f}"
        print(f"  {nm:18s}{row}")
    row = ""
    for sy in NOISE_SIGMAS:
        a = np.mean([C("ispline", nm, sy).mean() for nm in FAMILIES])
        b = np.mean([C("gp_log", nm, sy).mean() for nm in FAMILIES])
        row += f"{a:>12.3f}{b:>9.3f}{b/a:>5.1f}"
    print(f"  {'POOLED':18s}{row}")

    print("\n=== unconstrained failure rates vs noise (GP-log, pooled) ===")
    print(f"  {'sigma_y':>10s}{'non-monotone %':>16s}{'max dip':>10s}"
          f"{'dip>sy %':>10s}{'multi-valued %':>16s}")
    for sy in NOISE_SIGMAS:
        d = np.concatenate([C("gp_log", nm, sy, "dips") for nm in FAMILIES])
        md = np.concatenate([C("gp_log", nm, sy, "max_dip") for nm in FAMILIES])
        mv = np.concatenate([C("gp_log", nm, sy, "multivalued") for nm in FAMILIES])
        print(f"  {sy:>10g}{100*d.mean():>16.1f}{md.mean():>10.4f}"
              f"{100*np.mean(md>sy):>10.1f}{100*mv.mean():>16.1f}")
    print("\nwrote", cache)


# ===========================================================================
# ablation -- the unpenalized monotone I-spline (P=11) in its own loop across
# noise: the middle point of penalized vs unpenalized (the penalty) vs
# GP-log-pin (monotonicity)
# ===========================================================================
ABL_SIGMAS = [0.05, 0.1, 0.15]
ABL_SEEDS = list(range(50))
SX = ISO.SIGMA_X                # follow the one definition of the input jitter


def _job_ablation(job):
    fam, sy, seed = job
    fname, kw = ISO.STANDARD_SPECS[fam]
    f = ISO.make_isotherm(fname, **kw)
    yt = f(TG)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_x=SX, sigma_y=sy)
    rng = np.random.default_rng(seed)
    reg = M.UnpenalizedRegressor(domain=DOMAIN)
    for x in M.survey_design(5, DOMAIN, rng):
        reg.add(x, observe(x, rng))
    reg.fit(seed=seed)
    bad = draws = 0
    b, d = reg.monotone_report(); bad += b; draws += d
    for _ in range(15):
        xn = reg.next_x()
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)
        b, d = reg.monotone_report(); bad += b; draws += d
    mu, _ = reg.predict(TG)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                     mono_bad=int(bad), mono_draws=int(draws))


def run_ablation(argv):
    jobs, seeds, smoke = 3, ABL_SEEDS, "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    if smoke:
        t0 = time.time()
        _, out = _job_ablation(("Langmuir", 0.1, 0))
        print(f"  smoke Langmuir sy=0.1 seed0: sat_err={out['sat_err']:.4f} "
              f"mono {out['mono_bad']}/{out['mono_draws']} bad  "
              f"[{time.time()-t0:.1f}s for one 16-fit run_ablation]")
        print("SMOKE OK")
        return
    cache = os.path.join(DATA, "results_unpen11.json")
    joblist = [(fam, sy, s) for sy in ABL_SIGMAS for fam in FAMILIES for s in seeds]
    print(f"Penalty ablation (unpen. P=11): {len(joblist)} runs (x16 fits) "
          f"[sx={SX}, sigmas={ABL_SIGMAS}, jobs={jobs}]", flush=True)
    recs = poolutil.run_pool(_job_ablation, joblist, jobs, tasks_per_child=8,
                             label="penalty-ablation runs")
    out = {f"{fam}|{sy}": [recs[(fam, sy, s)] for s in seeds]
           for sy in ABL_SIGMAS for fam in FAMILIES}
    out["_meta"] = dict(seeds=seeds, sigmas=ABL_SIGMAS, families=FAMILIES, sigma_x=SX,
                        model="ispline unpenalized P=11 HalfNormal (monotone, no penalty)")
    with open(cache, "w") as fh:
        json.dump(out, fh)

    # console summary: penalized (results.json) vs unpenalized (here) vs
    # GP-log-pin (results_gparms), all at sigma_y=0.1
    print(f"\n{'='*70}\nPENALTY-vs-MONOTONICITY ablation @ sigma_y=0.1 (mean sat_err)\n{'='*70}")
    try:
        PEN = json.load(open(os.path.join(DATA, "results.json")))
        GP = json.load(open(os.path.join(DATA, "results_gparms.json")))
        print(f"  {'family':14s}{'penalized':>10s}{'unpen11':>11s}{'GP-log-pin':>12s}")
        for fam in FAMILIES:
            pe = np.mean([r['sat_err'] for r in PEN['mc'][fam]['uncertainty']])
            un = np.mean([r['sat_err'] for r in out[f"{fam}|0.1"]])
            gp = np.mean([r['sat_err'] for r in GP['gp_log_pin'][fam]])
            print(f"  {fam:14s}{pe:>10.4f}{un:>11.4f}{gp:>12.4f}")
    except Exception as e:
        print("  (summary skipped:", e, ")")
    print("\n  written to results_unpen11.json")



# ===========================================================================
# unpen25 -- the unpenalized monotone basis at production resolution (P=25);
# writes results_unpen25.json.
# ===========================================================================
UNPEN25_SIGMAS = [0.05, 0.1, 0.15]
UNPEN25_SEEDS = list(range(50))
SX = ISO.SIGMA_X                # follow the one definition of the input jitter
UNPEN25_NINT = 21                 # P = n_interior + degree + 1 = 21 + 3 + 1 = 25


def _unpen25_job(job):
    fam, sy, seed = job
    fname, kw = ISO.STANDARD_SPECS[fam]
    f = ISO.make_isotherm(fname, **kw)
    yt = f(TG)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_x=SX, sigma_y=sy)
    rng = np.random.default_rng(seed)
    reg = M.UnpenalizedRegressor(domain=DOMAIN, n_interior=UNPEN25_NINT)
    for x in M.survey_design(5, DOMAIN, rng):
        reg.add(x, observe(x, rng))
    reg.fit(seed=seed)
    bad = draws = 0
    b, d = reg.monotone_report(); bad += b; draws += d
    for _ in range(15):
        xn = reg.next_x()
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)
        b, d = reg.monotone_report(); bad += b; draws += d
    mu, _ = reg.predict(TG)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                     mono_bad=int(bad), mono_draws=int(draws))


def run_unpen25(argv):
    jobs, seeds, smoke = 3, UNPEN25_SEEDS, "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    if smoke:
        t0 = time.time()
        _, out = _unpen25_job(("Langmuir", 0.1, 0))
        print(f"  smoke Langmuir sy=0.1 seed0: sat_err={out['sat_err']:.4f} "
              f"mono {out['mono_bad']}/{out['mono_draws']} bad  "
              f"[{time.time()-t0:.1f}s for one 16-fit run_unpen25]")
        print("SMOKE OK")
        return

    joblist = [(fam, sy, s) for sy in UNPEN25_SIGMAS for fam in FAMILIES for s in seeds]
    print(f"unpenalized P=25 arm: {len(joblist)} runs (x16 fits) "
          f"[sx={SX}, sigmas={UNPEN25_SIGMAS}, jobs={jobs}]", flush=True)
    t0 = time.time()
    recs = poolutil.run_pool(_unpen25_job, joblist, jobs, tasks_per_child=8,
                             label="unpen25 runs")
    out = {f"{fam}|{sy}": [recs[(fam, sy, s)] for s in seeds]
           for sy in UNPEN25_SIGMAS for fam in FAMILIES}
    out["_meta"] = dict(seeds=seeds, sigmas=UNPEN25_SIGMAS, families=FAMILIES, sigma_x=SX,
                        n_interior=UNPEN25_NINT,
                        model="ispline unpenalized P=25 HalfNormal (monotone, no penalty)")
    with open(os.path.join(DATA, "results_unpen25.json"), "w") as fh:
        json.dump(out, fh)

    print(f"\nUNPEN25 COMPLETE in {time.time()-t0:.0f}s")
    print(f"  {'family':18s}" + "".join(f"{'sy='+str(sy):>12s}" for sy in UNPEN25_SIGMAS))
    for fam in FAMILIES:
        row = "".join(
            f"{np.mean([r['sat_err'] for r in out[f'{fam}|{sy}']]):>12.3f}"
            for sy in UNPEN25_SIGMAS)
        print(f"  {fam:18s}{row}")
    row = "".join(
        f"{np.mean([np.mean([r['sat_err'] for r in out[f'{fam}|{sy}']]) for fam in FAMILIES]):>12.3f}"
        for sy in UNPEN25_SIGMAS)
    print(f"  {'POOLED':18s}{row}")




# ===========================================================================
# gppin -- the origin-matched GP-log-pin arm across the noise sweep. Was
# gppin_noise.py; writes results_gppin_noise.json. NOTE: three sigma levels,
# not the four of run_noise -- the cache has no sigma_y=0.01 cell.
# ===========================================================================
GPPIN_SIGMAS = [0.05, 0.1, 0.15]
GPPIN_SEEDS = list(range(50))


def _gppin_job(job):
    nm, sy, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_y=sy)
    yt = f(TG)
    egrid = GA.ell_grid("log")
    rng = np.random.default_rng(seed)
    xs = list(M.survey_design(5, DOMAIN, rng))
    ys = [observe(x, rng) for x in xs]
    for _ in range(15):
        _, sd_a, _, _ = GA._gp_fit(xs, ys, GA.ACQ, "log", True, egrid)   # pin=True
        xn = float(GA.ACQ[int(np.argmax(sd_a))])
        xs.append(xn)
        ys.append(observe(xn, rng))
    mu, _, _, _ = GA._gp_fit(xs, ys, TG, "log", True, egrid)
    nv = RO.negative_variation(mu, TG, sigma_y=sy)
    uc = RO.upcrossings(mu, TG, 0.95)
    return job, dict(sat_err=float(M.saturation_error(TG, mu, yt)),
                     dips=float(nv["n_descending_intervals"] > 0),
                     max_dip=nv["max_dip_depth"],
                     multivalued=float(uc["count"] > 1))


def run_gppin(argv):
    jobs, seeds, smoke = 3, GPPIN_SEEDS, "--smoke" in argv
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            seeds = list(range(int(a.split("=", 1)[1])))
    if smoke:
        t0 = time.time()
        _, out = _gppin_job(("Langmuir", 0.1, 0))
        print(f"  smoke Langmuir sy=0.1 seed0: sat_err={out['sat_err']:.4f} "
              f"[{time.time()-t0:.1f}s]")
        print("SMOKE OK")
        return

    joblist = [(nm, sy, s) for sy in GPPIN_SIGMAS for nm in FAMILIES for s in seeds]
    print(f"GP-log-pin noise sweep: {len(joblist)} runs [sigmas={GPPIN_SIGMAS}, "
          f"jobs={jobs}]", flush=True)
    t0 = time.time()
    recs = poolutil.run_pool(_gppin_job, joblist, jobs, tasks_per_child=25,
                             label="gp-log-pin noise")
    out = {f"gp_log_pin|{nm}|{sy}": [recs[(nm, sy, s)] for s in seeds]
           for nm in FAMILIES for sy in GPPIN_SIGMAS}
    out["_meta"] = dict(seeds=seeds, sigmas=GPPIN_SIGMAS, families=FAMILIES)
    with open(os.path.join(DATA, "results_gppin_noise.json"), "w") as fh:
        json.dump(out, fh)

    def C(nm, sy):
        return np.array([r["sat_err"] for r in out[f"gp_log_pin|{nm}|{sy}"]])

    print(f"\nGP-LOG-PIN NOISE SWEEP COMPLETE in {time.time()-t0:.0f}s")
    print(f"  {'family':18s}" + "".join(f"{'sy='+str(sy):>12s}" for sy in GPPIN_SIGMAS))
    for nm in FAMILIES:
        print(f"  {nm:18s}" + "".join(f"{C(nm,sy).mean():>12.3f}" for sy in GPPIN_SIGMAS))
    print(f"  {'POOLED':18s}" + "".join(
        f"{np.mean([C(nm,sy).mean() for nm in FAMILIES]):>12.3f}" for sy in GPPIN_SIGMAS))




RUNNERS = {"main": run_main, "noise": run_noise, "ablation": run_ablation,
           "unpen25": run_unpen25, "gppin": run_gppin}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in RUNNERS:
        print(f"recovery.py: unknown subcommand {cmd!r}; "
              f"choose from {list(RUNNERS)}", file=sys.stderr)
        sys.exit(2)
    RUNNERS[cmd](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
