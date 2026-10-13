"""
robustness.py -- how the read-out behaves when the benchmark's clean assumptions
are violated, and why the constraint loses at high noise. Three subcommands,
each writing its own cache:

  subcommand  what it stress-tests                          cache / consumer
  ----------  ------------------------------------------    ------------------------------
  failure     misspecification: truncated window, plateau   results_failure.json
              outlier, paid replication, instrument drift,  (recompute_stats outside)
              exact duplicate -- each with a diagnostic
              computable on real data
  anchor      WHY the UNPENALIZED constrained surrogate     results_anchor_unpen11.json
              loses at sigma_y=0.15: plateau anchor (A) vs   (--model=unpen11, the
              noise prior (C) vs boundary bias (B).          mechanism-diagnostic arm) and
              --model=unpen11 runs the unpenalized P=11      results_anchor.json
              arm the mechanism diagnostic is about;         (--model=production, the
              --model=production (default) runs the          production surrogate,
              production surrogate.                          for contrast)
  window      does the recommended one-sided read-out       results_window.json
              track the curve or the window edge x_max?     (recompute_stats window)

Usage:  python3 robustness.py <failure|anchor|window> [--jobs=N] [--seeds=N]
        python3 robustness.py anchor --model=unpen11     (the mechanism-diagnostic arm)
"""
import os
import sys
import json

import numpy as np

import ispline_model as M
import isotherms as ISO
import readout as RO
import poolutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # result caches live in code/data
DOMAIN = ISO.DOMAIN
TG = M.eval_points(0.0, DOMAIN[1])
FAMILIES = list(ISO.STANDARD_SPECS)
ALPHA = 0.95


# ===========================================================================
# failure -- behaviour under misspecification (negative controls)
# ===========================================================================
#   truncated    x_max=0.5 s, window does NOT bracket saturation (the dangerous
#                one: plateau is DEFINED as f(x_max)); diagnostic window_sufficiency
#   outlier      the single x_max reading contaminated by +3 sigma_y; diagnostic
#                plateau_leverage d x_sat/d f(x_max)
#   outlier_rep  as outlier but three x_max reads, PAID from the N=20 budget
#   drift        linear instrument drift over the campaign (total 2 sigma_y);
#                diagnostic: regress near-duplicate response gaps on index gap
#   duplicate    an exact repeat measurement (expected non-event; measured anyway)
ARMS = ["baseline", "truncated", "outlier", "outlier_rep", "drift", "duplicate"]
TRUNC_XMAX = 0.5
DRIFT_TOTAL = 2.0 * ISO.SIGMA_Y


def _run(arm, f, seed):
    """One active-learning run under perturbation `arm`.  Returns the fitted
    regressor, its reporting grid, and any bookkeeping the diagnostics need."""
    domain = (DOMAIN[0], TRUNC_XMAX) if arm == "truncated" else DOMAIN
    tg = M.eval_points(0.0, domain[1])
    rng = np.random.default_rng(seed)
    base_observe = ISO.make_observe(f, domain=domain)
    idx = {"n": 0}

    def observe(x, r):
        y = base_observe(x, r)
        if arm == "drift":
            y += DRIFT_TOTAL * idx["n"] / 19.0
        idx["n"] += 1
        return y

    n_init = 5
    reg = M.ISplineRegressor(domain=domain, n_init=n_init, n_iters=15)
    survey = list(M.survey_design(n_init, domain, rng))
    if arm in ("outlier_rep",):
        # spend two of the twenty runs replicating the plateau anchor
        survey = survey + [domain[1], domain[1]]
        reg.n_init = n_init + 2
        reg.n_iters = 13
    for k, x in enumerate(survey):
        y = observe(x, rng)
        if arm in ("outlier", "outlier_rep") and abs(x - domain[1]) < 1e-12 \
                and k == (len(survey) - 1 if arm == "outlier" else n_init - 1):
            y += 3.0 * ISO.SIGMA_Y          # contaminate ONE x_max reading
        reg.add(x, y)
    if arm == "duplicate":
        # The duplicate is one of the twenty measurements, not an extra one.
        # It is already in t_obs, and n_act below subtracts len(t_obs), so
        # decrementing n_iters here as well would double-count it and end the
        # run at N=19. (Compare outlier_rep, which moves n_init and n_iters
        # together so the budget stays 20.)
        xd = float(survey[len(survey) // 2])
        reg.add(xd, observe(xd, rng))
    reg.fit(seed=seed)
    n_act = (reg.n_init + reg.n_iters) - len(reg.t_obs)
    for _ in range(n_act):
        xn = reg.next_x()
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)
    return reg, tg, domain


def _drift_estimate(t_obs, y_obs, tol):
    """Free drift detector: regress the response difference of every
    near-duplicate pair on their separation in acquisition index."""
    t = np.asarray(t_obs, float)
    y = np.asarray(y_obs, float)
    dz, dn = [], []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if abs(t[i] - t[j]) <= tol:
                dn.append(j - i)
                dz.append(y[j] - y[i])
    if len(dn) < 3:
        return np.nan, len(dn)
    dn = np.asarray(dn, float)
    dz = np.asarray(dz, float)
    return float((dn @ dz) / (dn @ dn)), len(dn)


def _job_failure(job):
    arm, nm, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    reg, tg, domain = _run(arm, f, seed)

    draws = reg.sample(tg)
    mu = draws.mean(0)
    yt = f(tg)
    med, sd, lo, hi, _ = RO.level_crossing_median(draws, tg, ALPHA)

    # what fraction of the TRUE (full) plateau does the recommended input give?
    A_true = 1.0
    theta_true = (float(f(np.array([med]))[0]) / A_true
                  if np.isfinite(med) else np.nan)
    slope, npair = _drift_estimate(reg.t_obs, reg.y_obs, 0.02)
    return job, dict(
        sat_err=float(M.saturation_error(tg, mu, yt)),
        f_xmax=float(mu[-1]),
        x_sat=float(med) if np.isfinite(med) else float("nan"),
        ci_lo=float(lo), ci_hi=float(hi),
        theta_true=float(theta_true),
        window_suff=RO.window_sufficiency(draws, tg),
        r_last_octave=float(np.nanmean(
            (draws[:, -1] - draws[:, int(np.argmin(np.abs(tg - tg[-1] / 2)))])
            / draws[:, -1])),
        plateau_lev=RO.plateau_leverage(draws, tg, ALPHA),
        drift_slope=float(slope) if np.isfinite(slope) else float("nan"),
        n_pairs=int(npair),
        sigma_post=float(np.mean(reg._sigma)),
        n_obs=int(len(reg.t_obs)),
    )


def run_failure(argv):
    jobs, nseeds, arms = None, 25, list(ARMS)
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a.startswith("--arms="):
            arms = a.split("=", 1)[1].split(",")
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {ARMS}")
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_failure.json")
    joblist = [(arm, nm, s) for arm in arms for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    jobs = min(jobs, len(joblist))
    recs = poolutil.run_pool(_job_failure, joblist, jobs, tasks_per_child=8,
                             label="failure-mode runs")

    # --arms= regenerates a subset and MERGES into the existing cache, so a
    # single-arm fix does not cost a full re-run. Each job is seeded by
    # (arm, family, seed) alone, so an arm's values do not depend on which other
    # arms ran; the untouched arms are left exactly as they were. The merge is
    # recorded in _meta so the cache's provenance is never implicit.
    partial = sorted(arms) != sorted(ARMS)
    out = {}
    if partial and os.path.exists(cache):
        with open(cache) as fh:
            out = json.load(fh)
    out.update({f"{arm}|{nm}": [recs[(arm, nm, s)] for s in seeds]
                for arm in arms for nm in FAMILIES})
    meta = dict(seeds=seeds, arms=list(ARMS), families=FAMILIES,
                trunc_xmax=TRUNC_XMAX, drift_total=DRIFT_TOTAL,
                domain=list(DOMAIN))
    if partial:
        prev = (out.get("_meta") or {}).get("regenerated_arms") or []
        meta["regenerated_arms"] = sorted(set(prev) | set(arms))
    out["_meta"] = meta
    with open(cache, "w") as fh:
        json.dump(out, fh)


    def C(arm, key, nm=None):
        rs = (out[f"{arm}|{nm}"] if nm else
              [r for f in FAMILIES for r in out[f"{arm}|{f}"]])
        return np.array([r[key] for r in rs], float)

    print("\n=== behaviour outside the benchmark's assumptions "
          "(pooled over 5 families) ===")
    print(f"  {'arm':13s}{'sat err':>9s}{'f(xmax)':>9s}{'x_sat':>8s}"
          f"{'CI width':>10s}{'true theta':>12s}{'r(oct)':>8s}"
          f"{'P(r<tol)':>10s}{'lever':>8s}{'sigma':>8s}")
    for arm in ARMS:
        w = C(arm, "ci_hi") - C(arm, "ci_lo")
        print(f"  {arm:13s}{C(arm,'sat_err').mean():>9.3f}"
              f"{C(arm,'f_xmax').mean():>9.3f}{np.nanmean(C(arm,'x_sat')):>8.2f}"
              f"{np.nanmean(w):>10.2f}{np.nanmean(C(arm,'theta_true')):>12.3f}"
              f"{C(arm,'r_last_octave').mean():>8.3f}"
              f"{C(arm,'window_suff').mean():>10.2f}"
              f"{np.nanmean(C(arm,'plateau_lev')):>8.3f}"
              f"{C(arm,'sigma_post').mean():>8.3f}")

    print("\n  window-sufficiency separation (the statistic r over the final "
          "octave):")
    rb = C("baseline", "r_last_octave")
    rt = C("truncated", "r_last_octave")
    print(f"    saturating window : mean {rb.mean():.3f}, "
          f"range [{rb.min():.3f}, {rb.max():.3f}]")
    print(f"    truncated window  : mean {rt.mean():.3f}, "
          f"range [{rt.min():.3f}, {rt.max():.3f}]")
    for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"    tol={tol:.2f}: flags {100*np.mean(rb>=tol):5.1f}% of "
              f"saturating runs (false alarm), "
              f"{100*np.mean(rt>=tol):5.1f}% of truncated runs (detection)")

    print("\n  plateau outlier (+3 sigma_y on the x_max reading):")
    for nm in [None] + FAMILIES:
        lab = nm or "POOLED"
        b, o, r = (C("baseline", "x_sat", nm), C("outlier", "x_sat", nm),
                   C("outlier_rep", "x_sat", nm))
        fb, fo, fr = (C("baseline", "f_xmax", nm), C("outlier", "f_xmax", nm),
                      C("outlier_rep", "f_xmax", nm))
        print(f"    {lab:18s} f(xmax) {fb.mean():.3f} -> {fo.mean():.3f} "
              f"(replicated {fr.mean():.3f});  x_sat {np.nanmean(b):.2f} -> "
              f"{np.nanmean(o):.2f} (replicated {np.nanmean(r):.2f}) s")

    print("\n  drift detector (near-duplicate pairs, true rate "
          f"{DRIFT_TOTAL/19:.4f} per measurement):")
    for arm in ("baseline", "drift"):
        s = C(arm, "drift_slope")
        s = s[np.isfinite(s)]
        print(f"    {arm:10s} estimated rate {s.mean():+.4f} +/- "
              f"{s.std(ddof=1)/np.sqrt(max(len(s),1)):.4f} "
              f"(n={len(s)} runs with >=3 pairs, "
              f"mean pairs/run {C(arm,'n_pairs').mean():.1f})")
    print("\nwrote", cache)


# ===========================================================================
# anchor -- why the bare monotone constraint (the unpenalized P=11 arm) loses to
# the GP at high noise: at sigma_y=0.15 it scores 0.087 against GP-log's 0.074
# (recovery.py ablation and noise). This discriminates three mechanisms:
#   (A) plateau ANCHOR  -- rescale to the TRUE plateau; if the error drops to the
#       GP level, the single-point f(x_max) reference is the cause
#   (B) boundary BIAS   -- whatever (A) and (C) leave; inflated f(x_max)
#   (C) noise PRIOR     -- widen s_sigma to 0.6; if unchanged, the prior is not
#       binding (excludes a misspecified-prior artifact)
# ===========================================================================
N_TOTAL = 20
CONDITIONS = [
    (0.05, 0.2),      # reference: the main-campaign configuration
    (0.15, 0.2),      # the condition where the ordering reverses
    (0.15, 0.6),      # mechanism (C): is the noise prior binding?
]
LABELS = {c: f"sigma_y={c[0]}, s_sigma={c[1]}" for c in CONDITIONS}
def _gp_ref():
    """Pooled GP-log saturation error per sigma_y, read from the noise sweep
    cache (recovery.py 'noise') rather than hardcoded, so the mechanism
    diagnostic cannot quote a stale comparator."""
    path = os.path.join(DATA, "results_noise.json")
    if not os.path.exists(path):
        return {}
    NO = json.load(open(path))
    out = {}
    for sy, _ in CONDITIONS:
        keys = [f"gp_log|{nm}|{sy}" for nm in FAMILIES]
        if all(k in NO for k in keys):
            out[sy] = float(np.mean([np.mean([r["sat_err"] for r in NO[k]])
                                     for k in keys]))
    return out


def _job_anchor(job):
    (sy, s_sigma), nm, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_y=sy)
    rng = np.random.default_rng(seed)
    yt = f(TG)

    reg, _ = M.run_active_learning(observe, domain=DOMAIN, seed=seed,
                                   truth=f, acquisition="uncertainty",
                                   sigma_prior=s_sigma)
    draws = reg.sample(TG)
    mu = draws.mean(0)
    med, sd, lo, hi, _ = RO.level_crossing_median(draws, TG, ALPHA)

    # (A) remove the plateau misestimate exactly, leaving shape error only
    mu_corr = mu * (yt[-1] / mu[-1]) if mu[-1] > 0 else mu
    t = np.asarray(reg.t_obs, float)
    return job, dict(
        sat_err=float(M.saturation_error(TG, mu, yt)),
        sat_err_plateau_corrected=float(M.saturation_error(TG, mu_corr, yt)),
        f_xmax=float(mu[-1]),
        f_xmax_true=float(yt[-1]),
        x_sat=float(med) if np.isfinite(med) else float("nan"),
        sigma_post=float(np.mean(reg._sigma)),
        n_at_xmax=int(np.sum(np.abs(t - DOMAIN[1]) < 0.02)),
    )


def _job_anchor_unpen11(job):
    """The same diagnostic on the UNPENALIZED monotone basis (P=11).

    This is the arm whose high-noise reversal the noise sweep attributes to a
    smoothness-adaptation gap, so it is the arm the mechanism diagnostic has to
    run on.  The loop is written out here rather than routed through
    run_active_learning (which is hardwired to the production surrogate) so that
    no existing bitwise-reproducible campaign is touched.
    """
    (sy, s_sigma), nm, seed = job
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_y=sy)
    rng = np.random.default_rng(seed)
    yt = f(TG)

    reg = M.UnpenalizedRegressor(domain=DOMAIN, n_init=5, n_iters=15,
                                 sigma_prior=s_sigma)
    for x in M.survey_design(5, DOMAIN, rng):
        reg.add(x, observe(x, rng))
    reg.fit(seed=seed)
    for _ in range(15):
        xn = reg.next_x()
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)

    draws = reg.sample(TG)
    mu = draws.mean(0)
    med, sd, lo, hi, _ = RO.level_crossing_median(draws, TG, ALPHA)
    mu_corr = mu * (yt[-1] / mu[-1]) if mu[-1] > 0 else mu
    t = np.asarray(reg.t_obs, float)
    return job, dict(
        sat_err=float(M.saturation_error(TG, mu, yt)),
        sat_err_plateau_corrected=float(M.saturation_error(TG, mu_corr, yt)),
        f_xmax=float(mu[-1]),
        f_xmax_true=float(yt[-1]),
        x_sat=float(med) if np.isfinite(med) else float("nan"),
        sigma_post=float(np.mean(reg._sigma)),
        n_at_xmax=int(np.sum(np.abs(t - DOMAIN[1]) < 0.02)),
    )


ANCHOR_ARMS = {"production": (_job_anchor, "results_anchor.json",
                              "production penalized monotone I-spline (P=25, second-order penalty)"),
               "unpen11": (_job_anchor_unpen11, "results_anchor_unpen11.json",
                           "unpenalized monotone I-spline (P=11, Half-Normal weights)")}


def run_anchor(argv):
    jobs, nseeds, model = None, 20, "production"
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
        if a.startswith("--model="):
            model = a.split("=", 1)[1].strip().lower()
    if model not in ANCHOR_ARMS:
        print(f"robustness.py anchor: unknown --model={model!r}; "
              f"choose from {list(ANCHOR_ARMS)}", file=sys.stderr)
        sys.exit(2)
    job_fn, cache_name, model_desc = ANCHOR_ARMS[model]
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, cache_name)
    joblist = [(c, nm, s) for c in CONDITIONS for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) // 2)
    jobs = min(jobs, len(joblist))
    recs = poolutil.run_pool(job_fn, joblist, jobs, tasks_per_child=8,
                             label=f"plateau-anchor runs ({model})")

    out = {"|".join(map(str, c)) + f"|{nm}": [recs[(c, nm, s)] for s in seeds]
           for c in CONDITIONS for nm in FAMILIES}
    out["_meta"] = dict(seeds=seeds, conditions=[list(c) for c in CONDITIONS],
                        n_total=N_TOTAL, families=FAMILIES,
                        gp_ref={str(k): v for k, v in _gp_ref().items()},
                        model=model, model_desc=model_desc)
    with open(cache, "w") as fh:
        json.dump(out, fh)

    def C(c, nm, k):
        return np.array([r[k] for r in out["|".join(map(str, c)) + f"|{nm}"]],
                        float)

    def P(c, k):
        return np.concatenate([C(c, nm, k) for nm in FAMILIES])

    gp_ref = _gp_ref()
    print(f"\n=== why does the constrained surrogate lose at high noise? "
          f"[{model_desc}] ===\n")
    print(f"  {'condition':26s}{'sat err':>9s}{'plateau-corr':>14s}"
          f"{'f(xmax)':>9s}{'sigma_post':>12s}{'obs at xmax':>13s}"
          f"{'GP ref':>9s}")
    for c in CONDITIONS:
        print(f"  {LABELS[c]:26s}{P(c,'sat_err').mean():>9.3f}"
              f"{P(c,'sat_err_plateau_corrected').mean():>14.3f}"
              f"{P(c,'f_xmax').mean():>9.3f}{P(c,'sigma_post').mean():>12.3f}"
              f"{P(c,'n_at_xmax').mean():>13.1f}{gp_ref.get(c[0], float('nan')):>9.3f}")

    base = CONDITIONS[1]
    b = P(base, "sat_err").mean()
    gp = gp_ref.get(0.15, float('nan'))
    print(f"\n  the gap to close at sigma_y=0.15: {b:.3f} (ours) vs "
          f"{gp:.3f} (unconstrained) = {b - gp:+.3f}\n")

    corr = P(base, "sat_err_plateau_corrected").mean()
    print(f"  (A) plateau anchor : correcting the plateau exactly gives "
          f"{corr:.3f}, closing {100*(b-corr)/(b-gp):.0f}% of the gap")
    pri = P(CONDITIONS[2], "sat_err").mean()
    print(f"  (C) noise prior    : widening s_sigma to 0.6 gives {pri:.3f}, "
          f"closing {100*(b-pri)/(b-gp):.0f}% of the gap")
    print(f"                       inferred sigma is "
          f"{P(base,'sigma_post').mean():.3f} against a true {0.15:.2f} "
          f"({100*(P(base,'sigma_post').mean()/0.15-1):+.0f}%), so the prior "
          f"is not binding")
    print(f"  (B) boundary bias  : whatever the two above leave unexplained; "
          f"f(x_max) sits {100*(P(base,'f_xmax').mean()/P(base,'f_xmax_true').mean()-1):+.1f}% "
          f"high at sigma_y=0.15 against "
          f"{100*(P(CONDITIONS[0],'f_xmax').mean()/P(CONDITIONS[0],'f_xmax_true').mean()-1):+.1f}% "
          f"at 0.05")

    print(f"\n  does the loop replicate the plateau anchor unprompted?")
    for c in CONDITIONS:
        v = P(c, "n_at_xmax")
        print(f"    {LABELS[c]:26s} {v.mean():.1f} of 20 measurements at "
              f"x_max (range {int(v.min())}-{int(v.max())})")

    print(f"\n  per family at sigma_y=0.15 (s_sigma=0.2):")
    print(f"    {'family':18s}{'sat err':>10s}{'plateau-corr':>14s}"
          f"{'f(xmax)':>10s}")
    for nm in FAMILIES:
        print(f"    {nm:18s}{C(base,nm,'sat_err').mean():>10.3f}"
              f"{C(base,nm,'sat_err_plateau_corrected').mean():>14.3f}"
              f"{C(base,nm,'f_xmax').mean():>10.3f}")
    print("\nwrote", cache)


# ===========================================================================
# window -- does the recommended one-sided read-out track the curve or the
# window edge x_max? (basis rebuilt on each window, so a shorter window is a
# genuinely coarser-at-the-top basis, as in practice)
# ===========================================================================
XMAXES = [1.5, 2.0, 2.5, 3.5, 5.0, 8.0]     # true x_sat = 2.0 s
X_MIN = ISO.DOMAIN[0]   # production x_min; this swept 0.005 before, alone among
                        # the campaigns, which put the window sweep on a different footing


def _job_window(job):
    xmax, nm, seed = job
    domain = (X_MIN, xmax)
    fam, kw = ISO.STANDARD_SPECS[nm]
    f = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(f, domain=domain)
    tg = M.eval_points(0.0, xmax)
    reg, _ = M.run_active_learning(observe, domain=domain, seed=seed,
                                   truth=f, acquisition="uncertainty")
    draws = reg.sample(tg)
    theta = RO.attained_fraction(draws)
    cross = np.array([RO._interp_crossing(tg, theta[s], ALPHA)
                      for s in range(theta.shape[0])])
    cross = cross[np.isfinite(cross)]
    x_true = 2.0
    med = float(np.median(cross)) if cross.size else float("nan")
    q95 = float(np.quantile(cross, 0.95)) if cross.size else float("nan")
    lo = float(np.quantile(cross, 0.025)) if cross.size else float("nan")
    hi = float(np.quantile(cross, 0.975)) if cross.size else float("nan")
    return job, dict(
        x_sat=med, one_sided=q95,
        covered_1s=bool(np.isfinite(q95) and x_true <= q95),
        covered_2s=bool(np.isfinite(lo) and lo <= x_true <= hi),
        bias=float(med - x_true) if np.isfinite(med) else float("nan"),
        r_octave=RO.window_sufficiency(draws, tg, tol=0.15),
        r_raw=float(np.nanmean(
            (draws[:, -1] - draws[:, int(np.argmin(np.abs(tg - tg[-1] / 2)))])
            / draws[:, -1])),
        f_xmax=float(draws[:, -1].mean()),
        true_frac=float(f(np.array([xmax]))[0]))


def run_window(argv):
    jobs, nseeds = None, 12
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        if a.startswith("--seeds="):
            nseeds = int(a.split("=", 1)[1])
    seeds = list(range(nseeds))
    cache = os.path.join(DATA, "results_window.json")
    joblist = [(xm, nm, s) for xm in XMAXES for nm in FAMILIES for s in seeds]
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) // 3)
    recs = poolutil.run_pool(_job_window, joblist, jobs, tasks_per_child=8,
                             label="window-sweep runs")
    out = {f"{xm}|{nm}": [recs[(xm, nm, s)] for s in seeds]
           for xm in XMAXES for nm in FAMILIES}
    # record x_min: this sweep rebuilds the basis per window, so the lower
    # anchor is part of the result's provenance and must not be implicit
    out["_meta"] = dict(seeds=seeds, xmaxes=XMAXES, x_true=2.0, x_min=X_MIN)
    with open(cache, "w") as fh:
        json.dump(out, fh)

    def P(xm, k):
        return np.concatenate([[r[k] for r in out[f"{xm}|{nm}"]]
                               for nm in FAMILIES])

    print("\n=== does the recommended read-out track curve or "
          "window? (true x_sat = 2.0 s) ===")
    print(f"  {'x_max':>7s}{'true frac':>11s}{'median x_sat':>14s}"
          f"{'one-sided lim':>15s}{'1-sided cov':>13s}{'2-sided cov':>13s}")
    for xm in XMAXES:
        print(f"  {xm:>7.1f}{np.nanmean(P(xm,'true_frac')):>11.3f}"
              f"{np.nanmean(P(xm,'x_sat')):>14.2f}{np.nanmean(P(xm,'one_sided')):>15.2f}"
              f"{100*np.mean(P(xm,'covered_1s')):>12.0f}%"
              f"{100*np.mean(P(xm,'covered_2s')):>12.0f}%")

    print("\n=== window-sufficiency r vs window generosity ===")
    print(f"  {'x_max':>7s}{'true frac':>11s}{'r (mean)':>10s}{'r (max)':>9s}"
          f"{'P(r>=0.15)':>12s}  <- false alarm above 2.0, detection at/below")
    for xm in XMAXES:
        r = P(xm, "r_raw")
        print(f"  {xm:>7.1f}{np.nanmean(P(xm,'true_frac')):>11.3f}"
              f"{r.mean():>10.3f}{r.max():>9.3f}{100*np.mean(r>=0.15):>11.0f}%")
    print("\nwrote", cache)


RUNNERS = {"failure": run_failure, "anchor": run_anchor, "window": run_window}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in RUNNERS:
        print(f"robustness.py: unknown subcommand {cmd!r}; "
              f"choose from {list(RUNNERS)}", file=sys.stderr)
        sys.exit(2)
    RUNNERS[cmd](rest)


if __name__ == "__main__":
    main(sys.argv[1:])
