"""
recompute_stats.py
==================
Recompute the reported numbers from the cached result files, section by
section, so that any reported figure-of-merit, ratio, percentage or table entry
can be checked against its source in one run.

This is the verification backbone of the study and the entry point a reader
should use: no MCMC, no re-running of any campaign, a few seconds.

    python3 recompute_stats.py            # every section
    python3 recompute_stats.py gp         # one section

Sections, in run order:
    iso acq level gp readout xsat window noise anchor heldout heldunif outside
    survey knots cost prior sigma band sampler diag learning regimes acqgrid
    decoupled monoproj ablation gpmatched studentt parametric misc
Every section reads the results_*.json caches in data/ (written by the campaign
scripts; see the README), and a missing cache is reported as skipped. Only
numpy and scipy are needed, plus isotherms.py for the analytic numbers in 'misc'.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")   # result caches live in code/data
SIGMA_Y = 0.1
FAM = ["Langmuir", "Dissociative", "Sips (n=2)", "Elovich", "D-R KWW"]
SHOW = {"Langmuir": "Langmuir", "Dissociative": "Michaelis-Menten",
        "Sips (n=2)": "Sips (n=2)", "Elovich": "Elovich", "D-R KWW": "KWW"}


def load(name):
    try:
        with open(os.path.join(DATA, name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def head(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def skip(name):
    print(f"  [skipped: {name} not present -- run the corresponding script]")


# ---------------------------------------------------------------------------
def sec_iso():
    head("Recovery across kinetic regimes (N=20, uncertainty sampling)")
    R = load("results.json")
    if R is None:
        return skip("results.json")
    print(f"  {'curve':18s}{'sat error':>18s}{'f(xmax)':>18s}{'theta':>18s}")
    se = []
    for nm in FAM:
        rs = R["mc"][nm]["uncertainty"]
        a = np.array([r["sat_err"] for r in rs])
        b = np.array([r["f_max"] for r in rs])
        c = np.array([r["theta"] for r in rs])
        se.append(a.mean())
        print(f"  {SHOW[nm]:18s}{a.mean():>10.3f}+/-{a.std():<7.3f}"
              f"{b.mean():>10.3f}+/-{b.std():<7.3f}"
              f"{c.mean():>10.3f}+/-{c.std():<7.3f}")
    print(f"  {'mean over regimes':18s}{np.mean(se):>10.3f}")


def sec_acq():
    head("Acquisition comparison, randomized x_sat")
    for tag, fn in (("log-uniform x_sat", "results_randxsat.json"),
                    ("uniform x_sat", "results_randxsat_uniform.json")):
        R = load(fn)
        if R is None:
            skip(fn)
            continue
        print(f"\n  -- {tag} --")
        meth = ["uncertainty", "grid", "random_log", "random"]
        print(f"  {'curve':18s}" + "".join(f"{m:>18s}" for m in meth))
        for nm in FAM:
            row = ""
            for m in meth:
                v = np.array([r["history"][-1] for r in R["mc"][nm][m]])
                row += f"{v.mean():>10.3f}+/-{v.std():<7.3f}"
            print(f"  {SHOW[nm]:18s}{row}")
        print(f"  {'ratios vs active':18s}", end="")
        for m in meth[1:]:
            num = np.mean([np.mean([r["history"][-1] for r in R["mc"][nm][m]])
                           for nm in FAM])
            den = np.mean([np.mean([r["history"][-1]
                                    for r in R["mc"][nm]["uncertainty"]])
                           for nm in FAM])
            print(f"  {m}={num/den:.2f}x", end="")
        print()
        per = {}
        for nm in FAM:
            a = np.array([r["history"][-1] for r in R["mc"][nm]["uncertainty"]])
            for m in meth[1:]:
                b = np.array([r["history"][-1] for r in R["mc"][nm][m]])
                per.setdefault(m, []).append(b.mean() / a.mean())
        for m, v in per.items():
            print(f"    per-family ratio range vs {m:11s} "
                  f"{min(v):.2f}x to {max(v):.2f}x")
        _acq_paired_p(R["mc"], meth, "history")
        xs = [r["x_sat"] for nm in FAM for r in R["mc"][nm]["uncertainty"]
              if "x_sat" in r]
        if xs:
            print(f"    drawn x_sat: median {np.median(xs):.2f} s over {len(xs)} runs")

    # the fixed-location campaign (x_sat = 2.0 s for every curve), results.json
    R = load("results.json")
    if R is None:
        return skip("results.json")
    print("\n  -- fixed x_sat = 2.0 s (results.json) --")
    meth = ["uncertainty", "grid", "random_log", "random"]
    print(f"  {'curve':18s}" + "".join(f"{m:>13s}" for m in meth)
          + "".join(f"{m + '/unc':>17s}" for m in meth[1:]))
    for nm in FAM:
        v = {m: np.mean([r["sat_err"] for r in R["mc"][nm][m]]) for m in meth}
        print(f"  {SHOW[nm]:18s}" + "".join(f"{v[m]:>13.3f}" for m in meth)
              + "".join(f"{v[m] / v['uncertainty']:>16.2f}x" for m in meth[1:]))
    pool = {m: np.mean([np.mean([r["sat_err"] for r in R["mc"][nm][m]]) for nm in FAM])
            for m in meth}
    print(f"  {'pooled':18s}" + "".join(f"{pool[m]:>13.3f}" for m in meth)
          + "".join(f"{pool[m] / pool['uncertainty']:>16.2f}x" for m in meth[1:]))
    _acq_paired_p(R["mc"], meth, "sat_err")


def _acq_paired_p(mc, meth, key):
    """Paired Wilcoxon over the 250 (family, seed) runs: each baseline against
    uncertainty sampling, on the N=20 error."""
    from scipy import stats

    def col(m):
        return np.array([(r[key][-1] if key == "history" else r[key])
                         for nm in FAM for r in mc[nm][m]], float)
    a = col(meth[0])
    print("    paired p vs uncertainty: " + ", ".join(
        f"{m} p={stats.wilcoxon(col(m) - a).pvalue:.1g}" for m in meth[1:]))


def sec_level():
    head("Level-wise response and input errors")
    R = load("results.json")
    if R is None:
        return skip("results.json")
    lev = [5, 25, 50, 75, 95]
    for key, lab, unit in (("y_err", "response error dy(a)", ""),
                           ("x_err", "input error dx(a)", " s")):
        print(f"\n  -- {lab}{unit} --")
        print(f"  {'curve':18s}" + "".join(f"{str(l)+'%':>17s}" for l in lev))
        for nm in FAM:
            rs = R["mc"][nm]["uncertainty"]
            row = ""
            for l in lev:
                v = np.array([r[key][str(l)] for r in rs])
                row += f"{v.mean():>+9.3f}+/-{v.std():<7.3f}"
            print(f"  {SHOW[nm]:18s}{row}")


def sec_gp():
    head("Gaussian-process ablation")
    G = load("results_gparms.json")
    R = load("results.json")
    if G is None or R is None:
        return skip("results_gparms.json")
    arms = ["gp_lin", "gp_log", "gp_log_pin"]
    isp = {nm: np.array([r["sat_err"] for r in R["mc"][nm]["uncertainty"]])
           for nm in FAM}
    print(f"  {'curve':18s}" + "".join(f"{a:>13s}" for a in arms)
          + f"{'I-spline':>13s}")
    for nm in FAM:
        row = "".join(f"{np.mean([r['sat_err'] for r in G[a][nm]]):>13.3f}"
                      for a in arms)
        print(f"  {SHOW[nm]:18s}{row}{isp[nm].mean():>13.3f}")
    pooled_i = np.mean([isp[nm].mean() for nm in FAM])
    print(f"  {'pooled':18s}"
          + "".join(f"{np.mean([np.mean([r['sat_err'] for r in G[a][nm]]) for nm in FAM]):>13.3f}"
                    for a in arms) + f"{pooled_i:>13.3f}")
    try:
        from scipy import stats
        a_all = np.concatenate([isp[nm] for nm in FAM])
        print(f"  {'ratio / paired p':18s}", end="")
        for a in arms:
            b = np.concatenate([np.array([r["sat_err"] for r in G[a][nm]])
                                for nm in FAM])
            p = stats.wilcoxon(b - a_all).pvalue
            print(f"{b.mean()/a_all.mean():>7.2f}x p={p:.0e}", end="")
        print()
        # conservative blocked test over the 50 seed-level mean differences
        print(f"  {'blocked p (seed-level)':18s}", end="")
        for a in arms:
            diffs = np.mean([np.array([r["sat_err"] for r in G[a][nm]]) - isp[nm]
                             for nm in FAM], axis=0)
            print(f"{'':7s}p={stats.wilcoxon(diffs).pvalue:.0e}", end="")
        print()
        # bootstrap 95% CIs on the headline pooled-mean ratios quoted in
        # the headline comparison (paired resampling of the 250 matched runs, 10k draws;
        # seed fixed so the printed brackets are reproducible)
        U25 = load("results_unpen25.json")
        brng = np.random.default_rng(0)

        def boot_ci(b, a=a_all, n=10000):
            idx = brng.integers(0, len(a), (n, len(a)))
            r = b[idx].mean(1) / a[idx].mean(1)
            return np.percentile(r, [2.5, 97.5])

        ci_arms = [("GP-linear / penalized",
                    np.concatenate([[r["sat_err"] for r in G["gp_lin"][nm]]
                                    for nm in FAM]))]
        # the unpen. P=25 arm keeps its slot even when its cache is absent, so
        # the RNG stream, and hence every printed bracket, does not depend on
        # which caches happen to be present
        ci_arms.append(("unpen. P=25 / penalized",
                        np.concatenate([[e["sat_err"] for e in U25[f"{nm}|0.1"]]
                                        for nm in FAM]) if U25 else None))
        ci_arms.append(("GP-log-pin / penalized",
                        np.concatenate([[r["sat_err"] for r in G["gp_log_pin"][nm]]
                                        for nm in FAM])))
        print(f"  bootstrap 95% CIs on the pooled ratios:")
        for lab, b in ci_arms:
            if b is None:                       # consume the block, print nothing
                brng.integers(0, len(a_all), (10000, len(a_all)))
                continue
            lo, hi = boot_ci(np.asarray(b, float))
            print(f"    {lab:26s} {b.mean()/a_all.mean():.2f}x "
                  f"[{lo:.2f}, {hi:.2f}]")
    except ImportError:
        pass

    # the two unpenalized monotone bases at sigma_y = 0.1, each in its own loop
    # (recovery.py ablation and unpen25), against the same 250 penalized runs
    U = {"unpen. P=11": load("results_unpen11.json"),
         "unpen. P=25": load("results_unpen25.json")}
    if all(v is not None for v in U.values()):
        from scipy import stats
        a_all = np.concatenate([isp[nm] for nm in FAM])
        print(f"\n  {'curve':18s}" + "".join(f"{k:>13s}" for k in U) + f"{'penalized':>13s}")
        for nm in FAM:
            print(f"  {SHOW[nm]:18s}"
                  + "".join(f"{np.mean([e['sat_err'] for e in u[f'{nm}|0.1']]):>13.3f}"
                            for u in U.values()) + f"{isp[nm].mean():>13.3f}")
        print(f"  {'pooled':18s}"
              + "".join(f"{np.mean([np.mean([e['sat_err'] for e in u[f'{nm}|0.1']]) for nm in FAM]):>13.3f}"
                        for u in U.values()) + f"{pooled_i:>13.3f}")
        print(f"  {'ratio / paired p':18s}", end="")
        for u in U.values():
            b = np.concatenate([[e["sat_err"] for e in u[f"{nm}|0.1"]] for nm in FAM])
            print(f"{b.mean() / a_all.mean():>7.2f}x p={stats.wilcoxon(b - a_all).pvalue:.0e}", end="")
        print()
        pin = np.concatenate([[r["sat_err"] for r in G["gp_log_pin"][nm]] for nm in FAM])
        u25 = np.mean([np.mean([e["sat_err"] for e in U["unpen. P=25"][f"{nm}|0.1"]])
                       for nm in FAM])
        pin_pool = np.mean([np.mean([r["sat_err"] for r in G["gp_log_pin"][nm]]) for nm in FAM])
        print(f"  unpen. P=25 / GP-log-pin: {u25 / pin_pool:.2f}x;  penalized lower than "
              f"GP-log-pin on {100 * np.mean(a_all < pin):.0f}% of the {len(a_all)} matched runs")
    else:
        skip("results_unpen11.json / results_unpen25.json")

    def col(a, k):
        return np.array([r[k] for nm in FAM for r in G[a][nm]], float)

    print(f"\n  {'failure mode':36s}" + "".join(f"{a:>13s}" for a in arms))
    rows = [("non-monotone posterior mean %", lambda a: 100 * col(a, "dips").mean()),
            ("total negative variation", lambda a: col(a, "tot_negvar").mean()),
            ("deepest dip", lambda a: col(a, "max_dip").mean()),
            ("deepest dip > sigma_y %", lambda a: 100 * col(a, "dip_gt_noise").mean()),
            ("multi-valued 95% read-out %", lambda a: 100 * col(a, "multivalued").mean()),
            ("inferred f(0) (true 0)", lambda a: col(a, "f0").mean()),
            ("selections on a falling mean %",
             lambda a: 100 * col(a, "n_desc").sum() / col(a, "n_active").sum()),
            ("selections near a local minimum %",
             lambda a: 100 * col(a, "n_nearmin").sum() / col(a, "n_active").sum()),
            ("interior hyperparameter optimum %",
             lambda a: 100 * (1 - col(a, "on_edge").mean()))]
    for lab, fn in rows:
        print(f"  {lab:36s}" + "".join(f"{fn(a):>13.3f}" for a in arms))
    # does a misled selection cost accuracy?  OLS of each GP-log-pin run's
    # saturation error on its count of falling-mean selections
    xd, ye = col("gp_log_pin", "n_desc"), col("gp_log_pin", "sat_err")
    X = np.column_stack([np.ones_like(xd), xd])
    beta, res, *_ = np.linalg.lstsq(X, ye, rcond=None)
    s2 = float(np.sum((ye - X @ beta) ** 2)) / (len(ye) - 2)
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    print(f"  GP-log-pin sat err on falling-mean selections: slope {beta[1]:+.4f} "
          f"per selection (s.e. {se:.4f}), {len(ye)} runs")
    for a in arms:
        mv = col(a, "multivalued").astype(bool)
        sp = col(a, "span95")
        print(f"  upcrossing span, {a:12s} "
              f"{(sp[mv].mean() if mv.any() else float('nan')):.2f} s "
              f"over {int(mv.sum())} multi-valued fits")


def sec_readout():
    head("The recommended input read two ways, and coverage")
    R = load("results_readout.json")
    if R is None:
        return skip("results_readout.json")

    def C(nm, k):
        return np.array([r[k] for r in R[nm]], float)

    print(f"  {'curve':18s}{'1/f(x95)':>10s}{'1/mu(xh)':>10s}{'theta(2)':>10s}"
          f"{'resp gap':>10s}{'time gap':>10s}{'2-sided':>9s}{'1-sided':>9s}")
    for nm in FAM:
        dt = C(nm, "x_sat_med").mean() - C(nm, "x_true")[0]
        print(f"  {SHOW[nm]:18s}{C(nm,'amp_true')[0]:>10.1f}"
              f"{np.median(C(nm,'amp_post')):>10.1f}"
              f"{C(nm,'theta_at_true').mean():>10.3f}"
              f"{C(nm,'resp_gap').mean():>10.3f}"
              f"{100*dt/C(nm,'x_true')[0]:>9.0f}%"
              f"{100*C(nm,'covered').mean():>8.0f}%"
              f"{100*C(nm,'covered_onesided').mean():>8.0f}%")
    a = np.concatenate([C(nm, "covered") for nm in FAM])
    b = np.concatenate([C(nm, "covered_onesided") for nm in FAM])
    print(f"  {'pooled':18s}{'':>50s}{100*a.mean():>8.0f}%{100*b.mean():>8.0f}%")
    ps = np.concatenate([C(nm, "x_sat_sd") for nm in FAM])
    ss = [C(nm, "x_sat_med").std(ddof=1) for nm in FAM]
    print(f"\n  within-fit posterior sd of the crossing: "
          f"{min(C(nm,'x_sat_sd').mean() for nm in FAM):.2f} to "
          f"{max(C(nm,'x_sat_sd').mean() for nm in FAM):.2f} s")
    print(f"  seed-to-seed sd of the point estimate  : "
          f"{min(ss):.2f} to {max(ss):.2f} s   "
          f"(ratio {ps.mean()/np.mean(ss):.1f}x)")
    print(f"  risk-bounded read-out (R<=0.05)        : "
          f"{min(C(nm,'x_risk').mean() for nm in FAM):.2f} to "
          f"{max(C(nm,'x_risk').mean() for nm in FAM):.2f} s")
    # the response-unit example: the Sips read-out at q = 0.7, as the input, the
    # expected attained fraction there, and the under-saturation risk there
    nm = "Sips (n=2)"
    j = int(np.argmin(np.abs(np.asarray(R[nm][0]["q"], float) - 0.7)))
    print(f"\n  {SHOW[nm]}, q=0.7: x_q {C(nm, 'x_q')[:, j].mean():.2f} s, expected "
          f"attained fraction {C(nm, 'theta_q')[:, j].mean():.3f}, under-saturation "
          f"risk {C(nm, 'risk_q')[:, j].mean():.2f} (means over {len(R[nm])} seeds)")
    print(f"\n  window sufficiency on saturating runs: r(last octave) "
          f"{min(C(nm,'r_last_octave').mean() for nm in FAM):.3f} to "
          f"{max(C(nm,'r_last_octave').mean() for nm in FAM):.3f}")
    print(f"  plateau leverage d x_sat/d f(xmax): "
          f"{min(C(nm,'plateau_lev').mean() for nm in FAM):.3f} to "
          f"{max(C(nm,'plateau_lev').mean() for nm in FAM):.3f} s per %")


def sec_noise():
    head("Output-noise sweep")
    R = load("results_noise.json")
    if R is None:
        return skip("results_noise.json")
    sig = R["_meta"]["sigmas"]

    def C(k, nm, sy, key="sat_err"):
        return np.array([r[key] for r in R[f"{k}|{nm}|{sy}"]], float)

    print(f"  {'curve':18s}" + "".join(f"{'sy='+str(sy):>26s}" for sy in sig))
    print(f"  {'':18s}" + "".join(f"{'I-spl':>10s}{'GP-log':>10s}{'ratio':>6s}"
                                  for _ in sig))
    for nm in FAM:
        row = ""
        for sy in sig:
            a, b = C("ispline", nm, sy).mean(), C("gp_log", nm, sy).mean()
            row += f"{a:>10.3f}{b:>10.3f}{b/a:>6.2f}"
        print(f"  {SHOW[nm]:18s}{row}")
    row = ""
    for sy in sig:
        a = np.mean([C("ispline", nm, sy).mean() for nm in FAM])
        b = np.mean([C("gp_log", nm, sy).mean() for nm in FAM])
        row += f"{a:>10.3f}{b:>10.3f}{b/a:>6.2f}"
    print(f"  {'pooled':18s}{row}")
    try:
        from scipy import stats as _st
        for sy in sig:
            ii = np.concatenate([C("ispline", nm, sy) for nm in FAM])
            gg = np.concatenate([C("gp_log", nm, sy) for nm in FAM])
            print(f"  paired p sy={sy:g}: {_st.wilcoxon(gg-ii).pvalue:.0e}  "
                  f"(GP better on {100*np.mean(gg<ii):.0f}%)")
    except ImportError:
        pass
    print(f"\n  {'sigma_y':>10s}{'GP non-monotone %':>20s}{'GP deepest dip':>17s}"
          f"{'dip>sy %':>10s}{'multi-valued %':>16s}")
    for sy in sig:
        d = np.concatenate([C("gp_log", nm, sy, "dips") for nm in FAM])
        md = np.concatenate([C("gp_log", nm, sy, "max_dip") for nm in FAM])
        mv = np.concatenate([C("gp_log", nm, sy, "multivalued") for nm in FAM])
        print(f"  {sy:>10g}{100*d.mean():>20.1f}{md.mean():>17.4f}"
              f"{100*np.mean(md>sy):>10.1f}{100*mv.mean():>16.1f}")

    # every arm of the GP ablation except GP-linear, pooled over families, and
    # each arm's error as a ratio to the penalized surrogate's
    U25, U11 = load("results_unpen25.json"), load("results_unpen11.json")
    GPP = load("results_gppin_noise.json")
    if U25 is None or U11 is None or GPP is None:
        return skip("results_unpen25/unpen11/gppin_noise.json")
    sws = [s for s in sig if f"Langmuir|{s:g}" in U25]

    def pool(fn):
        return {s: float(np.mean([np.mean(fn(nm, s)) for nm in FAM])) for s in sws}
    arms = {"penalized": pool(lambda nm, s: C("ispline", nm, s)),
            "GP-log": pool(lambda nm, s: C("gp_log", nm, s)),
            "unpen. P=25": pool(lambda nm, s: [e["sat_err"] for e in U25[f"{nm}|{s:g}"]]),
            "unpen. P=11": pool(lambda nm, s: [e["sat_err"] for e in U11[f"{nm}|{s:g}"]]),
            "GP-log-pin": pool(lambda nm, s: [r["sat_err"] for r in GPP[f"gp_log_pin|{nm}|{s:g}"]])}
    print(f"\n  {'pooled sat err':22s}" + "".join(f"{'sy=' + format(s, 'g'):>11s}" for s in sws))
    for lab, v in arms.items():
        print(f"  {lab:22s}" + "".join(f"{v[s]:>11.3f}" for s in sws))
    for lab, v in arms.items():
        if lab != "penalized":
            print(f"  {'ratio ' + lab:22s}"
                  + "".join(f"{v[s] / arms['penalized'][s]:>11.2f}" for s in sws))
    for lab, U in (("unpen. P=25", U25), ("unpen. P=11", U11)):
        bad = sum(e["mono_bad"] for k, v in U.items() if k != "_meta" for e in v)
        tot = sum(e["mono_draws"] for k, v in U.items() if k != "_meta" for e in v)
        print(f"  {lab}: {bad} non-monotone draws of {tot} at all levels")
    print(f"\n  per family at sigma_y=0.1, GP-log error / unpen. P=11 error: " + ", ".join(
        f"{SHOW[nm]} {C('gp_log', nm, 0.1).mean() / np.mean([e['sat_err'] for e in U11[f'{nm}|0.1']]):.2f}x"
        for nm in FAM))


def sec_heldout():
    head("Held-out generated benchmark")
    R = load("results_heldout.json")
    if R is None:
        return skip("results_heldout.json")
    m = R["_meta"]

    GPP = load("results_heldout_gppin.json")

    def C(k, key="sat_err", gk=None):
        src = GPP["gp_log_pin"] if (k == "gp_log_pin" and GPP) else R.get(k, [])
        rs = src if gk is None else [r for r in src if r["gkind"] == gk]
        return np.array([r[key] for r in rs], float)

    print(f"  generated from {m['n_mix']} mixture-of-exponential and "
          f"{m['n_hill']} Hill ground truths,")
    print(f"  x_sat drawn log-uniformly on {m['xsat_range']} s; "
          f"the configuration never saw any of them.\n")
    print(f"  {'generator':12s}{'n':>5s}{'surrogate':>11s}{'mean':>9s}"
          f"{'median':>9s}{'p90':>9s}{'max':>9s}{'>sigma_y':>10s}")
    arms = ("ispline", "gp_log_pin", "gp_log") if GPP else ("ispline", "gp_log")
    for gk, lab in (("mix", "mixture"), ("hill", "Hill"), (None, "ALL")):
        for k in arms:
            v = C(k, "sat_err", gk)
            if not v.size:
                continue
            print(f"  {lab:12s}{v.size:>5d}{k:>11s}{v.mean():>9.3f}"
                  f"{np.median(v):>9.3f}{np.quantile(v,0.9):>9.3f}"
                  f"{v.max():>9.3f}{100*np.mean(v>SIGMA_Y):>9.0f}%")
    a, b = C("ispline"), C("gp_log")
    MAIN = load("results.json")
    if MAIN:
        dev = np.mean([np.mean([r["sat_err"] for r in MAIN["mc"][nm]["uncertainty"]])
                       for nm in FAM])
        print(f"\n  development-set mean for reference: {dev:.3f}")
    print(f"  paired: I-spline better on {100*np.mean(a<b):.0f}% of curves, "
          f"median ratio GP/I-spline {np.median(b/a):.2f}")
    if GPP:
        p = C("gp_log_pin")
        print(f"  vs GP-log-pin: I-spline better on {100*np.mean(a<p):.0f}% of "
              f"curves, median ratio {np.median(p/a):.2f}")
    print(f"  unconstrained arm: non-monotone mean {100*C('gp_log','dips').mean():.0f}%, "
          f"multi-valued {100*C('gp_log','multivalued').mean():.0f}%")


def sec_outside():
    head("Behavior outside the benchmark's assumptions")
    R = load("results_failure.json")
    if R is None:
        return skip("results_failure.json")
    arms = R["_meta"]["arms"]

    def C(arm, key, nm=None):
        rs = (R[f"{arm}|{nm}"] if nm else
              [r for f in FAM for r in R[f"{arm}|{f}"]])
        return np.array([r[key] for r in rs], float)

    print(f"  {'arm':13s}{'sat err':>9s}{'f(xmax)':>9s}{'x_sat':>8s}"
          f"{'CI width':>10s}{'true theta':>12s}{'r(octave)':>11s}"
          f"{'leverage':>10s}{'sigma':>8s}")
    for arm in arms:
        w = C(arm, "ci_hi") - C(arm, "ci_lo")
        print(f"  {arm:13s}{C(arm,'sat_err').mean():>9.3f}"
              f"{C(arm,'f_xmax').mean():>9.3f}{np.nanmean(C(arm,'x_sat')):>8.2f}"
              f"{np.nanmean(w):>10.2f}{np.nanmean(C(arm,'theta_true')):>12.3f}"
              f"{C(arm,'r_last_octave').mean():>11.3f}"
              f"{np.nanmean(C(arm,'plateau_lev')):>10.3f}"
              f"{C(arm,'sigma_post').mean():>8.3f}")
    rb, rt = C("baseline", "r_last_octave"), C("truncated", "r_last_octave")
    print(f"\n  window-sufficiency statistic r over the final octave:")
    print(f"    saturating window: mean {rb.mean():.3f}, "
          f"max {rb.max():.3f}")
    print(f"    truncated window : mean {rt.mean():.3f}, "
          f"min {rt.min():.3f}")
    for tol in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"    tol={tol:.2f}: false alarm {100*np.mean(rb>=tol):5.1f}%, "
              f"detection {100*np.mean(rt>=tol):5.1f}%")
    print(f"\n  plateau outlier (+3 sigma_y on the x_max reading):")
    for lab, nm in [("POOLED", None)] + [(SHOW[f], f) for f in FAM]:
        fb, fo, fr = (C("baseline", "f_xmax", nm), C("outlier", "f_xmax", nm),
                      C("outlier_rep", "f_xmax", nm))
        b, o, r = (C("baseline", "x_sat", nm), C("outlier", "x_sat", nm),
                   C("outlier_rep", "x_sat", nm))
        print(f"    {lab:18s} f(xmax) {fb.mean():.3f} -> {fo.mean():.3f} "
              f"(replicated {fr.mean():.3f});  x_sat {np.nanmean(b):.2f} -> "
              f"{np.nanmean(o):.2f} (replicated {np.nanmean(r):.2f}) s")
    print(f"\n  drift detector, true rate "
          f"{R['_meta']['drift_total']/19:.4f} per measurement:")
    for arm in ("baseline", "drift"):
        s = C(arm, "drift_slope")
        s = s[np.isfinite(s)]
        print(f"    {arm:10s} estimate {s.mean():+.4f} +/- "
              f"{s.std(ddof=1)/np.sqrt(max(len(s),1)):.4f} "
              f"(n={len(s)}, mean pairs/run {C(arm,'n_pairs').mean():.1f})")


def sec_survey():
    head("Survey-fraction sweep")
    R = load("results_survey.json")
    X = load("results_randxsat.json")
    if R is None or X is None:
        return skip("results_survey.json")
    n0s = R["_meta"]["n0_grid"]

    def C(n0, nm):
        return np.array([r["sat_err"] for r in R[f"{n0}|{nm}"]], float)

    def g(nm):
        return float(np.mean([r["history"][-1] for r in X["mc"][nm]["grid"]]))

    print(f"  {'curve':18s}" + "".join(f"{'N0='+str(n):>10s}" for n in n0s)
          + f"{'log grid':>11s}")
    for nm in FAM:
        print(f"  {SHOW[nm]:18s}"
              + "".join(f"{C(n,nm).mean():>10.3f}" for n in n0s)
              + f"{g(nm):>11.3f}")
    print(f"  {'pooled':18s}"
          + "".join(f"{np.mean([C(n,nm).mean() for nm in FAM]):>10.3f}"
                    for n in n0s)
          + f"{np.mean([g(nm) for nm in FAM]):>11.3f}")


def sec_knots():
    head("Best monotone approximation vs knot geometry")
    # read the cache design_choices.py knots writes, so this script stays
    # cache-only and never imports JAX
    K = load("results_knots.json")
    if not K:
        return skip("results_knots.json  (run: python3 design_choices.py knots)")
    cfgs = K["_meta"]["configs"]
    print("  {:24s}".format("curve") + "".join(f"{c:>20s}" for c in cfgs))
    print("  " + "-" * (24 + 20 * len(cfgs)))
    for name in K["_meta"]["curves"]:
        print("  {:24s}".format(name)
              + "".join(f"{K[name][c]:>20.4f}" for c in cfgs))


def sec_cost():
    head("Recurring cost of the recommended input")
    x_true, cyc = 2.0, 23.0
    # the median and one-sided (q=0.95) read-outs, pooled means over the 250 read-out
    # refits, from the cache; the percentages use the unrounded values
    RO = load("results_readout.json")
    if RO is None:
        return skip("results_readout.json")
    runs = [r for nm in FAM for r in RO[nm]]
    med = float(np.mean([r["x_sat_med"] for r in runs]))
    one = float(np.mean([np.array(r["x_q"])[int(np.argmin(np.abs(np.array(r["q"]) - 0.95)))] for r in runs]))
    over_med, over_one = med - x_true, one - x_true
    print(f"  pulse {x_true} -> {med:.3f} s  = +{100*over_med/x_true:.0f}% precursor "
          f"per cycle at constant delivery")
    print(f"  cycle {cyc:.0f} s -> {cyc+over_med:.1f} s = +{100*over_med/cyc:.1f}% cycle time "
          f"(median read-out)")
    print(f"  one-sided read-out {one:.3f} s adds {over_one:.3f} s/cycle = +{100*over_one/x_true:.0f}% precursor, "
          f"+{100*over_one/cyc:.0f}% cycle time")
    print(f"  one extra second of pulse per cycle = +{100/cyc:.1f}% cycle time, +{100/x_true:.0f}% of a 2 s dose")
    R = load("results.json")
    if R is None:
        return
    nd, dis, xm = [], [], []
    for nm in FAM:
        for d in R["mc"][nm]["uncertainty"]:
            t = np.array(d["t_obs"])
            act = t[5:]
            nd.append(sum(1 for i, x in enumerate(act)
                          if np.min(np.abs(t[:5 + i] - x)) < 0.02))
            ts = np.sort(t)
            dis.append(1 + int(np.sum(np.diff(ts) > 0.02)))
            xm.append(int(np.sum(np.abs(t - t.max()) < 1e-6)))
    print(f"\n  near-duplicate active selections: {np.mean(nd):.1f}/15 "
          f"({100*np.mean(nd)/15:.0f}%)")
    print(f"  distinct inputs in a realized N=20 design: {np.mean(dis):.1f}")
    print(f"  measurements at x_max in a realized N=20 design: {np.mean(xm):.1f}")


def sec_xsat():
    head("The recommended input x_sat: GP-log-pin vs I-spline")
    GX = load("results_gpxsat.json")
    RD = load("results_readout.json")
    if GX is None or RD is None:
        return skip("results_gpxsat.json")
    print(f"  {'family':18s}{'I bias':>9s}{'I 2-sd':>8s}{'I 1-sd':>8s}"
          f"{'GP bias':>9s}{'GP 2-sd':>8s}{'GP 1-sd':>8s}")
    for nm in FAM:
        ib = np.mean([r["x_sat_med"] - r["x_true"] for r in RD[nm]])
        i2 = 100 * np.mean([r["covered"] for r in RD[nm]])
        i1 = 100 * np.mean([r["covered_onesided"] for r in RD[nm]])
        gb = np.nanmean([r["bias"] for r in GX[nm]])
        g2 = 100 * np.mean([r["covered"] for r in GX[nm]])
        g1 = 100 * np.mean([r["covered_1s"] for r in GX[nm]])
        print(f"  {SHOW[nm]:18s}{ib:>+9.2f}{i2:>7.0f}%{i1:>7.0f}%"
              f"{gb:>+9.2f}{g2:>7.0f}%{g1:>7.0f}%")
    ib = np.mean([r["x_sat_med"] - r["x_true"] for nm in FAM for r in RD[nm]])
    gb = np.nanmean([r["bias"] for nm in FAM for r in GX[nm]])
    i2 = 100 * np.mean([r["covered"] for nm in FAM for r in RD[nm]])
    g2 = 100 * np.mean([r["covered"] for nm in FAM for r in GX[nm]])
    print(f"  {'POOLED':18s}{ib:>+9.2f}{i2:>7.0f}%{'':>8s}"
          f"{gb:>+9.2f}{g2:>7.0f}%")


def sec_window():
    head("Read-out vs window edge")
    W = load("results_window.json")
    if W is None:
        return skip("results_window.json")
    xmaxes = W["_meta"]["xmaxes"]

    def P(xm, k):
        return np.concatenate([[r[k] for r in W[f"{xm}|{nm}"]] for nm in FAM])
    print(f"  {'x_max':>7s}{'true frac':>11s}{'median x_sat':>14s}"
          f"{'1-sided lim':>13s}{'1-sd cov':>10s}{'2-sd cov':>10s}"
          f"{'r mean':>9s}{'FA@.15':>9s}")
    for xm in xmaxes:
        r = P(xm, "r_raw")
        print(f"  {xm:>7.1f}{np.nanmean(P(xm,'true_frac')):>11.3f}"
              f"{np.nanmean(P(xm,'x_sat')):>14.2f}{np.nanmean(P(xm,'one_sided')):>13.2f}"
              f"{100*np.mean(P(xm,'covered_1s')):>9.0f}%"
              f"{100*np.mean(P(xm,'covered_2s')):>9.0f}%"
              f"{r.mean():>9.3f}{100*np.mean(r>=0.15):>8.0f}%")


def sec_heldunif():
    head("Held-out under uniform x_sat measure")
    for tag, fn in (("log-uniform x_sat", "results_heldout.json"),
                    ("uniform x_sat", "results_heldout_uniform.json")):
        R = load(fn)
        if R is None:
            skip(fn)
            continue

        def C(k, gk=None):
            rs = R[k] if gk is None else [r for r in R[k] if r["gkind"] == gk]
            return np.array([r["sat_err"] for r in rs], float)
        a, b = C("ispline"), C("gp_log")
        print(f"  {tag:20s} I-spline all {a.mean():.3f} (max {a.max():.3f}, "
              f">sy {100*np.mean(a>SIGMA_Y):.0f}%)  GP-log all {b.mean():.3f} "
              f"(max {b.max():.3f}, >sy {100*np.mean(b>SIGMA_Y):.0f}%)  "
              f"median ratio {np.median(b/a):.2f}, I-spline better on "
              f"{100*np.mean(a<b):.0f}%")
        pf = "results_heldout_gppin.json" if "uniform" not in fn else "results_heldout_gppin_uniform.json"
        P = load(pf)
        if P is None:
            skip(pf)
            continue
        p = np.array([r["sat_err"] for r in P["gp_log_pin"]], float)
        print(f"  {'':20s} GP-log-pin all {p.mean():.3f} (max {p.max():.3f}, "
              f">sy {100*np.mean(p>SIGMA_Y):.0f}%)  median ratio {np.median(p/a):.2f}, "
              f"I-spline better on {100*np.mean(a<p):.0f}%")


def sec_anchor():
    """The high-noise mechanism diagnostic concerns the UNPENALIZED arm's
    reversal, so results_anchor_unpen11.json is the cache it needs; the production
    arm (results_anchor.json) is printed after it for contrast."""
    for cache, lab in (("results_anchor_unpen11.json",
                        "unpenalized monotone basis (P=11) -- the mechanism-diagnostic arm"),
                       ("results_anchor.json",
                        "production penalized surrogate -- for contrast")):
        _anchor_one(cache, lab)


def _anchor_one(cache, lab):
    head(f"Why the constraint loses at high noise: {lab}")
    R = load(cache)
    if R is None:
        return skip(cache)
    conds = [tuple(c) for c in R["_meta"]["conditions"]]
    fams = R["_meta"]["families"]

    def P(c, k):
        return np.concatenate([[r[k] for r in R["|".join(map(str, c)) + f"|{nm}"]]
                               for nm in fams])
    print(f"  {'condition':24s}{'sat err':>9s}{'plateau-corr':>14s}"
          f"{'f(xmax)':>9s}{'sigma':>8s}{'obs@xmax':>10s}")
    for c in conds:
        print(f"  sigma_y={c[0]}, s_sig={c[1]:<6}"
              f"{P(c,'sat_err').mean():>9.3f}"
              f"{P(c,'sat_err_plateau_corrected').mean():>14.3f}"
              f"{P(c,'f_xmax').mean():>9.3f}{P(c,'sigma_post').mean():>8.3f}"
              f"{P(c,'n_at_xmax').mean():>10.1f}")
    base = conds[1]
    gp = float(R["_meta"].get("gp_ref", {}).get("0.15", float("nan")))
    b = P(base, 'sat_err').mean()
    print(f"\n  gap at sigma_y=0.15: {b:.3f} (this arm) vs {gp:.3f} (GP-log, cached "
          f"reference) = {b - gp:+.3f}")
    print(f"  (A) plateau anchor: rescaling to the true plateau -> "
          f"{P(base,'sat_err_plateau_corrected').mean():.3f} (change {P(base,'sat_err_plateau_corrected').mean() - b:+.3f})")
    s06 = P(conds[2], 'sat_err').mean()
    print(f"  (C) noise prior: inferred sigma {P(base,'sigma_post').mean():.3f} "
          f"vs true 0.15; s_sig=0.6 -> {s06:.3f} (change {s06 - b:+.3f})")
    print(f"  loop replicates x_max unprompted: {P(conds[0],'n_at_xmax').mean():.1f} "
          f"(sy=.05) -> {P(base,'n_at_xmax').mean():.1f} (sy=.15) of 20")


def sec_prior():
    head("Prior sensitivity")
    P = load("results_priors.json")
    if P is None:
        return skip("results_priors.json  (run: python3 design_choices.py priors)")
    m = P["_meta"]
    fams, budgets, seeds = m["families"], m["budgets"], m["seeds"]
    cfgs = [c for c in m["configs"] if c != "baseline"]
    print(f"  model: {m.get('model', 'n/a')}\n")
    print(f"  {'family':18s}" + "".join(f"{'N=' + str(N):>14s}" for N in budgets))
    worst = 0.0
    for nm in fams:
        cells = []
        for N in budgets:
            v = [max(P[f"{nm}|{N}|{s}"].values()) for s in seeds]
            worst = max(worst, max(v))
            cells.append(f"{np.mean(v):.3f}/{max(v):.3f}")
        print(f"  {SHOW.get(nm, nm):18s}" + "".join(f"{c:>14s}" for c in cells))
    print("  (mean over seeds / worst seed)")
    print(f"\n  worst shift anywhere: {worst:.3f}")
    print("\n  which prior drives it (mean over families and seeds):")
    for cfg in cfgs:
        row = "  ".join(
            f"N={N}: {np.mean([P[f'{nm}|{N}|{s}'][cfg] for nm in fams for s in seeds]):.3f}"
            for N in budgets)
        print(f"    {cfg:16s}  {row}")


def sec_diag():
    head("MCMC convergence diagnostics")
    D = load("results_diag.json")
    if D is None:
        return skip("results_diag.json  (run: python3 design_choices.py diag)")
    print(f"  model: {D['_meta'].get('model', 'n/a')}   "
          f"chains={D['_meta']['chains']}  draws={D['_meta']['draws']}\n")
    print(f"  {'family':18s}{'max Rhat':>10s}{'min ESS':>9s}{'divergences':>13s}")
    for nm, d in D["per_family"].items():
        print(f"  {SHOW.get(nm, nm):18s}{d['rhat']:>10.3f}{d['ess']:>9.0f}{d['div']:>13d}")
    sc = D["single_chain"]
    print(f"\n  single-chain across {sc['fits']} production designs: "
          f"{sc['total_div']} divergences total, "
          f"{100 * sc['any_div'] / sc['fits']:.0f}% of fits with any; "
          f"ESS min {sc['ess_min']:.0f}, median {sc['ess_median']:.0f}")


def sec_acqgrid():
    head("Acquisition-grid control: linear vs geometric candidate set")
    A = load("results_acqgrid.json")
    if A is None:
        return skip("results_acqgrid.json")
    print(f"  {'curve':18s}{'linear':>10s}{'geometric':>12s}{'|diff|':>10s}"
          f"{'n<thresh':>11s}{'max/run':>9s}")
    diffs, belows, mx = [], [], []
    for nm in FAM:
        lin = np.mean([r["grid_err"] for r in A[nm]["uncertainty"]])
        geo = np.mean([r["grid_err"] for r in A[nm]["uncertainty_geom"]])
        b = np.mean([r["n_below"] for r in A[nm]["uncertainty"]])
        m = max(r["n_below"] for r in A[nm]["uncertainty"])
        diffs.append(abs(lin - geo)); belows.append(b); mx.append(m)
        print(f"  {SHOW[nm]:18s}{lin:>10.4f}{geo:>12.4f}{abs(lin-geo):>10.4f}"
              f"{b:>11.2f}{m:>9d}")
    j = int(np.argmax(diffs))
    print(f"\n  largest shift {max(diffs):.4f} on {SHOW[FAM[j]]}")
    print(f"  near-origin active selections: {np.mean(belows):.2f} per run pooled, "
          f"at most {max(belows):.2f} per run on any family, "
          f"never more than {max(mx)} in a single run")


def sec_regimes():
    head("Noise-free regime sweeps (basis capacity)")
    R = load("results.json")
    if R is None or "regimes" not in R:
        return skip("results.json['regimes']")
    reg = R["regimes"]
    allv = []
    for nm, variants in reg.items():
        cells = "   ".join(f"{k} {v['sat_err']:.4f}" for k, v in variants.items())
        allv += [(nm, k, v["sat_err"]) for k, v in variants.items()]
        print(f"  {nm:12s} {cells}")
    allv.sort(key=lambda t: -t[2])
    worst = allv[0]
    rest = [v for _, _, v in allv[1:]]
    print(f"\n  worst variant: {worst[0]} {worst[1]} = {worst[2]:.4f}")
    print(f"  the other {len(rest)} variants span {min(rest):.4f} to "
          f"{max(rest):.4f}")


def sec_learning():
    head("Learning-curve statistics (budgets and reach rates)")
    R = load("results.json")
    if R is None:
        return skip("results.json")
    tag = R.get("tag", {})
    n0, na = int(tag.get("n_init", 5)), int(tag.get("n_iters", 15))
    Ns = np.arange(n0, n0 + na + 1)

    def first(hist, thr):
        i = np.where(np.asarray(hist) <= thr)[0]
        return Ns[i[0]] if i.size else np.nan
    print(f"  median budget to reach the output-noise level "
          f"(sigma_y={SIGMA_Y}), fixed-location campaign:")
    for nm in FAM:
        v = [first(r["history"], SIGMA_Y) for r in R["mc"][nm]["uncertainty"]]
        print(f"    {SHOW[nm]:18s} median N = {np.nanmedian(v):.0f}   "
              f"(reached by N=20 in {100*np.mean(~np.isnan(v)):.0f}% of seeds)")

    RX = load("results_randxsat.json")
    if RX:
        print(f"\n  fraction of seeds at or below the output-noise level at "
              f"N={Ns[-1]}, randomized-location campaign:")
        print(f"    {'family':18s}" + "".join(f"{st:>13s}" for st in
              ("uncertainty", "grid", "random_log", "random")))
        for nm in FAM:
            cells = []
            for st in ("uncertainty", "grid", "random_log", "random"):
                f = np.mean([r["history"][-1] <= SIGMA_Y for r in RX["mc"][nm][st]])
                cells.append(f"{100*f:.0f}%")
            print(f"    {SHOW[nm]:18s}" + "".join(f"{c:>13s}" for c in cells))


def sec_ablation():
    head("Penalty-order ablation (results_model_ablation.json)")
    MA = load("results_model_ablation.json")
    if MA is None:
        return skip("results_model_ablation.json")
    from scipy import stats
    fams, seeds = MA["_meta"]["families"], MA["_meta"]["seeds"]

    def col(arm, k):
        return np.array([MA[f"{f}|{s}"][arm][k] for f in fams for s in seeds], float)
    print("  medians over the 250 paired refits; coverage = fraction of runs covering x_sat=2.0")
    print(f"  {'arm':14s}{'sat err':>9s}{'bias (s)':>10s}{'coverage':>10s}{'paired p vs 2nd':>17s}")
    ref = col("second_order", "sat_err")
    for arm in MA["_meta"]["arms"]:
        se, b, c = col(arm, "sat_err"), col(arm, "bias"), col(arm, "covered")
        p = "" if arm == "second_order" else f"{stats.wilcoxon(se - ref).pvalue:.0e}"
        print(f"  {arm:14s}{np.median(se):9.3f}{np.median(b):+10.2f}{100*c.mean():9.0f}%{p:>17s}")
    se, b, c = col("hetero", "sat_err"), col("hetero", "bias"), col("hetero", "covered")
    bad = se > 0.5
    print(f"  hetero fits whose variance term collapsed (sat err > 0.5): {bad.sum()} of {len(se)}")
    print(f"  hetero on the {(~bad).sum()} converged fits: median bias {np.median(b[~bad]):+.2f} s, "
          f"coverage {100*c[~bad].mean():.0f}%")


def sec_sampler():
    head("Sampler configuration and the paired settings comparison")
    T = load("results_target_accept.json")
    if T is not None:
        fams = T["_meta"]["families"]
        nd = T["_meta"]["chains"] * T["_meta"]["draws"]
        print(f"  {'configuration':26s}{'Rhat range':>20s}{'ESS range':>16s}"
              f"{'div range':>16s}{'stranded':>10s}")
        for lab, *_ in T["_meta"]["grid"]:
            r = [T[f"{f}|{lab}"] for f in fams]
            rh = [x["rhat"] for x in r]; es = [x["ess"] for x in r]
            dv = [100 * x["div"] / nd for x in r]
            st = sum(1 for x in r if x["max_abs_mu"] > 10)
            print(f"  {lab:26s}{min(rh):9.3f}-{max(rh):9.3f}"
                  f"{min(es):7.0f}-{max(es):7.0f}"
                  f"{min(dv):7.1f}%-{max(dv):6.1f}%{st:>7d}/5")
        print("  (stranded = posterior-mean curve orders of magnitude above the "
              "plateau of one)")
    else:
        skip("results_target_accept.json")

    S = load("results_sampler_settings.json")
    if S is None:
        return skip("results_sampler_settings.json")
    from scipy import stats as _sst
    fams, sds = S["_meta"]["families"], S["_meta"]["seeds"]
    pr = np.array([S[f"{nm}|{s}"]["production"]["sat_err"] for nm in fams for s in sds])
    fx = np.array([S[f"{nm}|{s}"]["fixed"]["sat_err"] for nm in fams for s in sds])
    dm = np.array([S[f"{nm}|{s}"]["max_dmu"] for nm in fams for s in sds])
    dp = np.array([S[f"{nm}|{s}"]["production"]["div"] for nm in fams for s in sds])
    df = np.array([S[f"{nm}|{s}"]["fixed"]["div"] for nm in fams for s in sds])
    print(f"\n  paired over {len(pr)} fits on identical designs:")
    print(f"    saturation error   production {pr.mean():.4f}   fixed {fx.mean():.4f}"
          f"   paired Wilcoxon p={_sst.wilcoxon(pr, fx).pvalue:.2f}")
    print(f"    shift is {abs(fx.mean()-pr.mean())/pr.std(ddof=1):.2f} seed-to-seed sd")
    print(f"    max|dmu|           mean {dm.mean():.4f}  worst {dm.max():.4f}"
          f"   against sigma_y={SIGMA_Y}")
    print(f"    divergences/fit    production {dp.mean():.1f}   fixed {df.mean():.1f}"
          f"   ({dp.mean()/df.mean():.1f}x fewer)")


def sec_band():
    head("Calibration of the predictive band (results_band.json)")
    B = load("results_band.json")
    if B is None:
        return skip("results_band.json  (run: python3 controls.py band)")
    print("  coverage = fraction of (grid point, seed) pairs inside "
          "mu +/- 1.96 sigma_*,")
    print("  on the evaluation grid every other metric uses.\n")
    print("  coverage over the whole window and over the part with data (x >= x_min);")
    print("  Spearman rho_s between band width and realized error, three conventions:")
    print("  every point and seed, points with data only, and across seeds at fixed x.\n")
    print(f"  {'arm':18s}{'coverage':>10s}{'cov x>=xmin':>13s}{'rho_s all':>11s}"
          f"{'rho_s data':>12s}{'rho_s seeds':>13s}{'band/spread':>13s}")
    for lab, key in (("penalized P=25", "penalized"),
                     ("unpenalized P=11", "unpenalized_p11")):
        d = B[key]
        p = d["pooled"]
        bs = np.mean([d[nm]["band_over_spread"] for nm in FAM])
        print(f"  {lab:18s}{100 * p['coverage']:>9.1f}%{100 * p['coverage_in']:>12.1f}%"
              f"{p['rho']['pointwise']:>11.3f}{p['rho']['pointwise_in']:>12.3f}"
              f"{p['rho']['across_seeds']:>13.3f}{bs:>13.2f}")
    for tag, k in (("whole window", "coverage"), ("x >= x_min", "coverage_in")):
        print(f"\n  per-family coverage (penalized, {tag}): " + ", ".join(
            f"{SHOW[nm]} {100 * B['penalized'][nm][k]:.0f}%" for nm in FAM))
    if "sigma" in B:
        print(f"\n  inferred scales at N=20 (generative sigma_y={SIGMA_Y}):")
        for nm in FAM:
            d = B["sigma"][nm]
            print(f"    {SHOW[nm]:18s}sigma {d['sigma_mean']:.3f} +/- {d['sigma_sd']:.3f}"
                  f"   ({d['excess_pct']:+.0f}%)   tau_eta {d['tau_mean']:.3f}")


def sec_decoupled():
    head("Unconstrained-driven acquisition (results_decoupled.json)")
    D = load("results_decoupled.json")
    R = load("results.json")
    if D is None or R is None:
        return skip("results_decoupled.json")
    fams = D["_meta"]["families"]
    seeds = sorted({int(k.split("|")[1]) for k in D if k != "_meta"})
    cur = np.array([D[f"{nm}|{s}"]["curve"] for nm in fams for s in seeds])
    Ns, drv, cln = cur[0, :, 0], cur[:, :, 1].mean(0), cur[:, :, 2].mean(0)
    own = np.array([r["sat_err"] for nm in fams
                    for r in R["mc"][nm]["uncertainty"]])
    fin = np.array([D[f"{nm}|{s}"]["cleanup_final"] for nm in fams for s in seeds])
    from scipy import stats as _st
    ratio = cln / drv
    print(f"  ratio (monotone cleanup / unconstrained driver) at N=5: "
          f"{ratio[0]:.2f}x")
    j = Ns >= 8
    print(f"  ratio across N=8..20: {ratio[j].min():.3f} to {ratio[j].max():.3f}")
    print(f"  N=20 decoupled {fin.mean():.3f} vs own-loop {own.mean():.3f}  "
          f"(paired p={_st.wilcoxon(fin, own).pvalue:.3f})")
    print(f"  non-monotone draws in the cleanup fits: "
          f"{sum(D[f'{nm}|{s}']['cleanup_nonmono'] for nm in fams for s in seeds)}")


def sec_monoproj():
    head("Post-hoc monotonization of GP-log-pin (results_mono_projection.json)")
    P = load("results_mono_projection.json")
    R = load("results.json")
    if P is None:
        return skip("results_mono_projection.json")
    fams, seeds = P["_meta"]["families"], P["_meta"]["seeds"]

    def col(k):
        return np.array([P[f"{nm}|{s}"][k] for nm in fams for s in seeds], float)
    pen = (np.mean([np.mean([r["sat_err"] for r in R["mc"][nm]["uncertainty"]])
                    for nm in FAM]) if R else float("nan"))
    print(f"  curve error   raw GP {np.nanmean(col('raw_err')):.3f}   "
          f"PAVA {np.nanmean(col('pava_err')):.3f}   "
          f"Lin-Dunson {np.nanmean(col('ld_err')):.3f}   "
          f"penalized {pen:.3f}  (ratio {np.nanmean(col('raw_err'))/pen:.2f}x)")
    # bias is the MEDIAN of the per-run biases: the read-out convention used by
    # controls.py (the mean reads 0.01 lower)
    print(f"  x_sat(0.95)   Lin-Dunson bias {np.nanmedian(col('ld_bias')):+.2f} s, "
          f"coverage {100*np.nanmean(col('ld_cov')):.0f}%")
    print(f"  raw GP posterior mean non-monotone in "
          f"{100*np.nanmean(col('raw_dips')):.0f}% of fits")


def sec_gpmatched():
    head("GP matched-design control (results_gp_matched.json)")
    Q = load("results_gp_matched.json")
    if Q is None:
        return skip("results_gp_matched.json  (run: python3 gp_arms.py matched)")
    from scipy import stats
    M_, O, meta = Q["matched"], Q["own_loop"], Q["_meta"]
    print("  saturation error at N=20, GP arms fitted to the I-spline campaign's stored designs\n")
    lin_ok = "gp_lin" in M_
    print(f"  {'design':16s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}{'paired p':>11s}"
          + (f"{'GP-linear':>11s}" if lin_ok else ""))
    for st in meta["strategies"]:
        isp = np.array([r["ispline_sat_err"] for nm in FAM for r in M_["gp_log_pin"][st][nm]])
        pin = np.array([r["sat_err"] for nm in FAM for r in M_["gp_log_pin"][st][nm]])
        log = np.array([r["sat_err"] for nm in FAM for r in M_["gp_log"][st][nm]])
        lin = (f"{np.mean([r['sat_err'] for nm in FAM for r in M_['gp_lin'][st][nm]]):>11.4f}"
               if lin_ok and st in M_["gp_lin"] else "")
        print(f"  {st:16s}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}"
              f"{pin.mean()/isp.mean():>9.2f}{stats.wilcoxon(pin - isp).pvalue:>11.0e}{lin}")
    G = load("results_gparms.json")
    if G is not None and lin_ok:
        print(f"  GP-linear in its own loop: "
              f"{np.mean([r['sat_err'] for nm in FAM for r in G['gp_lin'][nm]]):.4f}")
    print("\n  own loops: design clustering and band flatness, mean over the 250 runs")
    print(f"  {'arm':11s}{'distinct/20':>12s}{'at xmax':>9s}{'flat share':>12s}{'sdmin/sdmax':>13s}{'ell med/span':>14s}")
    for arm in ["ispline"] + meta["arms"]:
        rs = [r for nm in FAM for r in O[arm][nm]]
        extra = ""
        if arm != "ispline":
            extra = (f"{np.mean([r['flat_share'] for r in rs]):>12.2f}{np.mean([r['sd_range'] for r in rs]):>13.2f}"
                     f"{np.median([r['ell_median'] for r in rs]) / meta['warped_span']:>14.2f}")
        print(f"  {arm:11s}{np.mean([r['n_distinct'] for r in rs]):>12.1f}"
              f"{np.mean([r['n_at_xmax'] for r in rs]):>9.1f}{extra}")
    _matched_sweeps(Q)
    U = load("results_unpen_matched.json")
    if U is not None:
        print("\n  unpenalized bases on the stored designs (refits.py unpen), pooled sat err at N=20")
        print(f"  {'design':14s}" + "".join(f"{a:>10s}" for a in U["_meta"]["arms"]))
        for st in U["_meta"]["strategies"]:
            print(f"  {st:14s}" + "".join(f"{np.mean([v for nm in FAM for v in U[a][st][nm]]):>10.4f}"
                                          for a in U["_meta"]["arms"]))


def _matched_sweeps(Q):
    """The matched-design control on the noise sweep and the held-out benchmark:
    GP arms fitted to the I-spline's designs, beside the GP-log-pin own-loop
    caches (cache-only; the same numbers gp_arms.py matched prints)."""
    from scipy.stats import wilcoxon
    if "noise" in Q:
        PN, NO = load("results_gppin_noise.json"), load("results_noise.json")
        print("\n  noise sweep: GP arms on the I-spline designs at each sigma_y")
        print(f"  {'sigma_y':8s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}"
              f"{'paired p':>11s}{'pin own loop':>14s}{'own/isp':>9s}{'log own loop':>14s}")
        N = Q["noise"]
        for sy in sorted({float(k.split("|")[1]) for k in N["gp_log_pin"]}):
            keys = [k for k in N["gp_log_pin"] if abs(float(k.split("|")[1]) - sy) < 1e-12]
            isp = np.array([r["ispline_sat_err"] for k in keys for r in N["gp_log_pin"][k].values()])
            pin = np.array([r["sat_err"] for k in keys for r in N["gp_log_pin"][k].values()])
            log = np.array([r["sat_err"] for k in keys for r in N["gp_log"][k].values()])
            po = (np.mean([r["sat_err"] for nm in FAM for r in PN.get(f"gp_log_pin|{nm}|{sy:g}", [])])
                  if PN else float("nan"))
            lo = (np.mean([r["sat_err"] for nm in FAM for r in NO.get(f"gp_log|{nm}|{sy:g}", [])])
                  if NO else float("nan"))
            print(f"  {sy:<8g}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}"
                  f"{pin.mean()/isp.mean():>9.2f}{wilcoxon(pin - isp).pvalue:>11.0e}"
                  f"{po:>14.4f}{po/isp.mean():>9.2f}{lo:>14.4f}")
    if "heldout" in Q:
        print("\n  held-out benchmark: GP arms on the I-spline designs")
        print(f"  {'x_sat draw':12s}{'I-spline':>10s}{'GP-log-pin':>12s}{'GP-log':>9s}{'pin/isp':>9s}"
              f"{'paired p':>11s}{'pin own loop':>14s}{'own/isp':>9s}  >sigma_y and max: isp | pin on isp designs")
        H = Q["heldout"]
        for measure, own in (("loguniform", "results_heldout_gppin.json"),
                             ("uniform", "results_heldout_gppin_uniform.json")):
            if measure not in H["gp_log_pin"]:
                continue
            isp = np.array([r["ispline_sat_err"] for r in H["gp_log_pin"][measure].values()])
            pin = np.array([r["sat_err"] for r in H["gp_log_pin"][measure].values()])
            log = np.array([r["sat_err"] for r in H["gp_log"][measure].values()])
            O = load(own)
            po = np.array([r["sat_err"] for r in O["gp_log_pin"]]) if O else np.array([np.nan])
            print(f"  {measure:12s}{isp.mean():>10.4f}{pin.mean():>12.4f}{log.mean():>9.4f}"
                  f"{pin.mean()/isp.mean():>9.2f}{wilcoxon(pin - isp).pvalue:>11.0e}"
                  f"{po.mean():>14.4f}{po.mean()/isp.mean():>9.2f}"
                  f"  {100*np.mean(isp > SIGMA_Y):.0f}% {isp.max():.3f} | "
                  f"{100*np.mean(pin > SIGMA_Y):.0f}% {pin.max():.3f}")


def sec_studentt():
    head("Student-t likelihood against a contaminated plateau reading (results_studentt.json)")
    S = load("results_studentt.json")
    if S is None:
        return skip("results_studentt.json  (run: python3 refits.py studentt)")
    fams, seeds = S["_meta"]["families"], S["_meta"]["seeds"]

    def col(arm, cond, k):
        return np.array([S[f"{nm}|{s}"][f"{arm}|{cond}"][k] for nm in fams for s in seeds], float)
    print(f"  {len(fams) * len(seeds)} refits of the stored N=20 designs, contamination: "
          f"{S['_meta']['contamination']}\n")
    print(f"  {'likelihood':22s}{'data':10s}{'sat err':>9s}{'f(xmax)':>9s}{'median x_sat':>14s}")
    for arm, lab in (("gauss", "Gaussian"), ("studentt", f"Student-t (nu={S['_meta']['nu']:g})")):
        for cond in ("clean", "contam"):
            print(f"  {lab:22s}{cond:10s}{col(arm, cond, 'sat_err').mean():>9.3f}"
                  f"{col(arm, cond, 'f_xmax').mean():>9.3f}"
                  f"{np.nanmedian(col(arm, cond, 'x_sat')):>14.2f}")
    print("\n  median per-run shift caused by the bad reading (contaminated minus clean):")
    sh = {}
    for arm, lab in (("gauss", "Gaussian"), ("studentt", "Student-t")):
        df = np.nanmedian(col(arm, "contam", "f_xmax") - col(arm, "clean", "f_xmax"))
        dx = np.nanmedian(col(arm, "contam", "x_sat") - col(arm, "clean", "x_sat"))
        sh[arm] = (df, dx)
        print(f"    {lab:12s} plateau {df:+.3f}   read-out {dx:+.2f} s")
    print(f"  Student-t / Gaussian shift: plateau {sh['studentt'][0] / sh['gauss'][0]:.2f}, "
          f"read-out {sh['studentt'][1] / sh['gauss'][1]:.2f}")


def sec_parametric():
    head("Parametric baselines: least-squares fits of the five closed forms (results_parametric.json)")
    PA = load("results_parametric.json")
    if PA is None:
        return skip("results_parametric.json  (run: python3 refits.py parametric)")
    fams, seeds = PA["_meta"]["families"], PA["_meta"]["seeds"]
    forms, TF = PA["_meta"]["forms"], PA["_meta"]["trueform"]
    xt = PA["_meta"]["true_xsat"]

    def rows(kind):
        vals = []
        for nm in fams:
            for s in seeds:
                rec = PA[f"{nm}|{s}"]
                if kind == "oracle":
                    if rec[TF[nm]]:
                        vals.append(rec[TF[nm]]["sat_err"])
                else:
                    w = [rec[f]["sat_err"] for f in forms if f != TF[nm] and rec[f]]
                    if w:
                        vals.append(np.mean(w) if kind == "mean" else max(w))
        return float(np.nanmean(vals))
    R = load("results.json")
    pen = (np.mean([np.mean([r["sat_err"] for r in R["mc"][nm]["uncertainty"]]) for nm in FAM])
           if R else float("nan"))
    print(f"  pooled saturation error on the stored N=20 designs (penalized surrogate {pen:.3f}):")
    for kind, lab in (("oracle", "oracle (true form)"), ("mean", "mean misspecified form"),
                      ("worst", "worst misspecified form")):
        v = rows(kind)
        print(f"    {lab:26s}{v:.3f}   ({v / pen:.2f}x the penalized surrogate)")
    print("\n  oracle x_sat bias per family (median over seeds, then mean):")
    for nm in fams:
        b = np.array([PA[f"{nm}|{s}"][TF[nm]]["x_sat"] - xt for s in seeds
                      if PA[f"{nm}|{s}"][TF[nm]]], float)
        print(f"    {SHOW.get(nm, nm):18s}{np.nanmedian(b):+.2f} s   {np.nanmean(b):+.2f} s")
    print("\n  mean saturation error of each wrong form, per family:")
    print(f"    {'family':18s}" + "".join(f"{f:>13s}" for f in forms))
    worst = (0.0, "", "")
    for nm in fams:
        cells = []
        for f in forms:
            v = [PA[f"{nm}|{s}"][f]["sat_err"] for s in seeds if PA[f"{nm}|{s}"][f]]
            m = float(np.mean(v)) if v else float("nan")
            cells.append(("(true)" if f == TF[nm] else f"{m:.3f}"))
            if f != TF[nm] and m > worst[0]:
                worst = (m, nm, f)
        print(f"    {SHOW.get(nm, nm):18s}" + "".join(f"{c:>13s}" for c in cells))
    print(f"  largest: {worst[0]:.3f}, the {worst[2]} form on {SHOW.get(worst[1], worst[1])}")


def sec_misc():
    head("Analytic and per-variant numbers (isotherms.py, the prior, results.json, results_readout.json)")
    import isotherms as ISO
    # the two steepest ground-truth slopes on [x_min, x_max], and the output
    # scatter the input jitter adds there
    xs = np.geomspace(ISO.DOMAIN[0], ISO.DOMAIN[1], 4000)
    h = 1e-7
    sl = {}
    for nm, f in ISO.standard_isotherms().items():
        sl[nm] = float(((f(xs + h) - f(np.maximum(xs - h, 0.0))) / (2 * h)).max())
    print("  steepest slope on [x_min, x_max]: " + ", ".join(
        f"{SHOW.get(nm, nm)} {v:.1f}" for nm, v in sorted(sl.items(), key=lambda t: -t[1])) + " s^-1")
    print(f"  sigma_x times the steepest slope: {ISO.SIGMA_X * max(sl.values()):.2f}")
    # the production prior on its own: the last log-weight of the second-order
    # walk (P=25), log w0 ~ N(log 1/P, 1.5), slope0 ~ N(0, 0.5),
    # tau_eta ~ HalfNormal(0.3), second differences tau_eta * N(0, 1)
    P, n = 25, 400_000
    rng = np.random.default_rng(0)
    lw0 = rng.normal(np.log(1.0 / P), 1.5, n)
    s0 = rng.normal(0.0, 0.5, n)
    tau = np.abs(rng.normal(0.0, 0.3, n))
    z = rng.normal(0.0, 1.0, (n, P - 2))
    fd = s0[:, None] + np.concatenate([np.zeros((n, 1)), np.cumsum(tau[:, None] * z, axis=1)], axis=1)
    last = lw0 + fd.sum(axis=1)
    print(f"  prior alone, last log-weight: s.d. {last.std():.1f}, "
          f"|log w| > 75 in {100 * np.mean(np.abs(last) > 75):.1f}% of draws")
    R = load("results.json")
    if R is not None and "regimes" in R:
        reg = R["regimes"]
        mm = reg["Dissociative"]["K=0.25"]
        f_true = ISO.make_isotherm("Dissociative", K=0.25)
        print(f"  noise-free sweep, Michaelis-Menten K=0.25: inferred plateau "
              f"{mm['f_max']:.3f}, true value at x_max {float(f_true(ISO.DOMAIN[1])):.3f}")
        print(f"  noise-free sweep, KWW beta=0.35: response error at the 25% level "
              f"{reg['DR-KWW']['b=0.35']['y_err']['25']:+.2f}")


def sec_sigma():
    head("Inferred noise and roughness scales")
    B = load("results_band.json")
    if B is None or "sigma" not in B:
        return skip("results_band.json  (run: python3 controls.py band)")
    S = B["sigma"]
    print(f"  refit of the stored N=20 designs, {S['_n_seeds']} seeds; "
          f"generative sigma_y={SIGMA_Y}, tau_eta prior HalfNormal({S['_tau_prior']['scale']}) "
          f"with median {S['_tau_prior']['median']:.3f}\n")
    print(f"  {'family':18s}{'sigma':>16s}{'excess':>9s}{'tau_eta':>16s}{'within-fit sd':>15s}")
    for nm in FAM:
        d = S[nm]
        print(f"  {SHOW[nm]:18s}{d['sigma_mean']:.3f} +/- {d['sigma_sd']:.3f}"
              f"{d['excess_pct']:+8.0f}%   {d['tau_mean']:.3f} +/- {d['tau_sd']:.3f}"
              f"{d['tau_psd_mean']:>15.3f}")


SECTIONS = {"iso": sec_iso, "acq": sec_acq, "level": sec_level, "gp": sec_gp,
            "readout": sec_readout, "xsat": sec_xsat,
            "window": sec_window, "noise": sec_noise, "anchor": sec_anchor,
            "heldout": sec_heldout, "heldunif": sec_heldunif,
            "outside": sec_outside, "survey": sec_survey,
            "knots": sec_knots, "cost": sec_cost, "prior": sec_prior,
            "sigma": sec_sigma,
            "band": sec_band, "sampler": sec_sampler, "diag": sec_diag,
            "learning": sec_learning, "regimes": sec_regimes,
            "acqgrid": sec_acqgrid,
            "decoupled": sec_decoupled, "monoproj": sec_monoproj,
            "ablation": sec_ablation, "gpmatched": sec_gpmatched,
            "studentt": sec_studentt, "parametric": sec_parametric, "misc": sec_misc}

def main(argv):
    want = argv if argv else list(SECTIONS)
    for k in want:
        if k not in SECTIONS:
            print(f"unknown section {k!r}; choose from {list(SECTIONS)}")
            return 1
        SECTIONS[k]()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
