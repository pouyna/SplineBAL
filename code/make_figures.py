"""
make_figures.py
===============
Figures for the monotone I-spline saturation-curve active-learning study.

Study design
------------
* Monte Carlo comparison (per isotherm): EVERY one of the five headline
  isotherms is fit over MC_SEEDS seeds under four acquisition strategies,
  uncertainty sampling (active), uniform random, log-uniform random and a
  log-spaced grid, under the standard noise model, recording the saturation
  error vs N.  The uncertainty-sampling arm doubles as the recovery study (its
  median-error seed illustrates each fit) and supplies the per-level error analysis
  below.
* Noise-free regime sweeps: three regime variants per family (see
  isotherms.family_variant_specs), fit once each with sigma_x = sigma_y = 0,
  isolating the approximation error of the basis from statistical noise.

Metrics
-------
* Saturation error across levels (overall fit quality): at the input where
  the true curve reaches each saturation level (5%, 10%, ..., 95%), the
  RMS difference between the predicted and true response.
* Per-level errors at 5/25/50/75/95% saturation, recorded per run:
    y_err[a] = posterior mean at the TRUE crossing time of a*A, minus a*A
               (vertical / response error, signed);
    x_err[a] = median model-free crossing time of a*f(x_max) across posterior
               draws, minus the true crossing time of a*A
               (horizontal / input error, signed, seconds).

Usage
-----
    python make_figures.py            # default set: basis final_fit details
    python make_figures.py all        # every figure
    python make_figures.py knots readout noise scorecard gp_compare   # by name
    python make_figures.py basis      # basis only (no MCMC)
    python make_figures.py --fresh    # force re-run of the campaign figures
    python make_figures.py --fresh --jobs=8   # re-run on 8 worker processes

The campaign behind final_fit/details is 5 families x 4 strategies x MC_SEEDS
seeds plus the regime variants; the runs are independent, so --fresh
distributes them over worker processes (default: all but one core).  Each run
is seeded, so the results are identical however many workers are used.
knots, readout, noise and scorecard read caches; gp_compare runs a short
live illustration loop (one seed) and is the only figure that refits.
"""

import sys
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# Uniform typography across all figures.  Each figure is included in the
# document at full column width, so a wide (multi-panel) figure is scaled down more
# than a narrow one.  To make the on-page font sizes match, we scale every
# figure's fonts by its width relative to a reference width FIG_W: a figure
# twice as wide uses twice the point size, and the two cancel on the page.
FIG_W = 6.5
_BASE = {"axes.titlesize": 10.0, "axes.labelsize": 9.5, "legend.fontsize": 8.0,
         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.titlesize": 11.0}


def _fonts(width):
    """Scale rcParams fonts for a figure of given width; return the scale."""
    s = width / FIG_W
    plt.rcParams.update({k: v * s for k, v in _BASE.items()})
    return s

import ispline_model as M
import isotherms as ISO
import poolutil

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(HERE, "data")   # result caches live in code/data
_FIG_DIR = os.path.join(HERE, "figures")  # generated PNGs
CACHE    = os.path.join(DATA, "results.json")
DOMAIN   = ISO.DOMAIN
MC_SEEDS = list(range(50))
N_INIT   = 5             # survey design: T_min, T_max, + 3 log-LHS
N_ITERS  = 15            # N = 5 .. 20
# Acquisition strategies compared per family: active (uncertainty sampling)
# against three baselines -- uniform random (the weak baseline), log-uniform
# random (the fair random baseline on a log-scaled domain), and a
# non-adaptive log-spaced grid (what a practitioner runs).
METHODS  = ["uncertainty", "random", "random_log", "grid"]
METHOD_LABEL = {"uncertainty": "uncertainty sampling (active)",
                "random": "random (uniform)",
                "random_log": "random (log-uniform)",
                "grid": "log-spaced grid"}
# Display names for figures/legends (decoupled from internal keys and from the
# figure filenames, which are unchanged).
DISPLAY = {"Langmuir": "Langmuir", "Dissociative": "Michaelis-Menten",
           "Sips (n=2)": "Sips ($n=2$)", "Elovich": "Elovich", "D-R KWW": "KWW"}
LEVELS5  = (0.05, 0.25, 0.50, 0.75, 0.95)   # per-level error read-out points
FAMILIES = list(ISO.STANDARD_SPECS)

_GPARMS_CACHE = {}


def _gparms():
    """Lazy loader for the Gaussian-process ablation (gp_arms.py).

    Returns None if the arm has not been run, so the isotherm figures still
    build from results.json alone.
    """
    if "d" not in _GPARMS_CACHE:
        p = os.path.join(DATA, "results_gparms.json")
        try:
            with open(p) as fh:
                _GPARMS_CACHE["d"] = json.load(fh)
        except (OSError, ValueError):
            _GPARMS_CACHE["d"] = None
    return _GPARMS_CACHE["d"]


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------
def _fit_record(f, seed, acquisition, record_history=False,
                sigma_x=ISO.SIGMA_X, sigma_y=ISO.SIGMA_Y, light=False):
    """One run's record.  light=True (used for the baseline arms) keeps the
    history, per-level errors, and scalars but drops the 400-point curve
    arrays, which are only needed for the uncertainty-sampling figures."""
    observe = ISO.make_observe(f, domain=DOMAIN, sigma_x=sigma_x, sigma_y=sigma_y)
    reg, hist = M.run_active_learning(
        observe, domain=DOMAIN, seed=seed, n_init=N_INIT, n_iters=N_ITERS,
        record_history=record_history, truth=f, acquisition=acquisition)
    t = M.eval_points(0.0, DOMAIN[1])
    mu, sd   = reg.predict(t)
    dmu, dsd = reg.predict_deriv(t)
    yt  = f(t)
    rep = reg.monotonicity_report()
    ts  = reg.tstar(ISO.S_TARGET)
    tstar_med = float(np.median(ts)) if len(ts) else float("nan")
    theta = float(f(np.array([tstar_med]))[0] / ISO.A_DEFAULT) if len(ts) else float("nan")
    # Per-level signed errors at 5/25/50/75/95% saturation.  True crossing
    # times are root-found on the ground-truth callable (a uniform grid is
    # too coarse near the origin); the predicted crossing follows the
    # model-free read-out (level relative to the model's own plateau).
    # Only the uncertainty arm feeds the per-level error analysis, so skip this
    # (relatively expensive) read-out for the light baseline records.
    y_err, x_err = {}, {}
    if not light:
        for a in LEVELS5:
            key = str(int(round(a * 100)))
            xa = ISO.true_level_time(f, a * ISO.A_DEFAULT)
            mu_a, _ = reg.predict(np.array([xa]))
            y_err[key] = float(mu_a[0]) - a * ISO.A_DEFAULT
            ts_a = reg.tstar(a, n_grid=2000)
            x_err[key] = float(np.median(ts_a)) - xa if len(ts_a) else float("nan")
    rec = dict(
        t_obs=list(map(float, reg.t_obs)), y_obs=list(map(float, reg.y_obs)),
        sat_err=M.saturation_error(t, mu, yt),
        f_max=rep["f_max"], non_monotone=rep["non_monotone"], n_draws=rep["n_draws"],
        non_monotone_all=getattr(reg, "mono_bad_total", None),
        n_draws_all=getattr(reg, "mono_draws_total", None),
        tstar_med=tstar_med, theta=theta,
        y_err=y_err, x_err=x_err,
        history=hist if record_history else None,
    )
    if not light:
        rec.update(t=t.tolist(), mu=mu.tolist(), sd=sd.tolist(),
                   dmu=dmu.tolist(), dsd=dsd.tolist(), yt=yt.tolist())
    return rec


def _run_job(job):
    """One independent study run, rebuilt from a picklable spec.

    job = (kind, keys, family, builder_kwargs, seed, method, record_history,
           sigma_x, sigma_y)
    """
    kind, keys, family, kwargs, seed, method, record_history, sx, sy = job
    f = ISO.make_isotherm(family, **kwargs)
    # Only the uncertainty arm needs the full 400-point curve arrays (for the
    # recovery, calibration, and per-level figures); baselines stay light.
    light = (method != "uncertainty")
    return job, _fit_record(f, seed, method, record_history=record_history,
                            sigma_x=sx, sigma_y=sy, light=light)


def _pool_run(indexed_job):
    """poolutil worker: (index, job) -> (index, record).  Keyed by index
    because the job tuple carries an unhashable kwargs dict."""
    idx, j = indexed_job
    return idx, _run_job(j)[1]


def _job_list():
    jobs = []
    # 1) Acquisition comparison: MC over seeds for EVERY isotherm, with
    #    history, under the standard noise model.  The uncertainty arm also
    #    serves as the recovery study and the per-level error analysis.
    for name, (fam, kw) in ISO.STANDARD_SPECS.items():
        for method in METHODS:
            for seed in MC_SEEDS:
                jobs.append(("mc", (name, method, seed), fam, kw, seed,
                             method, True, ISO.SIGMA_X, ISO.SIGMA_Y))
    # 2) Noise-free regime sweeps: one fit per regime variant, sigma = 0,
    #    isolating the approximation error of the basis.
    for fam in ISO._FAMILIES:
        for lab, kw in ISO.family_variant_specs(fam).items():
            jobs.append(("regimes", (fam, lab), fam, kw, MC_SEEDS[0],
                         "uncertainty", False, 0.0, 0.0))
    return jobs


def _store(res, job, rec):
    kind, keys = job[0], job[1]
    if kind == "mc":
        name, method, seed = keys
        res["mc"][name][method][MC_SEEDS.index(seed)] = rec
    else:
        fam, lab = keys
        res["regimes"][fam][lab] = rec


def run_experiments(fresh=False, jobs=None):
    """Run (or load) all experiments.  Cache keyed on seed list + budget.

    jobs : worker processes for a fresh run (None = all but one core).  Every
    run is independently seeded, so the result does not depend on the worker
    count or completion order.
    """
    tag = {"seeds": MC_SEEDS, "n_init": N_INIT, "n_iters": N_ITERS,
           "methods": METHODS, "metric": "saturation_error",
           "design": "per-isotherm-mc + noise-free-regimes",
           "levels": list(LEVELS5)}
    if not fresh and os.path.exists(CACHE):
        with open(CACHE) as fh:
            cached = json.load(fh)
        if cached.get("tag") == tag:
            return cached
        print("Config changed, re-running experiments.", flush=True)

    res = {"tag": tag,
           "mc": {nm: {m: [None] * len(MC_SEEDS) for m in METHODS}
                  for nm in ISO.STANDARD_SPECS},
           "regimes": {f: {} for f in ISO._FAMILIES}}

    job_list = _job_list()
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 1)
        # Each worker hosts a full JAX runtime whose peak commit charge is
        # roughly 1.5 GB during compilation; an unrestricted worker count can
        # exhaust the Windows page file (LLVM "unable to allocate section
        # memory"), so bound the default by available memory.
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            jobs = max(1, min(jobs, int(avail_gb / 1.5)))
        except ImportError:
            jobs = min(jobs, 4)
    jobs = min(jobs, len(job_list))

    if jobs > 1:
        # One compute thread per worker (the JAX FAQ recipe): with N workers
        # on one machine the per-process thread pools only fight each other,
        # and their stacks inflate the commit charge.  Children inherit the
        # parent's environment, so set it before the pool spawns; numpyro has
        # already put --xla_force_host_platform_device_count into XLA_FLAGS
        # at import time, so append rather than overwrite.
        poolutil.pin_threads()
        # Route through the recycling pool (poolutil): at the 0.001 s
        # acquisition resolution each worker's JAX cache and 5000-point
        # candidate arrays inflate the commit charge, and a non-recycled pool
        # can exhaust the Windows page file over a thousand runs.  A fresh pool
        # every few dozen tasks bounds the peak.  Jobs carry an unhashable
        # kwargs dict, so we key the pool by list index.
        indexed = list(enumerate(job_list))
        recs = poolutil.run_pool(_pool_run, indexed, jobs, tasks_per_child=8,
                                 label="main-campaign runs")
        for idx, rec in recs.items():
            _store(res, job_list[idx], rec)
    else:
        for n, j in enumerate(job_list, 1):
            print(f"  [{n}/{len(job_list)}] {j[0]}/{'/'.join(map(str, j[1]))} ...",
                  flush=True)
            job, rec = _run_job(j)
            _store(res, job, rec)

    # the worker pool fills the dicts in completion order; restore the canonical
    # order, which downstream consumers (figure panels, colors) rely on.
    res["regimes"] = {fam: {lab: res["regimes"][fam][lab]
                            for lab in ISO.family_variant_specs(fam)}
                      for fam in ISO._FAMILIES}

    with open(CACHE, "w") as fh:
        json.dump(res, fh)
    return res


# ---------------------------------------------------------------------------
# Spline building blocks (no MCMC)
# ---------------------------------------------------------------------------
def fig_basis():
    knots = M.make_knots(DOMAIN, n_interior=7, degree=3)
    t = np.exp(np.linspace(np.log(DOMAIN[0]), np.log(DOMAIN[1]), 2000))
    t = np.concatenate([[0.0], t])
    Mb = M.mspline_basis(t, knots, 3)
    Ib = M.ispline_basis(t, knots, 3)
    P    = Mb.shape[1]
    cols = plt.cm.viridis(np.linspace(0, 0.9, P))

    # Layout: (a) and (b) side by side on top, (c) stretched across the
    # bottom, so the individual blocks and the assembled curve get room.
    W = 6.5; s = _fonts(W)
    fig = plt.figure(figsize=(W, 5.2), layout="constrained")
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    ax = axA
    # Each hump has unit area, so peak heights span two orders of magnitude
    # (narrow near-origin humps are very tall); scale each to unit peak for
    # display so every hump is fully visible on one linear axis.
    for m in range(P):
        ax.plot(t, Mb[:, m] / Mb[:, m].max(), color=cols[m], lw=1.7)
    ax.set_title(r"(a) M-splines (rate blocks)", fontweight="bold")
    ax.set_xlabel("Input $x$ (s, log)")
    ax.set_ylabel(r"$M_m(x)/\max M_m$")
    ax.set_xscale("log"); ax.set_xlim(DOMAIN[0], DOMAIN[1]); ax.set_ylim(0, 1.05)

    ax = axB
    for m in range(P):
        ax.plot(t, Ib[:, m], color=cols[m], lw=1.7)
    ax.set_title(r"(b) I-splines (curve blocks)", fontweight="bold")
    ax.set_xlabel("Input $x$ (s, log)")
    ax.set_ylabel(r"$I_m(x)=\int_0^x\! M_m$")
    ax.set_xscale("log"); ax.set_xlim(DOMAIN[0], DOMAIN[1]); ax.set_ylim(-0.03, 1.05)
    # A log axis cannot display x = 0, which is exactly where the boundary
    # knots sit and where every I-spline vanishes.  Inset the origin region on
    # a LINEAR axis so the f(0)=0 construction is visible rather than implied.
    axIn = ax.inset_axes([0.10, 0.52, 0.44, 0.44])
    tin = np.linspace(0.0, 0.02, 400)
    Iin = M.ispline_basis(tin, knots, 3)
    for m in range(min(4, P)):
        axIn.plot(tin, Iin[:, m], color=cols[m], lw=1.3)
    axIn.axvline(0.0, color="#444", lw=0.8, ls="--")
    axIn.set_xlim(0, 0.02); axIn.set_ylim(-0.03, 1.05)
    axIn.set_title("linear axis at origin", fontsize=6.5 * s, pad=2)
    axIn.tick_params(labelsize=5.5 * s)
    axIn.set_facecolor("white"); axIn.grid(alpha=0.2)
    axIn.annotate(r"$I_m(0)\!=\!0$", xy=(0.0, 0.0), xytext=(0.004, 0.42),
                  fontsize=6 * s, color="#333")

    ax = axC
    # Weights chosen by non-negative least squares so the assembled sum
    # reproduces the headline Langmuir curve: the example is then a real
    # saturation curve on the real timescale, not an arbitrary step.
    from scipy.optimize import nnls
    w, _ = nnls(Ib, ISO.langmuir()(t))
    f = Ib @ w
    for m in range(P):
        ax.plot(t, w[m] * Ib[:, m], color=cols[m], lw=1.0, alpha=0.5)
    ax.plot(t, f, "k-", lw=2.6, label=r"$f=\sum_m w_m I_m$")
    ax.set_title("(c) Weighted sum = a curve", fontweight="bold")
    ax.set_xlabel("Input $x$ (s)"); ax.set_ylabel("Saturation")
    ax.set_xlim(0, DOMAIN[1]); ax.set_ylim(-0.03, 1.1); ax.legend(loc="lower right")

    for ax in (axA, axB, axC):
        ax.set_facecolor("white"); ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_basis")


# ---------------------------------------------------------------------------
# Recovery figures: the Sips fit and the five family figures (median-error seed)
# ---------------------------------------------------------------------------
def median_seed_idx(records):
    """Index of the seed whose saturation error is the median of the arm.

    The anatomy panels showed seed 0 throughout, which happens to be more
    favorable than the median for four of the five families (by >1 s.d. for
    Sips and Elovich).  Showing the median-error seed makes the illustrated
    fit representative rather than flattering.
    """
    ge = np.array([r["sat_err"] for r in records])
    return int(np.argsort(ge)[len(ge) // 2])


def fig_final_fit(res):
    rs_sips = res["mc"]["Sips (n=2)"]["uncertainty"]
    d = rs_sips[median_seed_idx(rs_sips)]
    t   = np.array(d["t"])
    mu, sd   = np.array(d["mu"]), np.array(d["sd"])
    dmu, dsd = np.array(d["dmu"]), np.array(d["dsd"])
    yt  = np.array(d["yt"])

    s = _fonts(FIG_W)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.1]},
                                   layout="constrained")
    fig.patch.set_facecolor("white")
    xlo = 1e-3
    for ax in (ax1, ax2):
        ax.set_facecolor("white"); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        # log input axis (matches the log-spaced knots) with the sub-window
        # region x < x_min shaded: no data can constrain the fit there.
        ax.set_xscale("log")
        ax.axvspan(xlo, ISO.DOMAIN[0], color="0.90", zorder=0, lw=0)
    ax1.text(np.sqrt(xlo * ISO.DOMAIN[0]), 1.02, r"$x<x_{\min}$",
             fontsize=7 * s, color="#777", ha="center", va="bottom")
    ax1.plot(t, yt, "k-", lw=2.2, label="True curve")
    ax1.plot(t, mu, "C0-", lw=1.9, label="Posterior mean")
    ax1.fill_between(t, mu - 1.96 * sd, mu + 1.96 * sd, alpha=0.20, color="C0",
                     label="95% credible band")
    ax1.scatter(d["t_obs"], d["y_obs"], c="C3", s=40, edgecolor="black",
                lw=0.6, zorder=5, label="Observations")
    ax1.axvline(ISO.X_SAT, color="#999", ls=":", lw=1.0)
    # onset overshoot: the smooth basis rounds the cooperative delay upward, so
    # the posterior mean sits above the truth through the early rise.
    om = (t >= 0.02) & (t <= 0.45)
    if om.any():
        k = np.argmax((mu - yt)[om])
        xk, yk = t[om][k], mu[om][k]
        ax1.annotate("onset overshoot", xy=(xk, yk), xytext=(0.045, 0.60),
                     textcoords="axes fraction", fontsize=7.5 * s, color="#555",
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color="#777", lw=0.9))
    ax1.set_ylim(-0.04, 1.12); ax1.set_ylabel("Saturation")
    ax1.set_title(f"Monotone I-spline posterior (Sips, $N$={len(d['t_obs'])})",
                  fontweight="bold", fontsize=9.5 * s)
    ax1.legend(loc="lower right", framealpha=0.95)
    ax2.plot(t, dmu, "C1-", lw=1.9)
    ax2.fill_between(t, np.maximum(dmu - 1.96 * dsd, 0), dmu + 1.96 * dsd,
                     alpha=0.20, color="C1")
    ax2.axhline(0, color="#888", ls=":", lw=0.9)
    ax2.set_ylabel(r"$f'(x)$"); ax2.set_xlabel("Input $x$ (s, log)")
    ax2.set_xlim(xlo, DOMAIN[1])
    # Show the full posterior-mean rate, including the narrow near-origin
    # feature in the innermost knot interval: no data lie below x_min, so the
    # rate there is only weakly identified (its credible band is very wide).
    # The previous limit clipped that feature off the top of the panel.
    ax2.set_ylim(-0.2, max(2.0, float(dmu.max()) * 1.18))
    ax2.annotate(r"$f'\!\geq\!0$ everywhere; $f'\!\to\!0$ as $x$ grows",
                 xy=(0.97, 0.86), xycoords="axes fraction", ha="right",
                 fontsize=8 * s, color="C1")
    if float(dmu[0]) > 1.5 * float(np.median(dmu)):
        ytop = ax2.get_ylim()[1]
        ax2.annotate("rate weakly identified\nbelow $x_{\\min}$ (no data)",
                     xy=(np.sqrt(xlo * ISO.DOMAIN[0]), min(float(dmu[0]), 0.9 * ytop)),
                     xytext=(0.30, 0.72),
                     textcoords="axes fraction", fontsize=7 * s, color="#666",
                     ha="left", arrowprops=dict(arrowstyle="->", color="#888",
                                                lw=0.8))
    _save(fig, "fig_final_fit")


# ---------------------------------------------------------------------------
# Per-isotherm detail figures: fit + noise-free regimes + learning + levels
# ---------------------------------------------------------------------------
ISO_FAMILY = {"Langmuir": "Langmuir", "Dissociative": "Dissociative",
              "Sips (n=2)": "Sips", "Elovich": "Elovich", "D-R KWW": "DR-KWW"}
ISO_SLUG = {"Langmuir": "langmuir", "Dissociative": "dissociative",
            "Sips (n=2)": "sips", "Elovich": "elovich", "D-R KWW": "kww"}
REGIME_SYM = {"tau": r"$\tau$", "K": "$K$", "n": "$n$",
              "g": r"$\gamma$", "b": r"$\beta$"}


def fig_isotherm_detail(res, nm):
    """One four-panel figure for isotherm nm: (a) noisy fit (median-error seed),
    (b) noise-free regime sweep of its family, (c) learning curves of the four
    designs and the GP-log arm over the MC seeds, (d) per-level signed errors."""
    fam = ISO_FAMILY[nm]
    rs  = res["mc"][nm]["uncertainty"]
    W = 7.0; s = _fonts(W)
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.8), layout="constrained")
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white"); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axA, axB, axC, axD = axes.ravel()

    # (a) fit under the standard noise model, MEDIAN-error seed (not seed 0,
    # which is more favorable than the median for four of the five families)
    d = rs[median_seed_idx(rs)]
    ge = np.array([r["sat_err"] for r in rs])
    t, mu, sd = np.array(d["t"]), np.array(d["mu"]), np.array(d["sd"])
    axA.plot(t, d["yt"], "k-", lw=2.0, label="True curve")
    axA.plot(t, mu, "C0-", lw=1.6, label="Posterior mean")
    axA.fill_between(t, mu - 1.96 * sd, mu + 1.96 * sd, alpha=0.20, color="C0",
                     label="95% band")
    axA.scatter(d["t_obs"], d["y_obs"], c="C3", s=18, edgecolor="black",
                lw=0.4, zorder=5, label="Observations")
    axA.axvline(ISO.X_SAT, color="#999", ls=":", lw=0.9)
    axA.set_xlim(0, DOMAIN[1]); axA.set_ylim(-0.08, 1.12)
    axA.set_title("(a) Fit under noise", fontweight="bold")
    axA.set_xlabel("Input $x$ (s)"); axA.set_ylabel("Saturation")
    axA.text(0.97, 0.06, f"median seed: {d['sat_err']:.3f}\n"
             f"50 seeds: {ge.mean():.3f} $\\pm$ {ge.std():.3f}",
             transform=axA.transAxes, fontsize=7.5 * s, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.5))
    axA.legend(loc="center right", fontsize=7 * s, framealpha=0.9)

    # (b) noise-free regime sweep of the family.  The true curve is drawn as
    # a thick translucent "halo" with the opaque dashed fit on top: where the
    # two coincide the dashed line rides inside the halo and both stay
    # visible, instead of the fit vanishing under an identical solid line.
    variants = list(res["regimes"][fam].items())
    cols = plt.cm.plasma(np.linspace(0.1, 0.8, len(variants)))
    for (lab, dv), c in zip(variants, cols):
        tv = np.array(dv["t"])
        muv, sdv = np.array(dv["mu"]), np.array(dv["sd"])
        axB.plot(tv, dv["yt"], "-", color=c, lw=4.0, alpha=0.30,
                 solid_capstyle="round")
        axB.plot(tv, muv, "--", color=c, lw=1.4)
        axB.fill_between(tv, muv - 1.96 * sdv, muv + 1.96 * sdv, color=c, alpha=0.10)
        key, val = lab.split("=")
        axB.plot([], [], "-", color=c, lw=2,
                 label=f"{REGIME_SYM.get(key, key)} = {float(val):.2g}"
                       f" ({dv['sat_err']:.3f})")
    axB.axvline(ISO.X_SAT, color="#bbb", ls=":", lw=0.8)
    axB.set_xlim(0, DOMAIN[1]); axB.set_ylim(-0.08, 1.12)
    axB.set_title("(b) Noise-free regimes", fontweight="bold")
    axB.set_xlabel("Input $x$ (s)"); axB.set_ylabel("Saturation")
    axB.legend(loc="lower right", fontsize=7 * s, framealpha=0.9,
               handlelength=1.2)

    # (c) learning curves: active vs three baselines over the MC seeds.  The
    # active arm and the two log-aware baselines carry shaded +/-1 s.d. bands;
    # the weak uniform-random baseline is shown as a thin dotted reference
    # line (no band) so it does not clutter the panel.
    STYLE = {"uncertainty": ("C0", "-", True),
             "random_log":  ("C1", "-", True),
             "grid":        ("C2", "-", True),
             "random":      ("0.55", ":", False)}
    for method in ["random", "grid", "random_log", "uncertainty"]:
        color, ls, band = STYLE[method]
        H = np.array([r["history"] for r in res["mc"][nm][method]])
        N = np.arange(N_INIT, N_INIT + H.shape[1])
        m, sdh = H.mean(0), H.std(0)
        axC.plot(N, m, ls, color=color, lw=1.8 if band else 1.2,
                 label=METHOD_LABEL[method])
        if band:
            axC.fill_between(N, np.maximum(m - sdh, 1e-4), m + sdh,
                             color=color, alpha=0.13)
    # The unconstrained surrogate at every budget, not just at N=20.  Plotted
    # for the resolution-matched arm (GP on the log-warped axis), which is the
    # honest comparison: the linear-axis GP arm is far worse and would
    # flatter the constrained surrogate here.
    GA = _gparms()
    if GA is not None and nm in GA.get("gp_log", {}):
        Hg = np.array([r["hist"] for r in GA["gp_log"][nm]])
        Ng = np.arange(N_INIT, N_INIT + Hg.shape[1])
        axC.plot(Ng, Hg.mean(0), "--", color="C4", lw=1.5,
                 label="unconstrained GP (log axis)")
    axC.axhline(ISO.SIGMA_Y, color="C3", ls="--", lw=1.0,
                label=fr"output noise $\sigma_y={ISO.SIGMA_Y}$")
    axC.set_yscale("log")
    # A narrow log range shows only the single decade tick (0.1) by default,
    # which is hard to read; force explicit decimal ticks and a common range
    # so all five isotherm panels are directly comparable.
    from matplotlib.ticker import NullLocator
    axC.set_ylim(0.015, 0.3)
    axC.set_yticks([0.02, 0.05, 0.1, 0.2])
    axC.set_yticklabels(["0.02", "0.05", "0.10", "0.20"])
    axC.yaxis.set_minor_locator(NullLocator())
    axC.set_title(f"(c) Learning curves ({len(MC_SEEDS)} seeds)",
                  fontweight="bold")
    axC.set_xlabel("Number of observations $N$")
    axC.set_ylabel("Saturation error across levels")
    axC.legend(loc="upper right", fontsize=6 * s, framealpha=0.9)

    # (d) per-level signed errors at N=20 (uncertainty arm)
    lev = [int(round(a * 100)) for a in LEVELS5]
    ym = np.array([np.mean([r["y_err"][str(l)] for r in rs]) for l in lev])
    ys = np.array([np.std([r["y_err"][str(l)] for r in rs]) for l in lev])
    # The input error is reported RELATIVE to the true crossing time.  In
    # absolute seconds the 95% value sets the axis range and renders the four
    # lower levels indistinguishable from zero; the families' crossings also
    # span very different time scales, so a fraction is the meaningful common unit.
    fam_kw = ISO.STANDARD_SPECS[nm]
    f_true = ISO.make_isotherm(fam_kw[0], **fam_kw[1])
    x_true = np.array([ISO.true_level_time(f_true, a * ISO.A_DEFAULT)
                       for a in LEVELS5])
    xm = np.array([np.mean([r["x_err"][str(l)] for r in rs]) for l in lev]) / x_true
    xs = np.array([np.std([r["x_err"][str(l)] for r in rs]) for l in lev]) / x_true
    off = 1.5
    axD.errorbar(np.array(lev) - off, ym, ys, fmt="o", color="C0", capsize=3,
                 lw=1.3, label=r"$\Delta y$ (left)")
    axD.axhline(0, color="C0", ls=":", lw=0.8, alpha=0.7)
    axD.set_ylabel(r"Response error $\Delta y(\alpha)$  ($A=1$)", color="C0")
    axD.tick_params(axis="y", labelcolor="C0")
    # Common axis limits across all five family figures so the panels
    # are directly comparable; both quantities fit these bounds except the one
    # sub-window KWW point, which is annotated off-scale below.
    axD.set_ylim(-0.13, 0.09)
    axD2 = axD.twinx()
    axD2.errorbar(np.array(lev) + off, xm, xs, fmt="s", color="C1", capsize=3,
                  lw=1.3, label=r"$\Delta x/x_\alpha$ (right)")
    axD2.axhline(0, color="C1", ls=":", lw=0.8, alpha=0.7)
    axD2.set_ylabel(r"Rel. input error $\Delta x/x_\alpha$", color="C1")
    axD2.tick_params(axis="y", labelcolor="C1")
    axD2.set_ylim(-1.05, 1.05)
    for l, m in zip(lev, xm):          # flag any point clipped by the common axis
        if m > 1.05:
            axD2.annotate(fr"$+{m:.1f}\!\uparrow$", xy=(l + off, 1.02),
                          ha="center", va="top", fontsize=6.2 * s, color="C1")
        elif m < -1.05:
            axD2.annotate(fr"${m:.1f}\!\downarrow$", xy=(l + off, -1.02),
                          ha="center", va="bottom", fontsize=6.2 * s, color="C1")
    axD2.spines[["top"]].set_visible(False)
    axD.set_title("(d) Level-wise errors at $N=20$", fontweight="bold")
    axD.set_xlabel(r"Saturation level $\alpha$ (%)")
    axD.set_xticks(lev)
    h1, l1 = axD.get_legend_handles_labels()
    h2, l2 = axD2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=7 * s,
               framealpha=0.9)

    fig.suptitle(DISPLAY[nm], fontweight="bold")
    _save(fig, f"fig_{ISO_SLUG[nm]}")


def fig_details(res):
    for nm in ISO.STANDARD_SPECS:
        fig_isotherm_detail(res, nm)


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(_FIG_DIR, exist_ok=True)
    # Figures use constrained_layout, so no tight_layout here.
    # 600 dpi, print quality for combination art.
    fig.savefig(os.path.join(_FIG_DIR, f"{name}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png  ->  {_FIG_DIR}")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def _axes(fig):
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)


# ===========================================================================
# Supporting figures (each loads its own cache; grouped here so all figures
# live in one module). fig_knots reuses the knot helpers from design_choices.
# ===========================================================================

def fig_readout():
    """The read-out figure (results_readout.json, results.json): amplification of
    the inversion, the conditioning check, and the read-out quantile family."""
    R = _load("results_readout.json")
    W = 7.0
    s = _fonts(W)
    # Manual layout: three axes of IDENTICAL size and vertical extent, titles
    # drawn at one common height above them (constrained layout gave each panel
    # its own height -- (b)'s equal-aspect box shrank and (c)'s secondary top
    # axis pushed its title up -- so the titles never lined up).
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.05))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.72,
                        wspace=0.45)
    fig.patch.set_facecolor("white")
    axA, axB, axC = axes
    for ax, ttl in zip(axes, (r"(a) Amplification $1/f'(x_\alpha)$",
                              "(b) Conditioning check",
                              "(c) The read-out family")):
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, 0.965, ttl, ha="center", va="top",
                 fontweight="bold", fontsize=8.8 * s)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(FAMILIES)))

    # (a) amplification 1/f'(x) of the true curves, level by level
    alphas = np.arange(0.05, 0.96, 0.05)
    for nm, c in zip(FAMILIES, cols):
        fam, kw = ISO.STANDARD_SPECS[nm]
        f = ISO.make_isotherm(fam, **kw)
        amp = []
        for a in alphas:
            xa = ISO.true_level_time(f, a)
            h = 1e-6
            d = (float(f(np.array([xa + h]))[0])
                 - float(f(np.array([max(xa - h, 0.0)]))[0])) / (2 * h)
            amp.append(1.0 / d if d > 0 else np.nan)
        axA.plot(100 * alphas, amp, "-", color=c, lw=1.6, label=DISPLAY[nm])
    axA.set_yscale("log")
    axA.set_xlabel(r"Saturation level $\alpha$ (%)")
    axA.set_ylabel("s per unit response")
    axA.legend(loc="upper left", fontsize=6.8 * s, framealpha=0.9,
               handlelength=1.1, labelspacing=0.25, borderpad=0.35,
               handletextpad=0.5)

    # (b) conditioning: the plateau-corrected relation, predicted vs observed input shift
    #     at 95%, one point per family.  dx ~ (alpha*dA - dy)/mu', with dy the
    #     95% response misfit and dA the plateau misestimate (results.json), 1/mu'
    #     the posterior amplification (results_readout).
    MAIN = _load("results.json")
    dy, dA, amp, obs = [], [], [], []
    for nm in FAMILIES:
        recs = MAIN["mc"][nm]["uncertainty"]
        dy.append(np.mean([r["y_err"]["95"] for r in recs]))
        dA.append(np.mean([r["f_max"] for r in recs]) - 1.0)
        amp.append(float(np.median([r["amp_post"] for r in R[nm]])))
        obs.append(np.mean([r["x_sat_med"] - r["x_true"] for r in R[nm]]))
    dy, dA, amp, obs = (np.array(v) for v in (dy, dA, amp, obs))
    pred = (0.95 * dA - dy) * amp
    lim = 1.2 * max(np.abs(np.concatenate([pred, obs])).max(), 0.6)
    axB.axhline(0, color="0.85", lw=0.8, zorder=0)
    axB.axvline(0, color="0.85", lw=0.8, zorder=0)
    axB.plot([-lim, lim], [-lim, lim], "0.55", ls="--", lw=1.0, zorder=1)
    labs = [DISPLAY[n].replace("Michaelis-Menten", "MM").replace(" ($n=2$)", "")
            for n in FAMILIES]
    # hand-placed offsets so no label overlaps a neighboring point or label
    # (Langmuir/KWW and Sips/MM are close pairs: label on opposite sides)
    offs = {"Langmuir": (8, -12, "left"), "MM": (-8, -11, "right"),
            "Sips": (8, -4, "left"), "Elovich": (8, -4, "left"),
            "KWW": (-8, 3, "right")}
    for i, (c, lab) in enumerate(zip(cols, labs)):
        axB.scatter(pred[i], obs[i], color=c, s=46, edgecolor="black",
                    lw=0.5, zorder=5)
        ox, oy, ha = offs.get(lab, (7, 4, "left"))   # label offsets (points)
        axB.annotate(lab, (pred[i], obs[i]), fontsize=7.0 * s, ha=ha,
                     textcoords="offset points", xytext=(ox, oy))
    axB.set_xlim(-lim, lim); axB.set_ylim(-lim, lim)
    # equal SCALE without shrinking the axes box (datalim widens the limits
    # instead), so this panel keeps the same drawn size as (a) and (c)
    axB.set_aspect("equal", adjustable="datalim")
    axB.set_xlabel(r"predicted $\Delta x(95\%)$ (s)")
    axB.set_ylabel(r"observed $\Delta x(95\%)$ (s)")

    # (c) the read-out family: q-th quantile of the crossing posterior vs q
    if "q" in R[FAMILIES[0]][0]:
        qs = np.array(R[FAMILIES[0]][0]["q"], float)
        for nm, c in zip(FAMILIES, cols):
            xq = np.array([r["x_q"] for r in R[nm]], float).mean(0)
            axC.plot(qs, xq, "o-", color=c, lw=1.5, ms=3)
        axC.axhline(ISO.X_SAT, color="k", ls="-", lw=1.0)
        axC.text(0.995, ISO.X_SAT * 1.03, r"true $x_{\rm sat}$", fontsize=7.2 * s,
                 ha="right", va="bottom")
        y0, y1 = axC.get_ylim()
        # q=0.5 marker labelled in the empty top-left corner, q=0.95 in the
        # empty bottom-right one, clear of the curves and of each other
        for qv, lab, ha, y, va in ((0.5, "median", "left", y1, "top"),
                                   (0.95, "one-sided", "right", y0, "bottom")):
            axC.axvline(qv, color="0.6", ls=":", lw=0.8)
            axC.text(qv + (0.01 if ha == "left" else -0.01), y,
                     f"$q={qv}$\n({lab})",
                     fontsize=6.8 * s, color="0.35", va=va, ha=ha)
        axC.set_xlabel(r"Read-out quantile $q$")
        axC.set_ylabel(r"Recommended input $\hat{x}_q$ (s)")
        secax = axC.secondary_xaxis(
            "top", functions=(lambda q: q / np.clip(1 - q, 1e-3, None),
                              lambda r: r / (1 + r)))
        secax.set_xticks([1, 2, 4, 9, 19])
        secax.set_xlabel(r"cost ratio $q/(1-q)$", fontsize=7.2 * s, labelpad=3)
        secax.tick_params(labelsize=7.0 * s)

    _axes(fig)
    _save(fig, "fig_readout")


def fig_knots():
    """Best monotone approximation vs knot geometry (prior/sampler/seed-free
    evidence for log knot placement). Reuses design_choices' knot helpers."""
    import design_choices as DC
    W = 6.5
    s = _fonts(W)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 2.6), layout="constrained")
    fig.patch.set_facecolor("white")

    # The plotted subset of design_choices.KNOT_CURVES (cache key -> label).
    # Values come from results_knots.json, so the figure and the recompute_stats
    # numbers cannot diverge; a live recompute happens only if the cache is absent.
    import json as _json
    SHOW = {"MM K=0.03 (steep)": "MM $K$=0.03 (steep)",
            "KWW beta=0.35 (fast)": "KWW $\\beta$=0.35",
            "Langmuir tau=0.15": "Langmuir $\\tau$=0.15",
            "Langmuir (headline)": "Langmuir (headline)",
            "Sips n=2 (headline)": "Sips $n$=2 (headline)",
            "Sips n=4 (late step)": "Sips $n$=4 (late step)"}
    missing = [k for k in SHOW if k not in DC.KNOT_CURVES]
    assert not missing, f"fig_knots: not in design_choices.KNOT_CURVES: {missing}"
    cfgs = list(DC.KNOT_CONFIGS.items())
    curves = {SHOW[k]: DC.KNOT_CURVES[k] for k in SHOW}
    _cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "results_knots.json")
    try:
        with open(_cache) as fh:
            K = _json.load(fh)
        vals = np.array([[K[k][cfg] for cfg, _ in cfgs] for k in SHOW])
    except (OSError, ValueError, KeyError):
        print("  fig_knots: results_knots.json absent or stale, recomputing "
              "(run design_choices.py knots to cache it)")
        vals = np.array([[DC.best_approx_err(DC.KNOT_CURVES[k], kf(DOMAIN, ni))
                          for _, (kf, ni) in cfgs] for k in SHOW])

    # horizontal dot plot: curves on the y-axis (legible labels), error on x
    y = np.arange(len(curves))
    cols = ["C0", "C1", "C2", "C3"]
    markers = ["o", "s", "^", "D"]
    FLOOR = 1e-4                              # explicit plotting floor
    for j, ((lab, _), c, mk) in enumerate(zip(cfgs, cols, markers)):
        axA.plot(np.maximum(vals[:, j], FLOOR), y, marker=mk, ls="none",
                 color=c, ms=6, label=lab)
    axA.set_xscale("log")
    axA.set_yticks(y)
    axA.set_yticklabels(list(curves), fontsize=6.6 * s)
    axA.set_xlabel("Best monotone approximation error")
    axA.axvline(ISO.SIGMA_Y, color="k", ls=":", lw=0.9)
    axA.text(ISO.SIGMA_Y * 1.15, -0.35, r"$\sigma_y$", fontsize=7 * s,
             ha="left", va="center")
    axA.axvline(FLOOR, color="0.7", ls=":", lw=0.8)
    axA.text(FLOOR, -0.72, r"$\leq10^{-4}$", fontsize=5.2 * s, color="0.5",
             ha="center", va="bottom")
    axA.set_xlim(6.5e-5, 2.2)
    axA.set_ylim(len(curves) - 0.4, -0.75)   # inverted, headroom for sigma_y
    axA.set_title("(a) Representational capacity",
                  fontweight="bold", fontsize=9.0 * s)
    handlesA, labelsA = axA.get_legend_handles_labels()
    fig.legend(handlesA, labelsA, loc="upper center", bbox_to_anchor=(0.5, 0.04),
               ncol=4, frameon=False, fontsize=6.0 * s, handlelength=1.0,
               columnspacing=1.4)
    axA.grid(alpha=0.25, axis="x")

    # (b) where the knots actually sit
    for j, ((lab, (kf, ni)), c) in enumerate(zip(cfgs, cols)):
        kn = kf(DOMAIN, ni)
        interior = kn[4:-4]
        axB.plot(np.maximum(interior, 1e-3), np.full(len(interior), 3 - j),
                 "|", color=c, ms=11, mew=1.6)
        axB.text(6.0, 3 - j, lab, fontsize=6.6 * s, va="center")
    zz = np.geomspace(1e-3, 5, 300)
    f = ISO.make_isotherm("Sips", n=2.0)
    axB.plot(zz, -0.6 + 1.2 * f(zz), "-", color="0.4", lw=1.3)
    axB.text(0.032, -0.36, "Sips $n$=2", fontsize=6.6 * s, color="0.35", ha="left")
    axB.axvline(2.0, color="#d1495b", ls="--", lw=1.1)
    axB.text(2.15, 3.5, r"$x_{\rm sat}=2$ s", fontsize=6.4 * s, color="#d1495b",
             ha="left", va="top")
    axB.set_xscale("log")
    axB.set_xlim(1e-3, 22)
    axB.set_ylim(-0.9, 3.6)
    axB.set_yticks([])
    axB.set_xlabel("Input $x$ (s)")
    axB.set_title("(b) Interior knot placement", fontweight="bold",
                  fontsize=9.0 * s)
    axB.spines[["left"]].set_visible(False)

    _axes(fig)
    axB.grid(alpha=0.25, axis="x")
    _save(fig, "fig_knots")


def fig_noise():
    """Every arm of the GP ablation other than GP-linear, as its pooled saturation error
    divided by the penalized I-spline's, against output noise on a LINEAR axis.
    The two GP ratios stay flat; both unpenalized ratios climb with the noise, and
    the unpenalized P=11 line crosses above GP-log's near sigma_y = 0.08 (the
    reversal of the bare constraint). results_noise + results_unpen25 +
    results_unpen11 + results_gppin_noise."""
    R = _load("results_noise.json")
    B = _load("results_unpen11.json")
    U = _load("results_unpen25.json")
    P = _load("results_gppin_noise.json")
    sig = [0.05, 0.1, 0.15]                 # levels where all arms exist
    # Drawn at the shared reference width (like every other figure) so that, once
    # included at \textwidth, marker and line sizes render 1:1 and match the rest.
    W = FIG_W
    s = _fonts(W)
    fig, ax = plt.subplots(figsize=(W, 3.5), layout="constrained")
    fig.patch.set_facecolor("white")

    def mof(getter):
        return np.mean([getter(nm) for nm in FAMILIES])

    def gp(sy):
        return mof(lambda nm: np.mean([r["sat_err"] for r in R[f"gp_log|{nm}|{sy}"]]))

    def gpin(sy):
        return mof(lambda nm: np.mean([r["sat_err"] for r in P[f"gp_log_pin|{nm}|{sy}"]]))

    def pen(sy):
        return mof(lambda nm: np.mean([r["sat_err"] for r in R[f"ispline|{nm}|{sy}"]]))

    def unp(sy):
        return mof(lambda nm: np.mean([r["sat_err"] for r in B[f"{nm}|{sy}"]]))

    def unp25(sy):
        return mof(lambda nm: np.mean([r["sat_err"] for r in U[f"{nm}|{sy}"]]))

    arms = [("GP-log", lambda sy: gp(sy) / pen(sy), "0.0", "o", 6.0, 2.0),
            ("GP-log-pin", lambda sy: gpin(sy) / pen(sy), "C0", "D", 5.5, 1.8),
            (r"unpenalized $P{=}25$", lambda sy: unp25(sy) / pen(sy), "C3", "s", 5.5, 1.8),
            (r"unpenalized $P{=}11$", lambda sy: unp(sy) / pen(sy), "C1", "^", 6.0, 1.8)]
    for lab, f, col, mk, ms, lw in arms:
        ax.plot(sig, [f(sy) for sy in sig], color=col, lw=lw, marker=mk, ms=ms,
                zorder=5, label=lab)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel(r"Output noise $\sigma_y$")
    ax.set_ylabel("error ratio to the penalized I-spline")
    ax.set_xticks(sig)
    ax.set_xticklabels(["0.05", "0.10", "0.15"])
    ax.set_ylim(0.85, 2.15)
    ax.text(0.97, 0.05, r"penalized I-spline more accurate ($>1$)",
            transform=ax.transAxes, fontsize=7 * s, color="0.35", ha="right")
    ax.legend(loc="lower left", ncol=2, fontsize=7.5 * s, framealpha=0.9,
              handlelength=1.8, borderpad=0.5, labelspacing=0.35, columnspacing=1.2)
    ax.margins(x=0.10)

    _axes(fig)
    _save(fig, "fig_noise")


# The scorecard's colour ramp runs green (low error) to red (high error) with
# lightness falling monotonically along it, so the cells still rank by shade for a
# reader who cannot separate red from green; the numbers printed in every cell
# carry the values regardless.
SCORECARD_RAMP = ["#e3f4dc", "#a9dba0", "#f1c25f", "#ee9a45", "#e56a35", "#c8392b", "#961c1c", "#5f0d0d"]


def fig_scorecard():
    """Surrogate x design scorecard: pooled saturation error at N=20 (lighter is
    better) for each surrogate fitted to each kind of design.  Column 1 is each
    surrogate driving its own uncertainty-sampling loop; the other columns fit the
    GP arms to the designs the I-spline campaign stored (results_gp_matched.json,
    the matched-design control) and the unpenalized bases to the same designs
    (results_unpen_matched.json).  Cells whose combination was not run are greyed.
    Caches only, no MCMC."""
    R = _load("results.json")["mc"]
    G = _load("results_gparms.json")
    Q = _load("results_gp_matched.json")["matched"]
    U25 = _load("results_unpen25.json")
    U11 = _load("results_unpen11.json")

    def pooled(vals):
        return float(np.mean(vals))

    def isp(strat):
        return pooled([r["sat_err"] for nm in FAMILIES for r in R[nm][strat]])

    def own(arm):
        return pooled([r["sat_err"] for nm in FAMILIES for r in G[arm][nm]])

    def matched(arm, strat):
        return pooled([r["sat_err"] for nm in FAMILIES for r in Q[arm][strat][nm]])

    def unpen(U):
        return pooled([r["sat_err"] for nm in FAMILIES for r in U[f"{nm}|0.1"]])

    # the unpenalized bases refit on the stored designs (refits.py unpen); absent
    # until that campaign has run, in which case those cells stay grey
    try:
        UM = _load("results_unpen_matched.json")
    except (OSError, FileNotFoundError):
        UM = None

    def unpen_matched(arm, strat):
        if UM is None or arm not in UM:
            return np.nan
        return pooled([v for nm in FAMILIES for v in UM[arm][strat][nm]])

    strats = ["uncertainty", "grid", "random_log", "random"]
    rows = [
        ("penalized I-spline", [isp("uncertainty")] + [isp(s) for s in strats]),
        ("GP-log-pin", [own("gp_log_pin")] + [matched("gp_log_pin", s) for s in strats]),
        ("GP-log", [own("gp_log")] + [matched("gp_log", s) for s in strats]),
        ("GP-linear", [own("gp_lin")] + [matched("gp_lin", s) if "gp_lin" in Q else np.nan for s in strats]),
        (r"unpenalized $P{=}25$", [unpen(U25)] + [unpen_matched("unpen25", s) for s in strats]),
        (r"unpenalized $P{=}11$", [unpen(U11)] + [unpen_matched("unpen11", s) for s in strats]),
    ]
    cols = ["own\nuncertainty\nsampling", "I-spline's\nuncertainty\ndesigns",
            "log-spaced\ngrid", "log-uniform\nrandom", "uniform\nrandom"]
    A = np.array([v for _, v in rows], float)

    W = FIG_W
    s = _fonts(W)
    fig, ax = plt.subplots(figsize=(W, 3.6), layout="constrained")
    fig.patch.set_facecolor("white")
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("scorecard_greenred", SCORECARD_RAMP)
    norm = Normalize(vmin=0.035, vmax=0.10)
    masked = np.ma.masked_invalid(A)
    # cell polygons instead of a rasterized image: no large resampling buffer at
    # 600 dpi, and the white cell borders come for free; masked cells show the
    # grey axes background
    ax.set_facecolor("0.88")
    im = ax.pcolormesh(np.arange(A.shape[1] + 1) - 0.5, np.arange(A.shape[0] + 1) - 0.5,
                       masked, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.5)
    ax.set_xlim(-0.5, A.shape[1] - 0.5)
    ax.set_ylim(A.shape[0] - 0.5, -0.5)          # row 0 at the top, as in a table
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(A[i, j]):
                ax.text(j, i, "not run", ha="center", va="center", fontsize=6.5 * s, color="0.45")
            else:
                dark = norm(A[i, j]) > 0.5
                ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                        fontsize=8 * s, color="white" if dark else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7 * s)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r for r, _ in rows], fontsize=8 * s)
    ax.set_xlabel("design the surrogate is fitted to")
    ax.set_ylabel("surrogate")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"pooled saturation error at $N{=}20$ (lower is better)", fontsize=7.5 * s)
    cb.ax.tick_params(labelsize=7 * s)
    _save(fig, "fig_scorecard")


def fig_gp_compare(FAMILY="Sips (n=2)", SEED=4, N_SURVEY=5, N_ACTIVE=7):
    """Each surrogate drives its OWN active-learning loop from a shared
    survey, one seed: GP-linear, GP-log-pin, and the monotone I-spline, with the
    mean's level-crossings. The five survey points (inputs and observations) are
    identical for every arm; the active points are chosen by each surrogate's own
    uncertainty sampling and so differ between panels. Illustrative only; campaign
    numbers are in the text."""
    import gp_arms as GA
    N_SHOW, ALPHA = 25, 0.95
    fam, kw = ISO.STANDARD_SPECS[FAMILY]
    truth = ISO.make_isotherm(fam, **kw)
    observe = ISO.make_observe(truth, domain=DOMAIN)
    tg = M.eval_points(0.0, DOMAIN[1])

    # ONE seed's noise realization, shared by every arm: the same five-point
    # survey (inputs and observations) and the same post-survey noise stream, so
    # the only thing that differs between arms is the surrogate and its own
    # uncertainty-driven active selections (no cache, no per-arm reseeding).
    g0 = np.random.default_rng(SEED)
    sx = np.asarray(M.survey_design(N_SURVEY, DOMAIN, g0), float)
    sy = np.array([observe(float(x), g0) for x in sx])
    post_state = g0.bit_generator.state

    def obs_rng():                     # a generator at the shared post-survey state
        g = np.random.default_rng(SEED)
        g.bit_generator.state = post_state
        return g

    draw_rng = np.random.default_rng(SEED)          # for the shown posterior draws

    def gp_fit_cov(xd, yd, kind, pin):
        xx, yy = (np.concatenate([[0.0], xd]), np.concatenate([[0.0], yd])) \
            if pin else (np.asarray(xd, float), np.asarray(yd, float))
        uo, ug = GA.warp(xx, kind), GA.warp(tg, kind)
        pin_var = np.zeros(len(uo))
        if pin:
            pin_var[0] = 1.0
        best = (-np.inf, None)
        for ell in GA.ell_grid(kind):
            for sf in GA.SF_GRID:
                for sn in GA.SN_GRID:
                    nvar = np.where(pin_var > 0, 1e-8, sn**2)
                    lml = GA._lml(uo, yy, nvar, ell, sf)
                    if lml > best[0]:
                        best = (lml, (ell, sf, sn))
        ell, sf, sn = best[1]
        nvar = np.where(pin_var > 0, 1e-8, sn**2)
        K = GA._rbf(uo, uo, ell, sf) + np.diag(nvar) + 1e-10 * np.eye(len(uo))
        Ks, Kss = GA._rbf(ug, uo, ell, sf), GA._rbf(ug, ug, ell, sf)
        Kinv = np.linalg.solve(K, np.eye(len(uo)))
        ybar = yy.mean()
        return ybar + Ks @ (Kinv @ (yy - ybar)), Kss - Ks @ Kinv @ Ks.T

    def gp_draws(mu, cov, n=500):
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        Lc = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mu)))
        return sd, (mu + (Lc @ draw_rng.standard_normal((len(mu), n))).T)

    def gp_loop(kind, pin):
        """One GP arm's own uncertainty-sampling loop from the shared survey,
        observing on the shared post-survey noise stream."""
        rng_g = obs_rng()
        xs, ys, ax_x, ax_y = list(sx), list(sy), [], []
        egrid = GA.ell_grid(kind)
        for _ in range(N_ACTIVE):
            _, sd_a, _, _ = GA._gp_fit(np.array(xs), np.array(ys), GA.ACQ,
                                       kind, pin, egrid)
            xn = float(GA.ACQ[int(np.argmax(sd_a))])
            yn = observe(xn, rng_g)
            xs.append(xn); ys.append(yn); ax_x.append(xn); ax_y.append(yn)
        mu, cov = gp_fit_cov(np.array(xs), np.array(ys), kind, pin)
        return mu, cov, np.array(ax_x), np.array(ax_y)

    lin_mu, lin_cov, lin_ax, lin_ay = gp_loop("linear", False)
    pin_mu, pin_cov, pin_ax, pin_ay = gp_loop("log", True)
    lin_sd, lin_draws = gp_draws(lin_mu, lin_cov)
    pin_sd, pin_draws = gp_draws(pin_mu, pin_cov)

    # (c) the I-spline runs its OWN loop under the identical survey and the same
    #     post-survey noise stream, treated exactly as the GP arms above
    rng_i = obs_rng()
    reg = M.ISplineRegressor(domain=DOMAIN)
    for x, y in zip(sx, sy):
        reg.add(float(x), float(y))
    reg.fit(seed=SEED)
    isp_ax, isp_ay = [], []
    for _ in range(N_ACTIVE):
        xn = reg.next_x()
        yn = observe(xn, rng_i)
        reg.add(xn, yn); reg.fit(seed=SEED)
        isp_ax.append(xn); isp_ay.append(yn)
    sp_mu, sp_sd = reg.predict(tg)
    sp_draws = reg.sample(tg)
    isp_ax, isp_ay = np.array(isp_ax), np.array(isp_ay)

    def n_bad(draws):
        return int(sum((np.diff(d) < -1e-9).any() for d in draws))

    def crossings(mu):
        lvl = ALPHA * mu[-1]
        xs = []
        for i in range(len(tg) - 1):
            if (mu[i] - lvl) * (mu[i + 1] - lvl) < 0:
                fr = (lvl - mu[i]) / (mu[i + 1] - mu[i])
                xs.append((tg[i] + fr * (tg[i + 1] - tg[i]), lvl))
        return xs

    W = 7.0
    s = _fonts(W)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9), layout="constrained",
                             sharey=True)
    fig.patch.set_facecolor("white")
    SURVEY_C, ACTIVE_C = "#f2a900", "#d62728"     # shared survey vs own active
    panels = [
        (axes[0], lin_mu, lin_sd, lin_draws, lin_ax, lin_ay, "C7", "(a) GP-linear"),
        (axes[1], pin_mu, pin_sd, pin_draws, pin_ax, pin_ay, "C4", "(b) GP-log-pin"),
        (axes[2], sp_mu, sp_sd, sp_draws, isp_ax, isp_ay, "C0", "(c) Monotone I-spline"),
    ]
    for ax, mu, sd, draws, ax_x, ax_y, c, title in panels:
        ax.set_facecolor("white"); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for dd in draws[:N_SHOW]:
            ax.plot(tg, dd, color=c, lw=0.6, alpha=0.30)
        ax.fill_between(tg, mu - 1.96 * sd, mu + 1.96 * sd, color=c, alpha=0.15)
        ax.plot(tg, truth(tg), "k-", lw=1.8)
        ax.plot(tg, mu, color=c, lw=1.8)
        ax.scatter(ax_x, ax_y, marker="o", c=ACTIVE_C, s=24, edgecolor="black",
                   lw=0.5, zorder=5)                    # this arm's own active pts
        ax.scatter(sx, sy, marker="D", c=SURVEY_C, s=30, edgecolor="black",
                   lw=0.5, zorder=6)                    # shared survey pts
        lvl = ALPHA * mu[-1]
        ax.axhline(lvl, color="#d1495b", ls="--", lw=0.9)
        for (xc, yc) in crossings(mu):        # rings mark where the mean meets the level
            ax.plot(xc, yc, "o", ms=7, mfc="none", mec="black", mew=1.4, zorder=7)
        ax.set_title(title, fontweight="bold", fontsize=8.5 * s)
        ax.set_xlabel("Input $x$ (s)")
        ax.set_xlim(0, DOMAIN[1]); ax.set_ylim(-0.2, 1.3)
    axes[0].set_ylabel("Response")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="k", lw=1.8, label="true curve"),
        Line2D([], [], marker="D", ls="none", ms=6, mfc=SURVEY_C, mec="black",
               mew=0.5, label="survey (shared)"),
        Line2D([], [], marker="o", ls="none", ms=6, mfc=ACTIVE_C, mec="black",
               mew=0.5, label="active (own)"),
        Line2D([], [], color="#d1495b", ls="--", lw=0.9, label=r"$0.95\times$ plateau"),
        Line2D([], [], marker="o", ls="none", ms=7, mfc="none", mec="black",
               mew=1.4, label="mean crossing"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, fontsize=6.2 * s)

    _save(fig, "fig_gp_compare")
    print(f"[{FAMILY} seed={SEED}] active pts lin={len(lin_ax)} pin={len(pin_ax)} "
          f"isp={len(isp_ax)}; dips lin={n_bad(lin_draws)} pin={n_bad(pin_draws)} "
          f"isp={n_bad(sp_draws)}/{len(sp_draws)}; mean-cross lin={len(crossings(lin_mu))} "
          f"pin={len(crossings(pin_mu))} isp={len(crossings(sp_mu))}; sat_err "
          f"lin={M.saturation_error(tg, lin_mu, truth(tg)):.3f} "
          f"pin={M.saturation_error(tg, pin_mu, truth(tg)):.3f} "
          f"isp={M.saturation_error(tg, sp_mu, truth(tg)):.3f}")


def main(argv):
    fresh  = "--fresh" in argv
    jobs   = None
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    argv   = [a for a in argv if not a.startswith("--")]
    # Default: the campaign set (per-isotherm detail figures + basis +
    # final fit).
    KNOWN = {"basis", "final_fit", "details", "readout", "knots",
             "noise", "scorecard", "gp_compare"}
    wanted = set(argv) if argv else {"basis", "final_fit", "details"}
    if "all" in wanted:
        wanted = set(KNOWN)
    unknown = wanted - KNOWN
    if unknown:
        print(f"make_figures.py: unknown figure name(s) {sorted(unknown)}; "
              f"choose from {sorted(KNOWN)} or 'all'", file=sys.stderr)
        sys.exit(2)

    if "basis" in wanted:
        fig_basis()

    needs_exp = wanted & {"final_fit", "details"}
    if needs_exp:
        res = run_experiments(fresh=fresh, jobs=jobs)
        if "final_fit"   in wanted: fig_final_fit(res)
        if "details"     in wanted: fig_details(res)

    # Supporting figures: readout, knots, noise and scorecard load caches;
    # gp_compare runs a short one-seed illustration loop at render time
    for name, fn in (("readout", fig_readout), ("knots", fig_knots),
                     ("noise", fig_noise), ("scorecard", fig_scorecard),
                     ("gp_compare", fig_gp_compare)):
        if name in wanted:
            fn()


if __name__ == "__main__":
    main(sys.argv[1:])
