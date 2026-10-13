"""
ispline_model.py
================
Bayesian monotonic I-spline regression for active learning of saturation
curves.

The saturating response is written as a non-negative weighted sum of
integrated splines (I-splines),

        f(x) = sum_m w_m I_m(x),      w_m >= 0,

where each I-spline I_m is the running integral of a non-negative M-spline
bump M_m.  Two physical properties then hold for EVERY choice of weights:

  * f(0) = 0           (every I-spline starts at zero), and
  * f'(x) = sum_m w_m M_m(x) >= 0   (sum of non-negative bumps times
    non-negative weights), so the curve can never decrease.

Self-limiting saturation is not imposed by a prior: the initial design
(survey_design) includes a measurement at the maximum input, so the
plateau height is pinned by data and the curve flattens because the data say
so, not because a constraint forces it.

Acquisition: uncertainty sampling (next_x), i.e. sample where the posterior
response std is largest, to learn the whole saturation curve in as few
measurements as possible.  Quality is scored by the saturation error across
levels (saturation_error), which emphasizes the rise rather than the flat
plateau.

Public API
----------
    ISplineRegressor(domain=(t_lo, t_hi), n_interior=21, degree=3,
                     sigma_prior=0.3, n_init=5, n_iters=15)
        .add(x, y)
        .fit(seed=...)
        .predict(x)             -> (mean, std) of f at x
        .predict_deriv(x)       -> (mean, std) of f'(x)
        .sample(x)              -> (S, len(x)) array of monotone posterior draws
        .next_x()               acquisition: uncertainty sampling (argmax std)
        .tstar(alpha=0.95)      -> posterior sample of the saturation input

    run_active_learning(observe, domain, seed=..., acquisition='uncertainty', **kw)
        acquisition is 'uncertainty' (default), 'uncertainty_geom', 'random',
        'random_log' or 'grid'.
    saturation_error(t, mu, yt) -> RMS vertical error across saturation levels
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
from scipy.interpolate import BSpline
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import isotherms as ISO   # SIGMA_X for the heteroscedastic arm (numpy/scipy only, no import cycle)

numpyro.set_host_device_count(1)


# ---------------------------------------------------------------------------
# Two distinct grids, kept separate on purpose
# ---------------------------------------------------------------------------
# 1. The ACQUISITION / DATAPOINT grid is the physical input resolution: an
#    input is commanded and measured to the nearest millisecond, so candidate
#    inputs and the inputs actually run live on a LINEAR lattice at
#    GRID_SPACING = 0.001 s.  This is what lets uncertainty sampling propose (and
#    re-measure, under fresh noise) any input to the millisecond, including
#    the sub-0.0175 s onset a coarse grid could not reach.
#
# 2. The REPORTING / EVALUATION grid is where the fitted spline is sampled to
#    draw the curve, score the saturation-error metric, and read the level
#    crossing.  The I-/M-spline basis is smooth in the LOG domain (log-spaced
#    knots), so a modest LOG-spaced grid reproduces every metric and crossing
#    that a 0.001 s linear grid does (verified to <=3e-4 on the metric and to
#    the interpolated crossing exactly), with ~10x fewer points and no near-
#    origin metric aliasing.  Using a linear 0.001 s grid here would only pile
#    thousands of redundant nodes on the flat plateau.
GRID_SPACING = 0.001     # acquisition/datapoint LINEAR resolution (s)
EVAL_N = 800             # reporting/evaluation LOG-spaced node count


def acq_points(lo, hi, spacing=GRID_SPACING):
    """Candidate/datapoint grid: LINEAR, at the physical input resolution."""
    return np.linspace(lo, hi, int(round((hi - lo) / spacing)) + 1)


def eval_points(lo, hi, n=EVAL_N):
    """Reporting/evaluation grid: LOG-spaced, matched to the smooth-in-log
    basis.  When lo <= 0 a node at 0 is prepended so f(0)=0 anchors the curve
    (the log part then starts at the input resolution GRID_SPACING)."""
    if lo > 0:
        return np.geomspace(lo, hi, n)
    return np.concatenate([[0.0], np.geomspace(GRID_SPACING, hi, n - 1)])


# ---------------------------------------------------------------------------
# Spline basis
# ---------------------------------------------------------------------------
def make_knots(domain, n_interior=7, degree=3):
    """Open knot vector on domain=(t_lo, t_hi).

    Left boundary pinned at 0 so every I-spline vanishes at the origin.
    Interior knots are log-spaced to concentrate resolution near the origin.
    """
    t_lo, t_hi = domain
    interior = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), n_interior + 2))[1:-1]
    return np.concatenate([np.zeros(degree + 1), interior, np.full(degree + 1, t_hi)])


def ispline_basis(x, knots, degree=3):
    """I-spline design matrix, shape (len(x), n_basis).

    Column i is I_i(x): monotone non-decreasing, 0 at x=0, 1 at the right
    knot.  Computed in closed form as the exact antiderivative of the
    underlying B-spline, normalized so the M-spline encloses unit area.

    x is clamped to the knot span: scipy returns NaN outside it (which
    nan_to_num would silently map to 0, i.e. the curve would collapse to
    zero just past the domain), whereas the correct continuation of an
    I-spline is 0 below the span and 1 above it.
    """
    x = np.clip(np.asarray(x, dtype=float), knots[0], knots[-1])
    nb = len(knots) - degree - 1
    X = np.zeros((len(x), nb))
    left = knots[0]
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        anti = BSpline(knots, c, degree, extrapolate=False).antiderivative()
        norm = (degree + 1) / (knots[i + degree + 1] - knots[i])
        X[:, i] = np.clip(np.nan_to_num((anti(x) - anti(left)) * norm, nan=0.0), 0.0, 1.0)
    return X


def mspline_basis(x, knots, degree=3):
    """M-spline (derivative) design matrix, shape (len(x), n_basis).

    Column i is M_i(x) >= 0, the bump whose running integral is I_i.
    f'(x) = sum_m w_m M_m(x).
    """
    x = np.asarray(x, dtype=float)
    nb = len(knots) - degree - 1
    X = np.zeros((len(x), nb))
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        val = BSpline(knots, c, degree, extrapolate=False)(x)
        norm = (degree + 1) / (knots[i + degree + 1] - knots[i])
        X[:, i] = np.nan_to_num(val, nan=0.0) * norm
    return X


# ---------------------------------------------------------------------------
# Probabilistic model (NumPyro)
# ---------------------------------------------------------------------------
def _model(Xb, mask, sigma_prior, tau_scale, y=None):
    """FIRST-order penalty. NOT the production model -- see the note below.

    Reached only through FirstOrderRegressor, for the penalty-order ablation.
    ISplineRegressor._MODEL is REASSIGNED to _model_second_order further down this file
    (search for "production penalty"), so anything that wants the production
    surrogate must take it from ISplineRegressor._MODEL rather than naming
    this function.

    Generative model for one fit: a penalized (P-spline) monotone I-spline.

    The curve keeps a fixed, deliberately over-complete basis of many
    log-spaced I-splines (P basis functions), and a first-order random walk on
    the LOG weights supplies the roughness penalty.  Putting the walk on the
    log weights keeps every weight positive by construction, so monotonicity
    and f(0)=0 hold exactly for every draw, while the increments are shrunk
    toward zero by an inferred roughness scale tau_eta (Eilers-Marx):

        log_w0  ~ Normal(log(1/P), 1.5)        overall level (pinned by the x_max datum)
        tau_eta ~ HalfNormal(tau_scale)        inferred roughness scale
        z_m     ~ Normal(0, 1)                 non-centered increments (funnel-safe)
        log w   = log_w0 + [0, cumsum(tau_eta * z)]
        w       = exp(log w) > 0               => f monotone, f(0)=0 exactly
        sigma   ~ HalfNormal(sigma_prior)      single observation-noise term
        y       ~ Normal(Xb @ w, sigma)        Gaussian likelihood (masked padding)

    Many fixed knots give the resolution; tau_eta adapts the effective
    smoothness to the noise, so the surrogate neither underfits a sharp onset
    nor overfits under heavy scatter.  The knots never move and are not chosen
    from the data, so no selective-inference penalty is incurred.  No flatness
    constraint: the plateau is pinned by the x_max measurement in survey_design.
    """
    P = Xb.shape[1]
    log_w0 = numpyro.sample("log_w0", dist.Normal(np.log(1.0 / P), 1.5))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 1]).to_event(1))
    logw = log_w0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)])
    w = numpyro.deterministic("w", jnp.exp(logw))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.Normal(Xb @ w, sigma).mask(mask), obs=y)


# ---------------------------------------------------------------------------
# Sampler configuration
# ---------------------------------------------------------------------------
# PRODUCTION defaults, unchanged: NumPyro's uniform initialization and target
# acceptance 0.8.  Every cached result was produced with these.
#
# RECOMMENDED for new work: init_to_median with target acceptance 0.95.  The
# second-order walk builds log w from a DOUBLE cumulative sum, so under the
# prior the last log-weight has sd ~23 and exceeds +-75 one time in a hundred,
# and w = exp(log w) is enormous in the tails.  A uniform draw in unconstrained
# space can start a chain out there;
# at target 0.8 the steps are big enough to walk back, which is what the
# production configuration's 11-22% divergence rate is paying for.  Raising the
# target ALONE makes it worse (smaller steps strand the chain: Rhat ~ 20,
# ESS ~ 2), and more warmup does not help.  Initializing at the prior median
# fixes it at no cost: Rhat 1.022 -> 1.006, ESS 207 -> 611, divergences
# 16.1% -> 1.9% on the seed-0 Langmuir fit.
#
# The production defaults are DELIBERATELY not changed. Paired over 50 fits
# (5 families x 10 seeds) on identical designs the two configurations give the
# same answers (saturation error 0.0387 vs 0.0385, paired Wilcoxon p=0.45), so
# switching would invalidate
# every cache for no change in any reported quantity.  See design_choices.py sampler
# and design_choices.py diag.
TARGET_ACCEPT = 0.8          # production
TARGET_ACCEPT_RECOMMENDED = 0.95


def _get_mcmc(warmup, draws, chains, model=_model,
              target_accept=TARGET_ACCEPT, init_median=False):
    """Fresh MCMC object per fit; the compiled NUTS kernel is still reused.

    Observations are padded to a fixed shape (see ISplineRegressor.fit), so
    JAX's structural jit cache serves every fit after the first from the same
    compiled executable (one cache entry, not one per fit), which bounds
    memory without any cache clearing.  Do not add jax.clear_caches() after
    fits (it forces a full recompilation per fit and triples the runtime),
    and do not pass
    jit_model_args=True: it re-traces the model with the data as traced
    arguments, which changes the floating-point schedule and hence the MCMC
    realization, breaking bitwise reproducibility of the cached results.
    """
    kw = {}
    if target_accept != 0.8:                 # NUTS default; keep the call
        kw["target_accept_prob"] = float(target_accept)   # identical otherwise
    if init_median:
        from numpyro.infer.initialization import init_to_median
        kw["init_strategy"] = init_to_median
    return MCMC(NUTS(model, **kw), num_warmup=warmup,
                num_samples=draws, num_chains=chains,
                chain_method="vectorized", progress_bar=False)


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------
class ISplineRegressor:
    """Monotone I-spline surrogate with NUTS posterior sampling.

    Parameters
    ----------
    domain : (float, float)
    n_interior, degree : int
        degree=3 (cubic) gives C^2 continuity;
        The default n_interior=21 yields P = n_interior + degree + 1 = 25
        basis functions; the coarse comparison basis is n_interior=7 (P = 11).
    sigma_prior : float
        Scale of the HalfNormal prior on observation noise.
    n_init, n_iters : int
        Active-learning budget.
    warmup, draws, chains : int
        NUTS settings.
    """

    def __init__(self, domain=ISO.DOMAIN, n_interior=21, degree=3,
                 sigma_prior=0.3, tau_scale=0.3,
                 n_init=5, n_iters=15, warmup=500, draws=500, chains=1,
                 acq_grid="linear", target_accept=TARGET_ACCEPT,
                 init_median=False):
        self.domain = domain
        self.acq_grid = acq_grid
        self.degree = degree
        self.knots = make_knots(domain, n_interior, degree)
        self.nb = len(self.knots) - degree - 1
        self.sigma_prior = sigma_prior
        self.tau_scale = tau_scale
        self.n_init, self.n_iters = n_init, n_iters
        self.warmup, self.draws, self.chains = warmup, draws, chains
        self.target_accept, self.init_median = target_accept, init_median
        self.t_obs, self.y_obs = [], []
        self._w = None
        self._sigma = None
        self._tau = None

    # --- model hooks: subclasses override these two to swap the surrogate,
    #     reusing all of the read-out machinery below unchanged ---
    _MODEL = staticmethod(_model)

    def _model_args(self, Xb, mask):
        return dict(Xb=jnp.array(Xb), mask=jnp.array(mask),
                    sigma_prior=self.sigma_prior, tau_scale=self.tau_scale)

    def add(self, x, y):
        self.t_obs.append(float(x))
        self.y_obs.append(float(y))
        return self

    def fit(self, seed=0):
        n = len(self.t_obs)
        nmax = self.n_init + self.n_iters
        Xb = np.zeros((nmax, self.nb))
        Xb[:n] = ispline_basis(np.array(self.t_obs), self.knots, self.degree)
        yy = np.zeros(nmax)
        yy[:n] = self.y_obs
        mask = np.zeros(nmax, bool)
        mask[:n] = True
        mcmc = _get_mcmc(self.warmup, self.draws, self.chains, model=self._MODEL,
                         target_accept=self.target_accept,
                         init_median=self.init_median)
        mcmc.run(jax.random.PRNGKey(seed), y=jnp.array(yy),
                 **self._model_args(Xb, mask))
        p = mcmc.get_samples()
        self._w = np.array(p["w"])
        self._sigma = np.array(p["sigma"])
        # the inferred roughness scale (recompute_stats sigma); absent for the unpenalized arm
        self._tau = np.array(p["tau_eta"]) if "tau_eta" in p else None
        return self

    def sample(self, x):
        Bg = ispline_basis(np.asarray(x, float), self.knots, self.degree)
        return self._w @ Bg.T

    def predict(self, x):
        fs = self.sample(x)
        return fs.mean(0), fs.std(0, ddof=1)

    def predict_deriv(self, x):
        Mg = mspline_basis(np.asarray(x, float), self.knots, self.degree)
        d = self._w @ Mg.T
        return d.mean(0), d.std(0, ddof=1)

    def next_x(self, n_grid=None, acq_grid=None):
        """Uncertainty sampling: next input at the maximum posterior response std.

        Standard active-learning rule for learning the whole curve; it drives
        down the curve-value uncertainty where the surrogate is least sure.
        The candidate set is built at GRID_SPACING (0.001 s by default), so the
        rule can select, and re-select, any input to the millisecond; when a
        previously measured point remains the most uncertain, it is chosen again
        and re-measured under a fresh noise draw (the readings are kept as
        independent replicates by the likelihood).

        acq_grid selects the CANDIDATE grid geometry:
          'linear'    -- equally spaced candidates at GRID_SPACING (default);
          'geometric' -- log-spaced candidates of the same count.
        Defaults to self.acq_grid ('linear' unless set at construction).
        """
        g = acq_grid or self.acq_grid
        if g == "geometric":
            tg = eval_points(self.domain[0], self.domain[1],
                             n=len(acq_points(self.domain[0], self.domain[1])))
        elif n_grid is not None:
            tg = np.linspace(self.domain[0], self.domain[1], n_grid)
        else:
            tg = acq_points(self.domain[0], self.domain[1])
        _, sd = self.predict(tg)
        return float(tg[int(np.argmax(sd))])

    @staticmethod
    def _crossings(t, fs, targets):
        """First-crossing time of each draw fs[s] at level targets[s] (interp)."""
        mask = fs >= targets[:, None]
        has = mask.any(axis=1)
        idx = np.clip(np.argmax(mask, axis=1), 1, fs.shape[1] - 1)
        out = []
        for s in range(fs.shape[0]):
            if not has[s]:
                continue
            if mask[s, 0]:          # already at/above the level at the first
                out.append(t[0])    # grid point: don't extrapolate left of it
                continue
            i = idx[s]
            y0, y1 = fs[s, i - 1], fs[s, i]
            t0, t1 = t[i - 1], t[i]
            if y1 > y0:
                out.append(t0 + (targets[s] - y0) * (t1 - t0) / (y1 - y0))
        return np.array(out)

    def tstar(self, alpha=0.95, n_grid=None):
        t = (np.linspace(self.domain[0], self.domain[1], n_grid)
             if n_grid else eval_points(self.domain[0], self.domain[1]))
        fs = self.sample(t)
        return self._crossings(t, fs, alpha * fs[:, -1])

    def monotonicity_report(self, n_grid=None):
        t = (np.linspace(0.0, self.domain[1], n_grid)
             if n_grid else eval_points(0.0, self.domain[1]))
        fs = self.sample(t)
        bad = int(sum((np.diff(fs[s]) < -1e-9).any() for s in range(fs.shape[0])))
        mu = fs.mean(0)
        return dict(n_draws=fs.shape[0], non_monotone=bad,
                    f0=float(mu[0]), f_max=float(mu[-1]),
                    frac_mono_mean=float((np.diff(mu) >= -1e-12).mean()))


# ---------------------------------------------------------------------------
# Evaluation metric: the saturation error across levels
# ---------------------------------------------------------------------------
SAT_LEVELS = np.arange(0.05, 0.96, 0.05)   # 5%, 10%, ..., 95% saturation


def _level_time(t, y, target):
    """Input at which monotone curve y first reaches `target` (interp)."""
    idx = np.where(y >= target)[0]
    if len(idx) == 0:
        return t[-1]
    i = idx[0]
    if i == 0:
        return t[0]
    y0, y1 = y[i - 1], y[i]
    t0, t1 = t[i - 1], t[i]
    return t0 + (target - y0) * (t1 - t0) / (y1 - y0) if y1 > y0 else t0


def saturation_error(t, mu, yt, levels=SAT_LEVELS, A=1.0):
    """RMS vertical error of the predicted curve at each saturation level.

    For each fraction alpha we find the input where the *true* curve reaches
    alpha*A and compare the predicted response there to alpha*A.  This
    measures how well the curve VALUE is recovered across the rising region,
    without the flat plateau dominating as it would in a uniform-grid RMSE.
    """
    err = []
    for a in levels:
        xa = _level_time(t, yt, a * A)
        err.append(float(np.interp(xa, t, mu)) - a * A)
    return float(np.sqrt(np.mean(np.array(err) ** 2)))


# ---------------------------------------------------------------------------
# Active-learning driver
# ---------------------------------------------------------------------------
def survey_design(n, domain, rng):
    """Initial survey design: the two domain endpoints plus log-LHS in between.

    The endpoint at the maximum input pins the saturated value (so no
    flatness prior is needed), the minimum-input point anchors the onset,
    and the remaining n-2 points are a proper log-spaced Latin-hypercube
    survey of the interior: [0,1] is split into n-2 equal-probability strata,
    one uniform draw is taken per stratum, and the strata are mapped through
    log so the survey points are log-uniform across the domain.
    """
    t_lo, t_hi = domain
    if n <= 1:
        return np.array([t_hi])
    if n == 2:
        return np.array([t_lo, t_hi])
    k = n - 2
    u = (np.arange(k) + rng.random(k)) / k      # stratified Latin-hypercube on [0,1]
    # The shuffle is redundant before the sort below, but removing it would
    # advance the rng stream differently and change every seeded data set,
    # invalidating the cached results; keep it for reproducibility.
    rng.shuffle(u)
    mid = t_lo * (t_hi / t_lo) ** u             # log-uniform mapping
    return np.concatenate([[t_lo], np.sort(mid), [t_hi]])


def run_active_learning(observe, domain=ISO.DOMAIN, seed=42,
                        n_init=5, n_iters=15, record_history=False,
                        truth=None, acquisition='uncertainty', **reg_kw):
    """Run the active-learning loop (or a non-adaptive baseline).

    acquisition :
        'uncertainty' -- uncertainty sampling: next point at argmax posterior std.
        'uncertainty_geom' -- the same rule over a log-spaced candidate set of
                         the same count (the candidate-grid control of
                         design_choices.py acqgrid).
        'random'      -- uniform random inputs over the domain.
        'random_log'  -- log-uniform random inputs (the domain is log-scaled,
                         so this is the fair random baseline; plain 'random'
                         starves the fast early rise).
        'grid'        -- non-adaptive: a fresh log-spaced design of N points at
                         each budget N (what a practitioner runs); shares no
                         survey/active state.
    record_history records the saturation error across levels vs N.

    Monotonicity is tallied after EVERY refit (reg.mono_bad_total /
    reg.mono_draws_total) over all posteriors of the run.
    """
    rng = np.random.default_rng(seed)
    tg = eval_points(0.0, domain[1])
    yt = truth(tg) if truth is not None else None
    hist = []
    tally = {"bad": 0, "draws": 0}

    def _record(r):
        rep = r.monotonicity_report()
        tally["bad"] += rep["non_monotone"]
        tally["draws"] += rep["n_draws"]
        if record_history and truth is not None:
            mu, _ = r.predict(tg)
            hist.append(saturation_error(tg, mu, yt))

    def _finish(r):
        r.mono_bad_total, r.mono_draws_total = tally["bad"], tally["draws"]
        # Bound memory across the many runs of a Monte Carlo study: clear the
        # JAX cache once per RUN (one recompilation), which does not affect any
        # numbers (compilation does not change the math).
        jax.clear_caches()
        gc.collect()
        return r, (hist if record_history else None)

    if acquisition == 'grid':
        reg = None
        for N in range(n_init, n_init + n_iters + 1):
            reg = ISplineRegressor(domain=domain, n_init=n_init, n_iters=n_iters, **reg_kw)
            for x in np.geomspace(domain[0], domain[1], N):
                reg.add(float(x), observe(float(x), rng))
            reg.fit(seed=seed)
            _record(reg)
        return _finish(reg)

    reg = ISplineRegressor(domain=domain, n_init=n_init, n_iters=n_iters, **reg_kw)
    for x in survey_design(n_init, domain, rng):
        reg.add(x, observe(x, rng))
    reg.fit(seed=seed)
    _record(reg)

    for _ in range(n_iters):
        if acquisition == 'uncertainty':
            xn = reg.next_x()
        elif acquisition == 'uncertainty_geom':
            xn = reg.next_x(acq_grid='geometric')
        elif acquisition == 'random_log':
            xn = float(domain[0] * (domain[1] / domain[0]) ** rng.random())
        else:                                   # 'random' (uniform)
            xn = float(rng.uniform(domain[0], domain[1]))
        reg.add(xn, observe(xn, rng))
        reg.fit(seed=seed)
        _record(reg)

    return _finish(reg)


# ===========================================================================
# MODEL VARIANTS -- comparators used by specific experiments. The production
# surrogate is ISplineRegressor above; these share its basis/grids/metrics.
# ===========================================================================


# --- heteroscedastic (errors-in-variables) likelihood -----------------------
def _model_hetero(Xb, Xd, mask, sigma_prior, tau_scale, sigma_x, y=None):
    """The PRODUCTION second-order penalized model with an errors-in-variables
    (heteroscedastic) likelihood: the input jitter maps to an output-scale term
    f'(x)*sigma_x, so steep regions are honestly noisier. The weights and their
    prior are identical to _model_second_order, so this arm differs from the
    production surrogate in the likelihood ALONE."""
    P = Xb.shape[1]
    log_w0 = numpyro.sample("log_w0", dist.Normal(np.log(1.0 / P), 1.5))
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.5))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 2]).to_event(1))
    first_diff = slope0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)])
    logw = log_w0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(first_diff)])
    w = numpyro.deterministic("w", jnp.exp(logw))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    scale = jnp.sqrt(sigma ** 2 + (Xd @ w * sigma_x) ** 2)
    numpyro.sample("y", dist.Normal(Xb @ w, scale).mask(mask), obs=y)


class HeteroRegressor(ISplineRegressor):
    """Production second-order penalty + heteroscedastic errors-in-variables
    likelihood (model-ablation arm, refits.py penalty)."""
    _MODEL = staticmethod(_model_hetero)

    def _model_args(self, Xb, mask):
        Md = np.zeros_like(Xb)
        n = len(self.t_obs)
        Md[:n] = mspline_basis(np.array(self.t_obs), self.knots, self.degree)
        args = super()._model_args(Xb, mask)
        args.update(Xd=jnp.array(Md), sigma_x=float(ISO.SIGMA_X))
        return args


# --- second-order roughness penalty (the production prior) ------------------
def _model_second_order(Xb, mask, sigma_prior, tau_scale, y=None):
    """The monotone I-spline with a SECOND-order random walk on the log-weights
    (curvature penalty, linear-in-log null space): the PRODUCTION model.
    Second differences ~ N(0, tau_eta)."""
    P = Xb.shape[1]
    log_w0 = numpyro.sample("log_w0", dist.Normal(np.log(1.0 / P), 1.5))
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.5))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 2]).to_event(1))
    first_diff = slope0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)])
    logw = log_w0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(first_diff)])
    w = numpyro.deterministic("w", jnp.exp(logw))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.Normal(Xb @ w, sigma).mask(mask), obs=y)


class SecondOrderRegressor(ISplineRegressor):
    """The second-order penalty under its own name; identical to the
    production ISplineRegressor, kept so the model-ablation refit can name
    every arm explicitly."""
    _MODEL = staticmethod(_model_second_order)


# --- Student-t likelihood (robust to occasional bad metrology readings) -----
_STUDENTT_NU = 4.0


def _model_studentt(Xb, mask, sigma_prior, tau_scale, y=None):
    """Production second-order prior with a Student-t(nu=4) likelihood in
    place of the Gaussian. Heavy tails downweight an outlying
    reading; monotonicity and f(0)=0 are unchanged since they live in the prior.
    This is the configuration recommended for field data with occasional bad
    metrology."""
    P = Xb.shape[1]
    log_w0 = numpyro.sample("log_w0", dist.Normal(np.log(1.0 / P), 1.5))
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.5))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 2]).to_event(1))
    first_diff = slope0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)])
    logw = log_w0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(first_diff)])
    w = numpyro.deterministic("w", jnp.exp(logw))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.StudentT(_STUDENTT_NU, Xb @ w, sigma).mask(mask), obs=y)


class StudentTRegressor(ISplineRegressor):
    """Production second-order prior with a Student-t(nu=4) robust likelihood."""
    _MODEL = staticmethod(_model_studentt)


# --- plateau as an inferred parameter ----------------------------------------
def _model_plateau(Xb, mask, sigma_prior, tau_scale, y=None):
    """f = A * sum_m v_m I_m with v on the simplex (the PRODUCTION second-order
    log-random-walk + softmax, keeping the roughness penalty) and A>0 inferred,
    so f(x_max)=A carries the plateau uncertainty. Monotone (v>=0), f(0)=0.
    Because sum(v)=1 and every I-spline equals 1 at x_max, f(x_max)=A, so the
    existing alpha*f(x_max) read-out equals alpha*A with A's uncertainty folded
    in. The softmax is shift-invariant, so the walk's level log_w0 is absorbed
    into A and only the first step slope0 and the curvature increments remain;
    this arm differs from the production surrogate in the plateau
    parameterization ALONE."""
    P = Xb.shape[1]
    logA = numpyro.sample("logA", dist.Normal(0.0, 0.5))
    A = jnp.exp(logA)
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.5))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 2]).to_event(1))
    first_diff = slope0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)])
    logv = jnp.concatenate([jnp.zeros(1), jnp.cumsum(first_diff)])
    v = jax.nn.softmax(logv)
    w = numpyro.deterministic("w", A * v)
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.Normal(Xb @ w, sigma).mask(mask), obs=y)


class PlateauRegressor(ISplineRegressor):
    """Production second-order penalty with the plateau as an inferred
    parameter (model-ablation arm, refits.py penalty)."""
    _MODEL = staticmethod(_model_plateau)


# --- unconstrained penalized (Eilers-Marx) spline, acquisition driver --------
def _model_signed(Xb, mask, sigma_prior, tau_scale, y=None):
    """Penalized UNCONSTRAINED spline (Eilers-Marx): same I-spline basis, signed
    weights with a first-order Gaussian random-walk roughness penalty. Can be
    non-monotone; f(0)=0 still holds because every I-spline vanishes at 0. Used
    as the acquisition DRIVER in the decoupling control (controls.py decoupled)."""
    P = Xb.shape[1]
    w0 = numpyro.sample("w0", dist.Normal(0.0, 1.0))
    tau_eta = numpyro.sample("tau_eta", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([P - 1]).to_event(1))
    w = numpyro.deterministic("w", w0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau_eta * z)]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.Normal(Xb @ w, sigma).mask(mask), obs=y)


class SignedRegressor(ISplineRegressor):
    """Unconstrained penalized spline; the decoupling control's acquisition driver."""
    _MODEL = staticmethod(_model_signed)


# ===========================================================================
# The production penalty is the SECOND-ORDER random walk. On the paired refit
# of the stored designs (refits.py penalty) it beats the first-order penalty on
# curve accuracy and on read-out bias and coverage, and like every variant here
# it keeps monotonicity exact. ISplineRegressor carries it; the first-order
# penalty exists as FirstOrderRegressor solely for that ablation.
# ===========================================================================
ISplineRegressor._MODEL = staticmethod(_model_second_order)


class FirstOrderRegressor(ISplineRegressor):
    """The original FIRST-order roughness penalty (pre-adoption model), kept only
    for the penalty-order ablation (first- vs second-order)."""
    _MODEL = staticmethod(_model)


#
# UNPENALIZED: monotone I-spline with INDEPENDENT Half-Normal weights and NO
# roughness penalty (the penalized model's predecessor, P=11 log knots). Used
# by the penalty ablation (recovery.py 'ablation') as the midpoint of the
# separable-axes argument,
#     penalized vs unpenalized     = what the roughness PENALTY buys (accuracy),
#     unpenalized vs GP-log-pin    = what MONOTONICITY buys (well-posedness).

def _model_unpenalized(Xb, mask, sigma_prior, y=None):
    """Unpenalized likelihood: independent Half-Normal weights, no
    adjacent-weight penalty (monotone by non-negativity, but no smoothness
    adaptation)."""
    tau_w = numpyro.sample("tau_w", dist.HalfNormal(1.0))
    w = numpyro.sample("w", dist.HalfNormal(tau_w * jnp.ones(Xb.shape[1])).to_event(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_prior))
    numpyro.sample("y", dist.Normal(Xb @ w, sigma).mask(mask), obs=y)


class UnpenalizedRegressor:
    """Monotone I-spline surrogate with P=11 log knots and Half-Normal weights
    (no roughness penalty; the penalized model's predecessor). Same
    add/fit/predict/sample/next_x API as ISplineRegressor."""

    def __init__(self, domain=ISO.DOMAIN, n_interior=7, degree=3,
                 sigma_prior=0.3, n_init=5, n_iters=15,
                 warmup=500, draws=500, chains=1):
        self.domain, self.degree = domain, degree
        self.knots = make_knots(domain, n_interior, degree)
        self.nb = len(self.knots) - degree - 1
        self.sigma_prior = sigma_prior
        self.n_init, self.n_iters = n_init, n_iters
        self.warmup, self.draws, self.chains = warmup, draws, chains
        self.t_obs, self.y_obs, self._w = [], [], None
        self._sigma = None

    def add(self, x, y):
        self.t_obs.append(float(x)); self.y_obs.append(float(y)); return self

    def fit(self, seed=0):
        n = len(self.t_obs); nmax = self.n_init + self.n_iters
        Xb = np.zeros((nmax, self.nb))
        Xb[:n] = ispline_basis(np.array(self.t_obs), self.knots, self.degree)
        yy = np.zeros(nmax); yy[:n] = self.y_obs
        mask = np.zeros(nmax, bool); mask[:n] = True
        mcmc = MCMC(NUTS(_model_unpenalized), num_warmup=self.warmup, num_samples=self.draws,
                    num_chains=self.chains, chain_method="vectorized",
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), Xb=jnp.array(Xb), mask=jnp.array(mask),
                 sigma_prior=self.sigma_prior, y=jnp.array(yy))
        p = mcmc.get_samples()
        self._w = np.array(p["w"])
        self._sigma = np.array(p["sigma"])   # same name as ISplineRegressor
        return self

    def sample(self, x):
        return self._w @ ispline_basis(np.asarray(x, float), self.knots, self.degree).T

    def predict(self, x):
        fs = self.sample(x); return fs.mean(0), fs.std(0, ddof=1)

    def next_x(self):
        tg = acq_points(self.domain[0], self.domain[1])
        _, sd = self.predict(tg)
        return float(tg[int(np.argmax(sd))])

    def monotone_report(self):
        t = eval_points(0.0, self.domain[1]); fs = self.sample(t)
        bad = int(sum((np.diff(fs[s]) < -1e-9).any() for s in range(fs.shape[0])))
        return bad, fs.shape[0]


if __name__ == "__main__":
    import time
    A, tau = 1.0, 0.6676

    def truth(x):
        x = np.maximum(x, 0.0)
        return A * (1.0 - np.exp(-x / tau))

    def observe(x, rng):
        x_true = np.clip(x + rng.normal(0, 0.05), *ISO.DOMAIN)
        return float(truth(np.array([x_true]))[0] + rng.normal(0, 0.05))

    t0 = time.time()
    reg, _ = run_active_learning(observe, domain=ISO.DOMAIN, seed=42, truth=truth)
    rep = reg.monotonicity_report()
    tg = eval_points(0.0, ISO.DOMAIN[1])
    mu, _ = reg.predict(tg)
    rmse = float(np.sqrt(np.mean((mu - truth(tg)) ** 2)))
    ts = reg.tstar(0.95)
    print(f"self-test: RMSE={rmse:.4f}, non-monotone={rep['non_monotone']}/{rep['n_draws']} (final), "
          f"{reg.mono_bad_total}/{reg.mono_draws_total} (all refits)")
    print(f"x*(0.95) median={np.median(ts):.2f}s  ({time.time()-t0:.0f}s)")
