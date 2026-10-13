"""
isotherms.py
============
Ground-truth saturation isotherms and the measurement-noise model used to
generate synthetic saturation-curve data for testing the ``ispline_model``
surrogate.

Five kinetic families are provided, each a callable ``f(x) -> response``:

  Langmuir       A (1 - exp(-x/tau))            ideal single-site chemisorption
  Dissociative   A x / (K + x)                  Michaelis-Menten / dissociative
  Sips           A (x/ts)^n / (1 + (x/ts)^n)    sigmoidal, cooperative (n>1)
  Elovich        min(A, (A/gamma) ln(1+x/tE))   logarithmic, capped (kinked)
  DR-KWW         A (1 - exp(-(x/tp)^beta))      dispersive stretched-exponential

All are normalized to a saturated value A (default 1.0).  Helper
``make_isotherm`` returns a family by name; ``standard_isotherms`` returns the
five "headline" instances, each tuned so that f(x_sat)=s*A at the SAME
x_sat=2.0 s (s=0.95) on the domain DOMAIN = [0.010, 5.0] s, so differences in recovery
reflect curve SHAPE alone.

Noise model (input + output):

        y_obs(x) = f( clip(x + N(0, sigma_x^2), x_min, x_max) ) + N(0, sigma_y^2)

i.e. input jitter sigma_x perturbs the true input before the curve is
evaluated, then output noise sigma_y is added to the reading.  ``make_observe``
wraps any ground-truth f into an ``observe(x, rng)`` callable of the form the
active-learning driver expects.
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Default benchmark constants
# ---------------------------------------------------------------------------
A_DEFAULT = 1.0          # saturated value (response normalized to 1)
X_SAT = 2.0              # input at which f = s * A
S_TARGET = 0.95          # saturation fraction defining X_SAT
DOMAIN = (0.010, 5.0)    # measurement window (s); x_min = sigma_x (can't resolve finer than the jitter)
SIGMA_X = 0.01           # input jitter (s)        -- input noise
SIGMA_Y = 0.1            # output noise


# ---------------------------------------------------------------------------
# Isotherm families.  Each builder returns f(x) with f(X_SAT)=S_TARGET*A.
# The shape parameter is solved (where closed-form) so all share x_sat.
# ---------------------------------------------------------------------------
def langmuir(A=A_DEFAULT, x_sat=X_SAT, s=S_TARGET, tau=None):
    """A (1 - exp(-x/tau)).  tau solved from x_sat unless given explicitly."""
    if tau is None:
        tau = -x_sat / np.log(1 - s)
    f = lambda x: A * (1.0 - np.exp(-np.maximum(x, 0.0) / tau))
    f.label = f"Langmuir (tau={tau:.3g})"
    f.params = dict(A=A, tau=tau)
    return f


def dissociative(A=A_DEFAULT, x_sat=X_SAT, s=S_TARGET, K=None):
    """A x / (K + x)  (Michaelis-Menten).  K solved from x_sat unless given."""
    if K is None:
        K = x_sat * (1 - s) / s
    f = lambda x: A * np.maximum(x, 0.0) / (K + np.maximum(x, 0.0))
    f.label = f"Dissociative (K={K:.3g})"
    f.params = dict(A=A, K=K)
    return f


def sips(A=A_DEFAULT, x_sat=X_SAT, s=S_TARGET, n=2.0, ts=None):
    """A (x/ts)^n / (1 + (x/ts)^n)  sigmoidal.  ts solved from x_sat unless given.

    For fixed n, f(x_sat)=s*A  <=>  (x_sat/ts)^n = s/(1-s).
    """
    if ts is None:
        ts = x_sat / (s / (1 - s)) ** (1.0 / n)
    def f(x):
        u = (np.maximum(x, 0.0) / ts) ** n
        return A * u / (1.0 + u)
    f.label = f"Sips (n={n:.3g}, ts={ts:.3g})"
    f.params = dict(A=A, n=n, ts=ts)
    return f


def elovich(A=A_DEFAULT, x_sat=X_SAT, s=S_TARGET, tE=None, gamma=None):
    """min(A, (A/gamma) ln(1 + x/tE)), capped at A.

    (A/gamma) ln(1+x_sat/tE) = s*A ties gamma, tE, and x_sat together; the
    one not given is solved so that f(x_sat)=s*A (tE defaults to 0.5 when
    neither is given, matching the headline curve).
    """
    if gamma is None:
        tE = 0.5 if tE is None else tE
        gamma = np.log(1 + x_sat / tE) / s
    elif tE is None:
        tE = x_sat / (np.exp(s * gamma) - 1.0)
    f = lambda x: np.minimum(A, (A / gamma) * np.log(1.0 + np.maximum(x, 0.0) / tE))
    f.label = f"Elovich (gamma={gamma:.3g}, tE={tE:.3g})"
    f.params = dict(A=A, gamma=gamma, tE=tE)
    return f


def dr_kww(A=A_DEFAULT, x_sat=X_SAT, s=S_TARGET, beta=0.55, tp=None):
    """A (1 - exp(-(x/tp)^beta))  dispersive KWW.  tp solved from x_sat unless given."""
    if tp is None:
        tp = x_sat / ((-np.log(1 - s)) ** (1.0 / beta))
    f = lambda x: A * (1.0 - np.exp(-(np.maximum(x, 0.0) / tp) ** beta))
    f.label = f"DR-KWW (beta={beta:.3g}, tp={tp:.3g})"
    f.params = dict(A=A, beta=beta, tp=tp)
    return f


_FAMILIES = {
    "Langmuir": langmuir,
    "Dissociative": dissociative,
    "Sips": sips,
    "Elovich": elovich,
    "DR-KWW": dr_kww,
}


def make_isotherm(name, **kw):
    """Return a ground-truth isotherm callable by family name.

    Extra keyword arguments are passed to the builder, e.g.
    make_isotherm("Sips", n=3) or make_isotherm("Langmuir", tau=1.2).
    """
    if name not in _FAMILIES:
        raise KeyError(f"unknown isotherm {name!r}; choose from {list(_FAMILIES)}")
    return _FAMILIES[name](**kw)


# Display name -> (family name, builder kwargs).  Kept as plain data so that
# parallel workers can rebuild an isotherm from a picklable spec (the isotherm
# callables themselves are lambdas and cannot cross process boundaries).
STANDARD_SPECS = {
    "Langmuir":     ("Langmuir", {}),
    "Dissociative": ("Dissociative", {}),
    "Sips (n=2)":   ("Sips", {"n": 2.0}),
    "Elovich":      ("Elovich", {}),
    "D-R KWW":      ("DR-KWW", {"beta": 0.55}),
}


def standard_isotherms(A=A_DEFAULT):
    """The five headline isotherms, all sharing x_sat=2.0 s and saturated value A.

    Returns an ordered dict {display_name: f}.
    """
    return {nm: make_isotherm(fam, A=A, **kw)
            for nm, (fam, kw) in STANDARD_SPECS.items()}


def true_level_time(f, target, lo=0.0, hi=DOMAIN[1]):
    """Exact input at which the monotone ground truth f first reaches target.

    Root-found on the callable itself (not a grid), so it stays accurate for
    crossings near the origin where a uniform grid is far too coarse.
    Returns hi if the level is not reached within [lo, hi].
    """
    flo = float(f(np.array([lo]))[0]) - target
    fhi = float(f(np.array([hi]))[0]) - target
    if flo >= 0.0:
        return lo
    if fhi <= 0.0:
        return hi
    return float(brentq(lambda x: float(f(np.array([x]))[0]) - target, lo, hi))


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
def make_observe(f, domain=DOMAIN, sigma_x=SIGMA_X, sigma_y=SIGMA_Y):
    """Wrap a ground-truth f into observe(x, rng) -> noisy measurement.

        y = f( clip(x + N(0, sigma_x^2), domain) ) + N(0, sigma_y^2)

    Input jitter is applied to the true input; output noise is added to the
    reading.  The recorded abscissa is the
    commanded x, so the surrogate sees jitter as extra apparent scatter --
    largest where the curve is steep -- and absorbs it into its single noise
    term.
    """
    t_lo, t_hi = domain

    def observe(x, rng):
        x_true = np.clip(x + rng.normal(0.0, sigma_x), t_lo, t_hi)
        return float(f(np.array([x_true]))[0] + rng.normal(0.0, sigma_y))

    return observe


# ---------------------------------------------------------------------------
# Parameter-sweep helper (for the "behavior within each family" study)
# ---------------------------------------------------------------------------
def family_variant_specs(name):
    """Picklable specs for the three regime variants of one family.

    Returns {variant_label: builder_kwargs}; rebuild each isotherm with
    make_isotherm(name, **kwargs).  Labels must not change: the results cache
    is keyed on them.

    The values are chosen to span each family's accessible structures on the
    measurement window.  The one-parameter families (Langmuir, dissociative)
    have a single time scale, so the sweep spans it from onset-dominated
    (nearly all curvature in the first tenths of a second) to saturation at
    the window edge (the slowest variant reaches 95% just inside x=5 s).
    For the two-parameter families the scale is re-solved so every variant
    keeps f(2.0 s) = 0.95 A and ONLY the shape changes: Sips from a barely
    sigmoidal onset (n=1.3) to a step-like one (n=4), Elovich from a gentle
    to a strong logarithmic bend (the cap then always falls near 2.2 s), and
    KWW from strong dispersion (beta=0.35, instant rise then a long crawl)
    to nearly single-exponential (beta=0.85).
    """
    if name == "Langmuir":      # vary tau: onset-dominated -> window-edge saturation
        return {f"tau={v}": {"tau": v} for v in (0.15, 0.6676, 1.5)}
    if name == "Dissociative":  # vary K: steep -> window-edge saturation (x_sat=19K)
        return {f"K={v}": {"K": v} for v in (0.03, 0.10526, 0.25)}
    if name == "Sips":          # vary Hill n at fixed x_sat: mild -> step-like onset
        return {f"n={v}": {"n": v} for v in (1.3, 2.0, 4.0)}
    if name == "Elovich":       # vary gamma at fixed x_sat: gentle -> strong log bend
        return {f"g={v}": {"gamma": v} for v in (1.2, 1.69415, 2.4)}
    if name == "DR-KWW":        # vary beta at fixed x_sat: strong -> weak dispersion
        return {f"b={v}": {"beta": v} for v in (0.35, 0.55, 0.85)}
    raise KeyError(f"unknown family {name!r}")


def family_variants(name, A=A_DEFAULT):
    """Three shape-parameter variants of one family, for sensitivity studies.

    Returns {variant_label: f}.  Each f is A-normalized; only the shape
    parameter changes (the others keep their headline value).
    """
    return {lab: make_isotherm(name, A=A, **kw)
            for lab, kw in family_variant_specs(name).items()}


if __name__ == "__main__":
    # Verify every standard isotherm hits s*A at x_sat and starts near 0.
    print(f"Checking standard isotherms (A={A_DEFAULT}, target f({X_SAT})={S_TARGET*A_DEFAULT}):")
    for nm, f in standard_isotherms().items():
        print(f"  {nm:14s} f(0.005)={float(f(np.array([0.005]))[0]):.3f}  "
              f"f({X_SAT})={float(f(np.array([X_SAT]))[0]):.4f}  "
              f"f(5.0)={float(f(np.array([5.0]))[0]):.4f}")
    print("\nParameter-sweep variants:")
    for fam in _FAMILIES:
        labels = list(family_variants(fam).keys())
        print(f"  {fam:14s} -> {labels}")
