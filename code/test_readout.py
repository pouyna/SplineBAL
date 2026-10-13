"""
test_readout.py
===============
Self-tests for readout.py, the module that turns posterior draws into the
quantities the study reports. Run it before trusting a number that came out of
that module:

    python3 test_readout.py            # or: python3 -m pytest test_readout.py

These check the properties the method actually claims, not merely that the
functions run:

  1. the median level crossing recovers a known truth;
  2. the risk curve R(x) is non-increasing, which is the monotone-safety
     property that makes the risk-bounded input
     unique;
  3. the cost-indexed quantile read-out x_q is non-decreasing in q and gives the
     median crossing at q = 0.5, so the family runs from the median to the
     conservative upper end;
  4. a level that is never reached returns infinity rather than a silently
     wrong finite answer;
  5. negative_variation and upcrossings detect a deliberately planted dip and
     size it correctly against the detector noise;
  6. window sufficiency separates a window that brackets saturation from one
     that does not.

No MCMC, no cached results, runs in under a second.
"""
import sys

import numpy as np

import readout as RO

X = np.linspace(0.0, 5.0, 400)
TAU = 0.6676                      # Langmuir headline: f(2.0 s) = 0.95


def langmuir(x, tau=TAU):
    return 1.0 - np.exp(-np.asarray(x, float) / tau)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return bool(ok)


def main():
    ok = True
    print("readout.py self-tests\n")

    # 1. median crossing recovers the truth
    draws = np.array([langmuir(X) * (1.0 + 0.02 * i) for i in range(-5, 6)])
    med, sd, lo, hi, none = RO.level_crossing_median(draws, X, 0.95)
    ok &= check("median crossing recovers x_sat", abs(med - 2.0) < 0.05,
                f"got {med:.3f}, true 2.000")
    ok &= check("no draw fails to cross", none == 0.0)

    # 2. risk curve is non-increasing for monotone draws
    r = RO.risk_curve(draws, 0.95)
    ok &= check("risk curve non-increasing", bool(np.all(np.diff(r) <= 1e-12)),
                f"R(0)={r[0]:.2f} -> R(x_max)={r[-1]:.2f}")

    # 3. the quantile read-out x_q is non-decreasing in q and gives the median
    #    crossing at q = 0.5. Uses draws with genuinely different shapes (a spread
    #    of time constants), so their crossings differ; the scaled draws above all
    #    share one attained fraction.
    shape_draws = np.array([langmuir(X, TAU * (1.0 + 0.05 * i)) for i in range(-5, 6)])
    xq = [RO.input_from_quantile(shape_draws, X, 0.95, q) for q in (0.5, 0.7, 0.9, 1.0)]
    ok &= check("x_q is non-decreasing in q",
                all(a <= b + 1e-12 for a, b in zip(xq, xq[1:])),
                " <= ".join(f"{v:.2f}" for v in xq))
    med_s = RO.level_crossing_median(shape_draws, X, 0.95)[0]
    dx = X[1] - X[0]
    ok &= check("q = 0.5 gives the median crossing", abs(xq[0] - med_s) <= 2 * dx,
                f"x_0.5={xq[0]:.3f} vs median {med_s:.3f}")

    # 4. an unreachable level fails loudly
    flat = np.tile(np.full_like(X, 0.5), (5, 1))
    ok &= check("unreachable level returns infinity",
                RO.input_from_risk(flat, X, 1.5, 0.05) == float("inf"))

    # 5. a planted dip is detected and sized
    mu = langmuir(X).copy()
    mu[150:170] -= 0.08
    nv = RO.negative_variation(mu, X, sigma_y=0.05)
    uc = RO.upcrossings(mu, X, 0.95)
    ok &= check("planted dip detected", nv["n_descending_intervals"] > 0)
    ok &= check("dip depth measured against sigma_y",
                nv["dip_exceeds_noise"] and nv["max_dip_depth"] > 0.05,
                f"depth {nv['max_dip_depth']:.4f}")
    ok &= check("monotone curve gives a single upcrossing", uc["count"] == 1)

    # 6. window sufficiency separates the two cases
    sat = langmuir(X)[None, :]
    trunc = (1.0 - np.exp(-X / 8.0))[None, :]        # nowhere near plateau
    s1 = RO.window_sufficiency(sat, X, tol=0.05)
    s2 = RO.window_sufficiency(trunc, X, tol=0.05)
    ok &= check("window sufficiency separates", s1 > 0.9 and s2 < 0.1,
                f"saturating {s1:.2f}, truncated {s2:.2f}")

    # 7. plateau leverage is positive: a higher plateau moves the crossing later
    lev = RO.plateau_leverage(draws, X, 0.95)
    ok &= check("plateau leverage is positive", lev > 0, f"{lev:.3f} s per %")

    print("\n" + ("ALL PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def test_readout_module():
    """pytest entry point; the script form is main()."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
