"""
test_surrogates.py -- guarantee + smoke tests for the model variants.

Runs as a script (like test_readout.py) or under pytest:
    python3 test_surrogates.py
Covers: knots and grids respect x_min; the production model and sampler
defaults are what the campaigns assume; the first-order arm reproduces a
frozen bitwise reference; and, for each surrogate variant, the guarantees
hold (monotone arms: f'>=0 and f(0)=0; signed driver: f(0)=0 and
non-monotone representable).
"""
import numpy as np

import isotherms as ISO
import ispline_model as M

DATA = [(0.02, 0.1), (0.1, 0.4), (0.5, 0.8), (2.0, 0.97), (5.0, 1.0)]
# Frozen reference posterior mean for the FIRST-order penalty (seed 0,
# warmup=draws=50). FirstOrderRegressor must reproduce it bitwise, so the
# penalty-order ablation arm cannot be changed silently. (ISplineRegressor
# carries the production second-order penalty and is guarded separately.)
FIRST_ORDER_REF = np.array([0.300922, 0.709963, 0.896691, 1.071553])


def _fit(cls, warmup=50, draws=50, seed=0, n_iters=0):
    r = cls(domain=ISO.DOMAIN, n_interior=21, degree=3,
            n_init=5, n_iters=n_iters, warmup=warmup, draws=draws, chains=1)
    for x, y in DATA:
        r.add(x, y)
    return r.fit(seed=seed)


# --- Task 1: settings change ------------------------------------------------
def test_xmin_is_sigma_x():
    assert ISO.DOMAIN == (0.010, 5.0)
    assert ISO.DOMAIN[0] == ISO.SIGMA_X


def test_knots_and_grids_respect_xmin():
    knots = M.make_knots(ISO.DOMAIN, n_interior=21, degree=3)
    interior = knots[(knots > 0) & (knots < ISO.DOMAIN[1])]
    assert interior.min() >= ISO.DOMAIN[0] - 1e-12
    acq = M.acq_points(*ISO.DOMAIN)
    assert acq.min() >= ISO.DOMAIN[0] - 1e-12
    ev = M.eval_points(*ISO.DOMAIN)
    # DOMAIN[0] > 0, so the log reporting grid starts at x_min with no sub-x_min node
    assert np.isclose(ev[0], ISO.DOMAIN[0])
    assert ev.min() >= ISO.DOMAIN[0] - 1e-12


# --- the production model is the SECOND-order walk, and everything that fits
#     the "production surrogate" must actually use it ------------------------
def test_production_model_is_second_order():
    """ISplineRegressor._MODEL must be the second-order walk.

    Regression test for a real defect: when the second-order penalty was
    adopted, `ISplineRegressor._MODEL` was reassigned but the bare `_model`
    symbol was not, so anything naming `M._model` kept sampling the
    pre-adoption first-order model.  Two campaigns did exactly that.
    """
    assert M.ISplineRegressor._MODEL is M._model_second_order, (
        "production surrogate is not the second-order penalty")
    assert M.FirstOrderRegressor._MODEL is M._model, (
        "FirstOrderRegressor must keep the first-order penalty for the ablation")
    # a second-order walk needs two starting values (level + first step); a
    # first-order walk has only the level.
    import inspect
    src2 = inspect.getsource(M._model_second_order)
    assert "slope0" in src2 and "cumsum(first_diff)" in src2
    assert "slope0" not in inspect.getsource(M._model)
    # The heteroscedastic and plateau ablation arms must differ from production
    # by ONE feature, so they carry the same
    # second-order walk. They were first-order until 2026-09, which confounded
    # the penalty-order ablation with the likelihood/plateau change.
    for fn in (M._model_hetero, M._model_plateau):
        src = inspect.getsource(fn)
        assert "slope0" in src and "cumsum(first_diff)" in src, \
            f"{fn.__name__} is not the second-order walk"


def test_campaigns_take_the_production_model_from_one_place():
    """design_choices (prior sensitivity, MCMC diagnostics) must not carry its
    own copy of the model; both campaigns had drifted to the first-order one."""
    import design_choices as D
    assert D.PRODUCTION_MODEL is M.ISplineRegressor._MODEL
    src = open(D.__file__, encoding="utf-8").read()
    assert "NUTS(M._model)" not in src, "a campaign names the first-order model"
    assert "_priors_model" not in src, "a private copy of the model is back"


def test_production_sampler_defaults_unchanged():
    """The cached campaigns were produced at NumPyro's defaults; the recommended
    configuration is opt-in.  Flipping the defaults would silently invalidate
    every results_*.json, so guard them."""
    import inspect
    assert M.TARGET_ACCEPT == 0.8
    sig = inspect.signature(M.ISplineRegressor.__init__)
    assert sig.parameters["target_accept"].default == M.TARGET_ACCEPT
    assert sig.parameters["init_median"].default is False
    r = M.ISplineRegressor(domain=ISO.DOMAIN)
    assert r.target_accept == 0.8 and r.init_median is False


# --- first-order penalty preserved bitwise (FirstOrderRegressor) ------------
def test_first_order_bitwise_reference():
    r = _fit(M.FirstOrderRegressor)
    got = np.round(r.predict(np.array([0.05, 0.5, 2.0, 5.0]))[0], 6)
    np.testing.assert_array_equal(got, FIRST_ORDER_REF)


# --- Tasks 3-6: per-variant guarantees --------------------------------------
def test_hetero_is_monotone_and_fits():
    r = _fit(M.HeteroRegressor)
    fs = r.sample(M.eval_points(*ISO.DOMAIN))
    assert np.all(np.diff(fs, axis=1) >= -1e-6)
    assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6)


def test_second_order_is_monotone_and_fits():
    r = _fit(M.SecondOrderRegressor)
    fs = r.sample(M.eval_points(*ISO.DOMAIN))
    assert np.all(np.diff(fs, axis=1) >= -1e-6)
    assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6)


def test_plateau_monotone_and_infers_A():
    r = _fit(M.PlateauRegressor, warmup=100, draws=100)
    fs = r.sample(M.eval_points(*ISO.DOMAIN))
    assert np.all(np.diff(fs, axis=1) >= -1e-6)
    assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6)
    plateau = fs[:, -1].mean()
    assert 0.7 < plateau < 1.3


def test_signed_can_be_nonmonotone_but_f0_zero():
    knots = M.make_knots(ISO.DOMAIN, n_interior=21, degree=3)
    grid = M.eval_points(*ISO.DOMAIN)
    Bg = M.ispline_basis(grid, knots, 3)
    w = np.ones(Bg.shape[1]); w[10] = -3.0
    f = Bg @ w
    assert np.min(np.diff(f)) < 0
    assert np.isclose((M.ispline_basis(np.array([0.0]), knots, 3) @ w)[0], 0.0, atol=1e-8)
    r = _fit(M.SignedRegressor)
    assert r.sample(grid).shape[1] == len(grid)
    assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6)


# --- Task 7: roll-up guarantee matrix ---------------------------------------
def test_guarantee_matrix():
    grid = M.eval_points(*ISO.DOMAIN)
    for cls in (M.ISplineRegressor, M.HeteroRegressor, M.SecondOrderRegressor,
                M.PlateauRegressor, M.StudentTRegressor):
        r = _fit(cls, seed=1)
        fs = r.sample(grid)
        assert np.all(np.diff(fs, axis=1) >= -1e-6), f"{cls.__name__} not monotone"
        assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6), f"{cls.__name__} f(0)!=0"
        assert np.isfinite(np.median(r.tstar(0.95))), f"{cls.__name__} tstar failed"
    r = _fit(M.SignedRegressor, seed=1)
    assert np.allclose(r.sample(np.array([0.0])), 0.0, atol=1e-6)


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    sys.exit(1 if failed else 0)
