# SplineBAL: shape-constrained Bayesian active learning of saturation curves

A **saturation curve** is a monotone response that rises from zero and levels off: surface
coverage against pressure, reaction rate against substrate concentration, growth per cycle
against precursor pulse time. SplineBAL fits one from a handful of expensive, noisy
measurements with a surrogate that cannot violate that structure, and chooses where to
measure next.

The surrogate expands the response in **integrated splines (I-splines)** with non-negative
weights and a second-order roughness penalty on their logarithms, so for *every* posterior
draw, not only the posterior mean:

- `f(0) = 0` exactly, because every I-spline vanishes at the left boundary knot;
- `f'(x) >= 0` everywhere, because the derivative is a non-negative sum of non-negative bumps;
- the plateau is fixed by a **measurement** at the maximum input, not by a flatness prior.

Monotonicity is arithmetic, not a penalty. The surrogate's pipeline has no isotonic
projection, no running-maximum guard, and no post-hoc repair (the projections in
`controls.py monoproj` act only on an unconstrained Gaussian-process comparator, as a
control). The input that reaches any saturation level is read from the posterior by level
crossing, with no kinetic form assumed.

The method, the benchmark and the results are described in

> P. Navabi and C. G. Takoudis, *Shape-Constrained Bayesian Active Learning of
> Self-Limiting Saturation Curves*, [arXiv:2606.26577](https://arxiv.org/abs/2606.26577).

Please cite it if you use this code.

---

## Quick start

All code is in `code/`, and every command below runs from there.

```bash
pip install -r requirements.txt          # see "Requirements"
cd code
python3 test_readout.py                  # fast self-tests (seconds)
python3 test_surrogates.py
python3 recompute_stats.py               # every reported number, from the caches, ~10 s
```

The result caches (`code/data/results_*.json` and `results_band_raw.npz`, about 40 MB) are
distributed as an archive attached to the
[v1.0.0 release](https://github.com/pouyna/SplineBAL/releases/tag/v1.0.0); unzip it in the
repository root so the files land in `code/data/`. Without them `recompute_stats.py` reports
each section as skipped, and `reproduce.py` regenerates every cache from scratch (about a
day at `--jobs=3`).

`recompute_stats.py` recomputes every reported figure-of-merit, ratio, percentage and table
entry from the cached results without re-running any campaign. Run one section at a time:

```bash
python3 recompute_stats.py gp          # the Gaussian-process ablation
python3 recompute_stats.py readout     # the read-out and its coverage
python3 recompute_stats.py heldout     # the held-out benchmark
python3 recompute_stats.py outside     # behaviour outside the assumptions
```

Sections: `iso acq level gp readout xsat window noise anchor heldout heldunif outside
survey knots cost prior sigma band sampler diag learning regimes acqgrid decoupled
monoproj ablation gpmatched studentt parametric misc`.

### Fit your own measured curve

```bash
python3 real_data.py mycurve.csv --out=result
```

where `mycurve.csv` is two columns, input and response, with a header. This applies the
**frozen** surrogate (the knot geometry and priors of the benchmark, nothing retuned; the
sampler starts at the prior median with target acceptance 0.95, the setting the sampler
diagnostics recommend, in place of the campaigns' NumPyro defaults) and prints the level
crossing, the cost-indexed quantile family `x_q` (pick `q` from your cost ratio), the
one-sided upper limit, the response-domain risk curve, and two diagnostics that say whether
the answer should be trusted at all. Add `--robust`
for the Student-t (nu = 4) likelihood recommended where an occasional bad reading is
plausible. The script docstring explains how to reduce an in-situ thickness trace into the
required two columns.

---

## What is where

Eighteen Python files in `code/`: five core libraries, seven campaign dispatchers (each
with subcommands), one figure module, three utilities, and two test modules.

### Core libraries

| file | what it is |
|---|---|
| `ispline_model.py` | The surrogate. M-spline/I-spline basis (normalized B-splines and their exact antiderivatives, equivalent to Ramsay's recursion), the NumPyro model with the log-random-walk roughness penalty, NUTS sampling, uncertainty-sampling acquisition, the level-crossing read-out, and the active-learning driver. Also holds the comparison variants used by specific experiments (unpenalized, first-order, heteroscedastic, plateau-parameter, Student-t, signed). |
| `isotherms.py` | The five ground-truth kinetic families (Langmuir, Michaelis-Menten, Sips/Hill, Elovich, KWW), the noise model `y = f(clip(x + N(0, sx^2), x_min, x_max)) + N(0, sy^2)`, and the regime-sweep specifications. |
| `readout.py` | Everything derived from the posterior draw array: attained fraction, under-saturation risk curve, the quantile and risk-bounded read-outs, the conditioning factor `1/mu'(x)`, window sufficiency, plateau leverage, and the dip metrics used for the unconstrained arms. Pure NumPy. |
| `gp_arms.py` | Three unconstrained Gaussian-process arms differing by one feature at a time (linear axis, log axis, log axis with pinned origin), isolating multi-scale resolution from the shape constraint. No MCMC. `gp_arms.py` (no argument) runs the arm benchmark; `gp_arms.py matched` fits the GP arms to the stored I-spline designs and records how flat the GP bands were, the matched-design control. |
| `poolutil.py` | A recycling process pool that frees each worker's JAX/XLA memory periodically, so a long campaign does not exhaust system memory. |

### Campaign dispatchers: `python3 <file>.py <subcommand>`, each writing its cache in `code/data/`

| command | what it establishes |
|---|---|
| `recovery.py main` | Main campaign: 5 families x 4 strategies x 50 seeds plus noise-free regimes. Resumable. |
| `recovery.py noise` | Output noise sy in {0.01, 0.05, 0.1, 0.15}, the penalized surrogate against the resolution-matched GP arm. |
| `recovery.py ablation` | The unpenalized monotone I-spline (P = 11) in its own loop across noise. |
| `recovery.py unpen25` | The unpenalized monotone basis at production resolution (P = 25). |
| `recovery.py gppin` | The origin-matched GP arm across the noise sweep. No MCMC. |
| `readout_study.py readout` | The recommended input: interval coverage, bias, conditioning, and the quantile read-out family. |
| `readout_study.py gpxsat` | The unconstrained arm's read-out, same crossing convention. No MCMC. |
| `robustness.py failure\|anchor\|window` | Misspecification controls; the high-noise mechanism diagnostic (`anchor --model=unpen11`, with the production arm as a contrast); window-edge dependence. |
| `generalization.py heldout\|randxsat\|survey\|heldoutgppin` | Held-out benchmark on unseen curves; randomized rise location; survey-fraction sweep; the origin-matched GP arm on the held-out set. `HELDOUT_XSAT=uniform` and `RANDXSAT_XDIST=uniform` select the change-of-measure controls. |
| `design_choices.py knots\|priors\|acqgrid\|diag\|sampler\|targets` | Knot geometry; prior sensitivity; acquisition candidate grid; MCMC convergence diagnostics; the two sampler probes. |
| `refits.py penalty\|studentt\|parametric\|unpen` | Refits of the stored designs under a different penalty order or model variant, a Student-t likelihood, the five closed kinetic forms, or the two unpenalized bases on every strategy's designs. |
| `controls.py monoproj\|band\|decoupled` | Post-hoc monotonization of an unconstrained GP; credible-band calibration and the inferred noise and roughness scales (`band` also writes `results_band_raw.npz`); acquisition driven by an unconstrained spline. |

The whole pipeline takes about a day at `--jobs=3`; the main campaign and the two
randomized-location campaigns dominate.

### Figures, utilities and tests

| file | what it does |
|---|---|
| `make_figures.py` | Every figure. `make_figures.py all`, or by name: `basis final_fit details` (campaign figures) and `knots readout noise scorecard gp_compare`. `knots`, `readout`, `noise` and `scorecard` load caches; `gp_compare` runs a short one-seed illustration loop at render time. |
| `recompute_stats.py` | Every reported number, recomputed from the caches. |
| `reproduce.py` | One entry point that runs the whole pipeline (tests, campaigns, figures, diagnostics, statistics) in dependency order; `python3 reproduce.py --list` prints the steps. |
| `real_data.py` | Apply the frozen surrogate to your own measured curve (see above). |
| `test_readout.py` | Self-tests for `readout.py`: the median crossing recovers a known truth, the risk curve is non-increasing, the quantile read-out rises with `q` from the median crossing at `q = 0.5`, an unreachable level returns infinity, the dip and window diagnostics fire correctly, and the plateau leverage is positive. |
| `test_surrogates.py` | Self-tests for the surrogates: the production model is the second-order penalty, the first-order arm reproduces its frozen reference bitwise, every monotone variant stays monotone with `f(0) = 0`, the signed arm can be non-monotone, and knots and grids respect `x_min`. |

### Cached results

Every script reads and writes its `results_*.json` cache in `code/data/` (see
`code/data/README.md`). `results.json` is the large one (main campaign, with the stored
designs and posterior summaries for all 50 seeds x 5 families x 4 strategies). Every run is
independently seeded, so results do not depend on the worker count or on completion order;
`--jobs=N` only changes wall time.

```
SplineBAL/
  README.md, LICENSE
  requirements.txt  pinned package versions
  code/             the 18 source files (tables above)
  code/data/        results_*.json caches (release archive, see code/data/README.md)
  code/figures/     generated PNGs, written by make_figures.py
```

---

## Requirements

Python 3.13 or newer (also verified on 3.14). The versions used for the published results,
also pinned in `requirements.txt`:

```
numpy 2.3.3    scipy 1.16.2    matplotlib (any recent)
jax 0.10.1     numpyro 0.21.0
```

CPU only, no GPU. A single posterior fit takes about 2.5 s on a six-core desktop processor.
Everything that imports `ispline_model.py` needs JAX/NumPyro: the campaign dispatchers,
`gp_arms.py` (which imports helpers from `ispline_model.py` but runs no MCMC),
`make_figures.py`, `real_data.py` and `test_surrogates.py`. `readout.py` and
`recompute_stats.py` run without them.

Memory: each JAX worker peaks near 2 GB resident and about 3 GB committed, so keep `--jobs`
at most `min(cores - 2, (RAM - 4 GB) / 3 GB)`; on the 16 GB machine used for the published
results, `--jobs=3` was the largest setting that ran reliably.

---

## Scope

Every data set produced by this code is **synthetic**, generated from the ground truths in
`isotherms.py` and the held-out generators in `generalization.py`, under the noise model of
`isotherms.py`; no measurements are included.
`real_data.py` is provided so the frozen surrogate can be applied to a measured curve
without modification.

## License

MIT (see `LICENSE`).
