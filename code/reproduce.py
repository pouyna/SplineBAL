"""
reproduce.py -- one entry point to regenerate every cache, figure, and verified
number for the penalized monotone I-spline saturation-curve study.

It runs the whole pipeline in dependency order. Each step is an independent
subprocess; JAX/XLA memory is freed per run (poolutil recycles workers), so the
run is memory-safe. Picking --jobs: each JAX worker commits ~3 GB of virtual
memory (about 2 GB resident), so budget at most
    jobs = min(cores - 2, (RAM - ~4 GB for the OS) / 3 GB);
on the 16 GB machine the published results used, --jobs=3 was the largest
setting that ran reliably, and one worker too many fails mid-campaign with
Windows' "The paging file is too small" rather than just running slower. See
the root README for the details.

Layout
------
* Core libraries : isotherms, ispline_model (penalized + unpenalized models), readout,
                   gp_arms, poolutil.
* Campaign dispatchers (each `py <file>.py <subcommand>` writes one cache):
    recovery.py       main | noise | ablation | unpen25 | gppin
    readout_study.py  readout | gpxsat
    robustness.py     failure | anchor | window
                      (anchor --model=unpen11 runs the mechanism-diagnostic arm)
    generalization.py heldout | randxsat | survey | heldoutgppin
    design_choices.py knots | priors | acqgrid | diag | sampler | targets
    refits.py         penalty | studentt | parametric | unpen
    controls.py       monoproj | band | decoupled
* Figures         : make_figures.py <name> (basis/final_fit/details campaign
                    figures + knots/readout/noise/scorecard/gp_compare).
* Tests           : test_readout, test_surrogates (run first, before any
                    campaign; `reproduce.py tests` runs just these).
* Verify / util   : recompute_stats, real_data (apply to your own data).

Usage
-----
    python3 reproduce.py                # full pipeline: campaigns -> figures -> verify
    python3 reproduce.py --jobs=6       # more workers (better machine)
    python3 reproduce.py campaigns      # only the MCMC/GP campaigns (write caches)
    python3 reproduce.py figures        # only regenerate figures from existing caches
    python3 reproduce.py verify         # only recompute_stats
    python3 reproduce.py --list         # print the steps and exit

Notes
-----
* Every result cache lives in the code/data/ folder (all scripts read and write
  there). Campaigns overwrite their own results_*.json, so a re-run recomputes
  from scratch. To rebuild only part, run a single subcommand directly (each
  accepts --jobs=N) or delete just the caches you want from code/data/.
* The two "uniform" change-of-measure controls live in the same dispatcher and
  are selected by env vars (HELDOUT_XSAT, RANDXSAT_XDIST); the noise level and
  model live in isotherms.py (SIGMA_X, SIGMA_Y) and ispline_model.py (the
  production penalized prior, P=25). Change those once and every step follows.
* Total wall time is roughly a day at --jobs=3; the main recovery campaign and
  the two randomized-x_sat campaigns dominate.
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable]                       # the interpreter running this script

# (label, [script, subcommand, extra args...], env-overrides or None).  Order
# respects dependencies: results.json first (designs the read-out/noise
# reuse), then the GP arms, then the campaigns that splice cached arms, then the
# figures (need every cache), then the verification.
CAMPAIGNS = [
    ("main recovery (results.json)",   ["recovery.py", "main", "--fresh"],            None),
    ("Gaussian-process arms",          ["gp_arms.py"],                                None),
    ("noise sweep (penalized vs GP)",  ["recovery.py", "noise"],                      None),
    ("penalty ablation (unpen. P=11)", ["recovery.py", "ablation"],                   None),
    ("decoupled acquisition",          ["controls.py", "decoupled"],                  None),
    ("read-out analysis",              ["readout_study.py", "readout"],               None),
    ("GP x_sat read-out",              ["readout_study.py", "gpxsat"],                None),
    ("held-out benchmark",             ["generalization.py", "heldout"],              None),
    ("held-out uniform control",       ["generalization.py", "heldout"],  {"HELDOUT_XSAT": "uniform"}),
    ("randomized x_sat",               ["generalization.py", "randxsat", "--fresh"],  None),
    ("randomized x_sat uniform",       ["generalization.py", "randxsat", "--fresh"], {"RANDXSAT_XDIST": "uniform"}),
    ("survey-fraction sweep",          ["generalization.py", "survey"],               None),
    ("failure modes",                  ["robustness.py", "failure"],                  None),
    ("plateau anchor",                 ["robustness.py", "anchor"],                   None),
    ("window-edge sweep",              ["robustness.py", "window"],                   None),
    ("knot geometry",                  ["design_choices.py", "knots"],                None),
    ("prior sensitivity",              ["design_choices.py", "priors"],               None),
    ("acquisition-grid control",       ["design_choices.py", "acqgrid"],              None),
    # --- ablation, refit, and control arms.  Several reuse caches written
    #     above (refits.py refits the stored designs in results.json; the
    #     controls read the GP arms), so they run after the main campaigns.
    ("unpenalized P=25 arm",           ["recovery.py", "unpen25"],                    None),
    ("GP-log-pin noise sweep",         ["recovery.py", "gppin"],                      None),
    ("GP-log-pin held-out",            ["generalization.py", "heldoutgppin"],         None),
    # the matched-design control refits the designs of the main campaign, the
    # noise sweep and the held-out benchmark, and prints the GP-log-pin own-loop
    # caches beside them, so it runs after all of those
    ("GP matched-design control",      ["gp_arms.py", "matched"],                     None),
    ("penalty-order model ablation",   ["refits.py", "penalty"],                      None),
    ("unpenalized bases on stored designs", ["refits.py", "unpen"],                   None),
    ("Student-t robustness",           ["refits.py", "studentt"],                     None),
    ("parametric baselines",           ["refits.py", "parametric"],                   None),
    ("post-hoc monotonization",        ["controls.py", "monoproj"],                   None),
    ("band calibration + inferred sigma", ["controls.py", "band"],                   None),
    ("mechanism diagnostic (unpen. arm)",
     ["robustness.py", "anchor", "--model=unpen11"],                                  None),
    ("sampler-settings comparison",    ["design_choices.py", "sampler"],              None),
    ("sampler configuration grid",     ["design_choices.py", "targets"],              None),
    ("MCMC convergence diagnostics",   ["design_choices.py", "diag"],                 None),
]

FIGURES = [
    ("campaign figures (basis+fit+detail)",
     ["make_figures.py", "basis", "final_fit", "details"], {"MPLBACKEND": "Agg"}),
    ("remaining figures (knots/readout/noise/scorecard/gp)",
     ["make_figures.py", "knots", "readout", "noise", "scorecard", "gp_compare"],
     {"MPLBACKEND": "Agg"}),
]

# Fast regression guards. test_surrogates asserts every campaign takes the
# production model from one place, so it should never be optional. Both run
# in well under a minute.
TESTS = [
    ("read-out unit tests",  ["test_readout.py"], None),
    ("surrogate unit tests", ["test_surrogates.py"], None),
]

VERIFY = [
    ("recompute_stats (prints every quoted number)", ["recompute_stats.py"], None),
]


def run_step(label, argv, env_over, jobs, pass_jobs):
    env = dict(os.environ)
    if env_over:
        env.update(env_over)
    cmd = PY + [os.path.join(HERE, argv[0])] + argv[1:]
    if pass_jobs:
        cmd.append(f"--jobs={jobs}")
    print(f"\n=== {label} ===\n    {' '.join(argv)}"
          + (f"  [{', '.join(f'{k}={v}' for k, v in env_over.items())}]" if env_over else ""),
          flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=HERE, env=env).returncode
    print(f"    -> exit {rc}  ({time.time() - t0:.0f}s)", flush=True)
    return rc


def main(argv):
    jobs = 3      # fits a 16 GB machine: ~3 GB commit per JAX worker + ~4 GB
                  # for the OS (see module docstring); raise on larger hardware
    which = "all"
    for a in argv:
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        elif a in ("campaigns", "figures", "verify", "tests"):
            which = a
        elif a == "--list":
            for grp, steps in (("TESTS", TESTS), ("CAMPAIGNS", CAMPAIGNS),
                               ("FIGURES", FIGURES), ("VERIFY", VERIFY)):
                print(f"\n[{grp}]")
                for lab, av, ev in steps:
                    print(f"  {lab:34s} {' '.join(av)}")
            return

    plan = []
    if which in ("all", "tests"):
        plan += [(s_, False) for s_ in TESTS]      # test scripts take no --jobs
    if which in ("all", "campaigns"):
        plan += [(s, True) for s in CAMPAIGNS]
    if which in ("all", "figures"):
        plan += [(s, False) for s in FIGURES]      # figure scripts take no --jobs
    if which in ("all", "verify"):
        plan += [(s, False) for s in VERIFY]

    print(f"reproduce.py: {len(plan)} steps, jobs={jobs}", flush=True)
    t0 = time.time()
    failed = []
    for (label, av, ev), pass_jobs in plan:
        if run_step(label, av, ev, jobs, pass_jobs) != 0:
            failed.append(label)
    print(f"\n{'='*60}\nDONE in {time.time()-t0:.0f}s. "
          + ("all steps exited 0." if not failed else f"FAILED: {failed}"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
