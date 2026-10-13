"""
poolutil.py
===========
A process pool that bounds memory across a long MCMC campaign, so accumulated
JAX compiled-executable caches and live buffers in the workers do not run away
the machine's commit charge (and, on Windows, the page file).

The active-learning campaigns fit many problem shapes in one run (different
domains for the truncated-window arm, different budgets for the replication
arm, and so on). Each new shape compiles a new XLA executable, and a
persistent worker keeps every one it has ever compiled; on a 16 GB machine
three such workers can push the page file past the free disk.

We bound this by MANUAL BATCHING: the joblist is processed in chunks, and a
fresh ProcessPoolExecutor is opened and fully closed for each chunk, so every
worker exits and releases all its memory between chunks. This deliberately
avoids ProcessPoolExecutor(max_tasks_per_child=...), which deadlocks under the
Windows spawn start method for this workload.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


_XLA_PIN = "--xla_cpu_multi_thread_eigen=false"


def pin_threads():
    """Pin XLA/BLAS to one thread per worker. Idempotent; call before jax runs.

    Do not try to pin through XLA_FLAGS: ``intra_op_parallelism_threads`` is
    not an XLA flag (as a bare token XLA ignores it silently, and written as
    ``--intra_op_parallelism_threads=1`` XLA rejects unknown flags fatally,
    "Unknown flag in XLA_FLAGS", and aborts the process). The
    OMP/OPENBLAS/MKL variables below are what actually pin each worker.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_cpu_multi_thread_eigen" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " " + _XLA_PIN).strip()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def run_pool(fn, joblist, jobs, tasks_per_child=8, every=50, label="runs"):
    """Map `fn` over `joblist` on `jobs` workers, recycling the whole pool
    every `tasks_per_child * jobs` tasks. `fn(job) -> (key, value)`.
    Returns {key: value}. Pins BLAS/XLA to one thread per worker.
    """
    pin_threads()

    chunk = max(1, jobs * max(1, tasks_per_child))
    recs = {}
    done = 0
    total = len(joblist)
    print(f"Running {total} {label} on {jobs} workers "
          f"(fresh pool every {chunk} tasks) ...", flush=True)
    for start in range(0, total, chunk):
        batch = joblist[start:start + chunk]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(fn, j) for j in batch]
            for fut in as_completed(futs):
                key, val = fut.result()
                recs[key] = val
                done += 1
                if done % every == 0:
                    print(f"  [{done}/{total}]", flush=True)
        # pool is now fully torn down; workers have exited and freed memory
    print(f"  [{done}/{total}] done", flush=True)
    return recs
