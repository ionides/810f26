"""Time the generation of 10^8 standard normal random numbers, first
serially and then in parallel, split into 10, 10^2, 10^3 and 10^4 jobs.

Unlike test.py, the parallel jobs return 0 rather than their random numbers,
so the results don't have to be sent back to the main process. Comparing
test2.csv with test.csv shows how much time that communication takes.
"""

import csv
import os
import time

import numpy as np
from joblib import Parallel, delayed


def available_cores():
    """Number of cores allocated by Slurm if we are on a cluster,
    or the number of cores on this machine otherwise."""
    return int(os.environ.get("SLURM_NTASKS_PER_NODE", os.cpu_count()))


def rnorm(n):
    """Return n standard normal random numbers, like rnorm(n) in R."""
    return np.random.default_rng().standard_normal(n)


def rnorm0(n):
    """Generate n standard normal random numbers, then discard them and
    return 0."""
    rnorm(n)
    return 0


def system_time(f, *args):
    """Return (user, system, elapsed) seconds taken by f(*args), like
    system.time() in R. User and system times are for this process only,
    not the worker processes."""
    t0, e0 = os.times(), time.perf_counter()
    f(*args)
    t1, e1 = os.times(), time.perf_counter()
    return t1.user - t0.user, t1.system - t0.system, e1 - e0


def main():
    cores = available_cores()
    with Parallel(n_jobs=cores) as parallel:
        # Start the workers now, so that start-up time isn't counted below.
        parallel(delayed(rnorm0)(1) for _ in range(cores))

        times = {
            "time0": system_time(rnorm, 10**8),
            "time1": system_time(
                parallel, (delayed(rnorm0)(10**7) for _ in range(10))),
            "time2": system_time(
                parallel, (delayed(rnorm0)(10**6) for _ in range(10**2))),
            "time3": system_time(
                parallel, (delayed(rnorm0)(10**5) for _ in range(10**3))),
            "time4": system_time(
                parallel, (delayed(rnorm0)(10**4) for _ in range(10**4))),
        }

    with open("test2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "user", "system", "elapsed"])
        for name, t in times.items():
            writer.writerow([name, *(f"{x:.3f}" for x in t)])


if __name__ == "__main__":
    main()
