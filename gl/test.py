"""Time the generation of 10^8 standard normal random numbers, first
serially and then in parallel, split into 10, 10^2, 10^3 and 10^4 jobs.

This is the Python version of test.R. The results are saved in test.csv.
"""

import csv
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

# Workers that return large arrays keep a high memory footprint after the
# arrays are freed. joblib's loky backend mistakes this for a memory leak and
# replaces the worker, which is harmless but prints a scary warning.
warnings.filterwarnings("ignore", message="A worker stopped while some jobs")


def available_cores():
    """Number of cores allocated by Slurm if we are on a cluster,
    or the number of cores on this machine otherwise."""
    return int(os.environ.get("SLURM_NTASKS_PER_NODE", os.cpu_count()))


def rnorm(n):
    """Return n standard normal random numbers, like rnorm(n) in R."""
    return np.random.default_rng().standard_normal(n)


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
        parallel(delayed(rnorm)(1) for _ in range(cores))

        times = {
            "time0": system_time(rnorm, 10**8),
            "time1": system_time(
                parallel, (delayed(rnorm)(10**7) for _ in range(10))),
            "time2": system_time(
                parallel, (delayed(rnorm)(10**6) for _ in range(10**2))),
            "time3": system_time(
                parallel, (delayed(rnorm)(10**5) for _ in range(10**3))),
            "time4": system_time(
                parallel, (delayed(rnorm)(10**4) for _ in range(10**4))),
        }

    with open("test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "user", "system", "elapsed"])
        for name, t in times.items():
            writer.writerow([name, *(f"{x:.3f}" for x in t)])


if __name__ == "__main__":
    main()
