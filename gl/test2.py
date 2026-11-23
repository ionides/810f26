import os
import time
import csv
import numpy as np
from joblib import Parallel, delayed

# code to find how many cores are allocated, if we are on a cluster
# or how many cores the machine has, if we are on a single-user computer.
cores = os.environ.get("SLURM_NTASKS_PER_NODE")
cores = int(cores) if cores is not None else os.cpu_count()


def rnorm(n):
    return np.random.default_rng().standard_normal(n)


def rnorm0(n):
    np.random.default_rng().standard_normal(n)
    return 0


def timer(f):
    """Return (user, system, elapsed) times in seconds for calling f(),
    similar to system.time() in R. User and system times are for the
    main process only, not the worker processes."""
    t0 = os.times()
    e0 = time.perf_counter()
    f()
    e1 = time.perf_counter()
    t1 = os.times()
    return (t1.user - t0.user, t1.system - t0.system, e1 - e0)


with Parallel(n_jobs=cores) as parallel:
    # start up the worker processes before timing anything
    parallel(delayed(rnorm0)(1) for i in range(cores))

    time0 = timer(lambda: rnorm(10**8))
    time1 = timer(lambda: parallel(delayed(rnorm0)(10**7) for i in range(10)))
    time2 = timer(lambda: parallel(delayed(rnorm0)(10**6) for i in range(10**2)))
    time3 = timer(lambda: parallel(delayed(rnorm0)(10**5) for i in range(10**3)))
    time4 = timer(lambda: parallel(delayed(rnorm0)(10**4) for i in range(10**4)))

with open("test2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["", "user", "system", "elapsed"])
    for name, t in zip(["time0", "time1", "time2", "time3", "time4"],
                       [time0, time1, time2, time3, time4]):
        w.writerow([name] + [f"{x:.3f}" for x in t])
