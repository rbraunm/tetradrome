# The scheduler

**One rule: every calculation is a scheduler job.** An engine describes its work as jobs; the
scheduler decides where and when each one runs. A calculation done in the caller's process,
outside a job, is a defect: nothing can place it, bound its memory, or run it beside other work.

Code: `src/tetradrome/scheduler/`. Engines enter through `scheduler.run_to_result` (one
calculation) or `scheduler.run_many` (several that share the machine).

## Jobs

A job is a callable with its inputs, the jobs it depends on, and:

- **paths**: the ways it can run, each with the cores, RAM and VRAM it needs there;
- **cost**: its predicted work, in the unit of its kind (states for a slice, operations for a
  reduction);
- **kind**: the family of jobs that share that unit, so measurements of one family never
  predict another.

A result never passes through the scheduler's process as a value. The process that computed it
serializes it: up to 64 KiB rides in the job's completion message, and anything larger is written
to the run's result store, a directory in RAM-backed `/dev/shm` where the system has one, with
only its location reported. A consumer reads it in its own process. A stored result with one
consumer is handed over, and that consumer deletes the file; one with several is deleted when the
last of them has finished. A result can be partitioned into shards so a consumer takes only its
part. The store is removed when the run ends, and a store left by a process that died is removed
by the next run.

## Handlers

| Path | Runs in | Costs | For |
|---|---|---|---|
| inline | the scheduler's own process | nothing to dispatch; holds up the loop while it runs | work shorter than a dispatch |
| CPU pinned | a pool worker, pinned for the job to CPUs of one NUMA node | a hand-over; a process start when none is waiting | most work |
| CPU unpinned | a pool worker on any CPUs | the same, no locality | work too large for one node |
| GPU, warm | the standing worker that holds the device context | queueing behind other warm jobs | small GPU jobs |
| GPU, fresh | its own process and context | creating a context | large GPU jobs |

A run keeps a pool of CPU workers. A worker takes one job after another: it pins itself to the
job's CPUs, runs it, publishes the result and waits. One left holding more than 128 MiB after a
job is retired, so what a large job leaves behind is not charged to the jobs after it; the pool
ends with the run. Workers are forked from a template process that has already imported numpy
and bound the compiled grid kernels (`engines/floer/prewarm.py`), so a worker starts ready. On
Windows workers are spawned cold and each is charged a measured import baseline.

## What is measured

Each handler has a speed and a cost. A benchmark measures them once for a machine and its software
(`tetradrome/calibrate.py`), and every calculation after that starts out with them:

- **work rate**: seconds per cost unit, per kind and device. The inline path and both CPU
  placements run the same code, so they share a rate.
- **sharing a core**: per kind, a job's time when a physical core's two hyperthreads both run it,
  over its time with the core to itself. Measured with two pinned workers released together, on
  one core while the rest of the machine is idle.
- **dispatch overhead**: seconds to hand a job with nothing to do to a waiting worker and reap
  it, and what starting a worker adds when none is waiting (2 ms and 20 ms on the container).
- **worker start-up**: a fresh worker's RAM baseline and a GPU context's VRAM.

The benchmark runs each engine's `benchmark()` through the scheduler: every Seifert and cube
invariant of one six-crossing knot, and the histogram, HFK-hat and τ graphs of its grid (grid
number 8). It takes the fastest of three passes, and about 18 s in all on the 32-CPU container.

The figures are one JSON file per machine in the user's cache directory (`TETRADROME_CACHE_DIR`
moves it; `scheduler/calibration_store.py`). They are used while both keys hold:

- **the machine**: processor model, the CPUs the scheduler may use with their NUMA nodes and
  physical cores, and the GPUs. Not RAM, load, uptime or kernel, so a reboot changes nothing.
- **the software**: a hash of the package's source, and the versions of the interpreter, numpy,
  Numba and cupy.

When a key has changed or there are no figures, the first calculation runs the benchmark and logs
that it is doing so and why. `python -m tetradrome.calibrate` runs it on demand, for a change the
keys do not see; the provisioner runs it at install.

While jobs run, a worker's private memory is sampled; the ledger charges the larger of declared
and measured, and logs a job that exceeds its declaration. A kind the benchmark does not cover is
learned while jobs run, for the life of the process.

Open:

- One rate per kind. A kind whose time per unit changes with its size is predicted roughly.
- GPU rates are not benchmarked; there has been no GPU host to build that on.
- Two processes that both find no figures both run the benchmark, and slow each other's.

## The planner

For each ready job (`scheduler/placement.py`):

1. Drop the paths this machine can never serve. If none is left the job is infeasible, reported
   before the run starts, and only its own component is abandoned.
2. Choose among the rest by measurement. Inline, when the predicted run time is no more than a
   worker's dispatch overhead. Otherwise the worker path with the smallest predicted run plus
   dispatch time. Where a figure is unmeasured, the order the builder declared.
3. Admit the job if the chosen path's CPUs and memory are free now, per NUMA node and against
   the global ceiling less a margin; otherwise it waits. A busy path never degrades a job to a
   slower one.

A job gets a physical core to itself while any is free, on either node. It takes the second
hyperthread of a busy core only when the benchmark measured that two jobs of its kind sharing a
core finish more than one alone (each less than twice as slow), and the same holds for the kind
already there.
A kind with no such figure never shares. On the 32-CPU container generation slices share
(1.5 times slower each, so a third more done) and histogram slices do not (2 times slower each).

Each pass takes a bounded number of ready jobs in the order the graph lists them, so in a graph
of many calculations (a roster of knots) the earlier ones finish, and free what they hold, before
the later ones start. Among those, the job with the longest tail is tried first: the predicted
time of the longest chain of work from its start to the end of the graph, from each kind's
measured rate. The work the finish time hangs on therefore starts as early as it can.

Each calculation in a graph has a peak: the largest footprint among its jobs. While anything is
running, a job is admitted only if the peak of every calculation already started, and of its own,
would still fit beside it. Without that, the quick producers of many calculations fill memory
with held results until no consumer fits and nothing can run. When nothing is running a job is
admitted whenever it fits.

Under memory pressure the largest held results are spilled to disk rather than stalling.

## Failure

A job that raises, or whose worker exits without reporting (killed for memory, or unable to load
the job), fails. Its whole component is abandoned; other components finish. `run_to_result`
raises on any failure, so no partial answer is returned.

## What an engine does

- Builds a graph and calls `run_to_result`. For several independent calculations it builds each
  one's jobs and calls `run_many`, which puts them in one graph under their names and returns each
  one's answer or failure (`hfk_hat_many`, `tau_many`).
- Prices every job before it runs. The Floer engine first runs a histogram graph (generators per
  grading) and builds the main graph from its result.
- Cuts its work where the mathematics is independent, so no long job stands alone. HFK-hat's
  slices are split by Alexander grading, each grading is merged by its own job, and each boundary
  map's rank is its own job; no job holds more than one grading.
- Keeps a serial pure-Python form of each calculation as the oracle its scheduled form is tested
  against (`grid_complexes`, `grading_histogram_reference`, `tau_reference`).

## Measured

On a 32-CPU, two-socket container, one knot of grid number 10 (3.6 million states):

| | Seconds |
|---|---|
| HFK-hat, end to end, histogram included | 15 |
| of which the histogram | 2 |
| τ, end to end | 314 |
| handing a job to a waiting worker | 0.002 |
| starting a worker | 0.020 |
| a determinant through `compute()`, against 0.04 ms called directly | 0.0003 |

The 185 roster knots of grid number up to 10, as one `run_many` graph each:

| | Seconds |
|---|---|
| HFK-hat for all 185, sharing the machine with four other processes | 3,730 |
| τ for all 185, alone | 1,358 |

A batch of six grid-number-10 knots, HFK-hat with the histograms already known:

| | Seconds | CPUs busy, average |
|---|---|---|
| generation slices of 4096 states | 179 | 9.3 |
| slices of 32768 states | 139 | 9.8 |
| and results kept out of the scheduler's process | 86 | 13.4 |
| and a merge per Alexander grading, a rank per boundary map | 49 | 25.7 |
| and long-lived workers, longest chain first, second threads on any measured gain | 46 | 26.5 |

The old one-knot-at-a-time path took 5 h 15 min to 5 h 32 min for 43 knots' HFK-hat per process,
four processes side by side. The τ run's last five minutes were two knots' single-core rank and
level jobs.

NUMA contention (`tools/gen_contention_probe.py`, five runs; x is per-state time against one
worker alone, 15 µs per state for a generation slice and 1.7 µs for a histogram slice):

| Workers at once | Generation slice | Histogram slice |
|---|---|---|
| 8, one node's physical cores | x1.4 – 1.6 | x1.0 – 1.1 |
| 16, both nodes' physical cores | x1.0 – 1.1 | x1.0 – 1.2 |
| 32, every logical CPU | x1.4 – 2.0 | x2.0 – 3.5 |

- Spreading one calculation's jobs over both nodes costs nothing measurable.
- A second hyperthread adds throughput to generation and loses it on histogram slices: 32
  histogram workers do less than 16. The planner now decides this per kind (above).
- Not explained: 8 workers on one node run slower than the same node's 8 workers when the other
  node is also busy, and node 1 runs generation slower than node 0 (19 – 24 against 15 µs). The
  host runs other guests on these sockets and scales its clock (1.2 to 3.6 GHz), so both may be
  the host rather than the work.

## Not scheduled yet

- A Khovanov, Lee or Rasmussen calculation is one job; its build and per-grading reductions are
  not split.
- tau's rank and level search are two long single jobs on the set-based reducer.
- Oracle validation calls run in the caller.
- `compute()` takes one knot; only the Floer engine has many-knot entry points.
