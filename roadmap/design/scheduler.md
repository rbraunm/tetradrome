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

CPU jobs run in a pool of long-lived workers. A worker takes one job after another: it pins
itself to the job's CPUs, runs it, publishes the result and waits. One left holding more than
128 MiB after a job is retired. A job is charged for what it adds to its worker; what the
workers hold between jobs is charged to the pool. The engines' session keeps one pool for the life of the process, so a calculation
after the first finds workers waiting; between runs it keeps idle workers only up to half the
admission margin of memory. A Scheduler used on its own makes a pool for its run. Workers are forked from a template process that has already imported numpy
and bound the compiled grid kernels (`engines/floer/prewarm.py`), so a worker starts ready. On
Windows workers are spawned cold and each is charged a measured import baseline.

## What is measured

Each handler has a speed and a cost. A benchmark measures them once for a machine and its software
(`tetradrome/calibrate.py`), and every calculation after that starts out with them:

- **work rate**: per kind and device, a fixed part per job and seconds per cost unit, fitted to
  the benchmark's jobs so that small ones are predicted as well as large ones. The inline path
  and both CPU placements run the same code, so they share a rate.
- **sharing a core**: per kind, a job's time when a physical core's two hyperthreads both run it,
  over its time with the core to itself. Measured with two pinned workers released together, on
  one core while the rest of the machine is idle.
- **dispatch overhead**: seconds to hand a job with nothing to do to a waiting worker and reap
  it, how much of that is the scheduler's own time, and what starting a worker adds when none is
  waiting (2 ms, 1.3 ms and 20 ms on the container).
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

- A fixed part and one rate per kind. A kind whose time per unit changes with its size is still
  predicted roughly.
- GPU rates are not benchmarked; there has been no GPU host to build that on.
- A benchmark taken while the machine is busy stores skewed figures (the shared-core factors come
  out near 1) until the code changes or `python -m tetradrome.calibrate` is run on a quiet machine.
- Two processes that both find no figures both run the benchmark, and slow each other's.
- On Windows the benchmark takes minutes, because every worker there is a spawned process.

## The planner

For each ready job (`scheduler/placement.py`):

1. Drop the paths this machine can never serve. If none is left the job is infeasible, reported
   before the run starts, and only its own component is abandoned.
2. Choose among the rest by measurement. Inline, when the job is predicted to be over sooner
   than handing it to a worker would cost: the scheduler's own hand-over time while other jobs
   wait on its loop, or the whole dispatch, with a worker's start when none is idle, for a job
   that is the entire graph. Otherwise the worker path with the smallest predicted run plus
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
- Offers each job a worker path and the inline path, so a small calculation runs without a
  worker. Sizes its slices by the calculation, never by the machine: at least 64 slices where the
  states allow, each of 1,024 to 32,768 states.
- Cuts its work where the mathematics is independent, so no long job stands alone. HFK-hat's
  slices are split by Alexander grading, each grading is merged by its own job, and each boundary
  map's rank is its own job; no job holds more than one grading.
- Keeps a serial pure-Python form of each calculation as the oracle its scheduled form is tested
  against (`grid_complexes`, `grading_histogram_reference`, `tau_reference`).

## Measured

On a 32-CPU, two-socket container, end to end, histogram included:

| | HFK-hat | τ |
|---|---|---|
| one knot of grid number 10 (3.6 million states) | 15 s | 24 s |
| grid number 8, repeated in one process | 240 ms | 105 ms |
| grid number 7 | 70 ms | 22 ms |
| grid number 5 | 14 ms | 2 ms |

| | Seconds |
|---|---|
| handing a job to a waiting worker | 0.003 |
| of which the scheduler's own time | 0.0008 |
| starting a worker | 0.018 |
| a determinant through `compute()`, against 0.04 ms called directly | 0.0003 |

The 185 roster knots of grid number up to 10, as one `run_many` graph each:

| | Seconds |
|---|---|
| HFK-hat for all 185 | 1,036 |
| τ for all 185 | 161 |

A batch of six grid-number-10 knots, HFK-hat with the histograms already known:

| | Seconds | CPUs busy, average |
|---|---|---|
| generation slices of 4096 states | 179 | 9.3 |
| slices of 32768 states | 139 | 9.8 |
| and results kept out of the scheduler's process | 86 | 13.4 |
| and a merge per Alexander grading, a rank per boundary map | 49 | 25.7 |
| and long-lived workers, longest chain first, second threads on any measured gain | 46 | 26.5 |

The whole heavy tier (`pytest --heavy`, 1,234 tests) takes 59 minutes, 27 of them two
single-threaded comparisons of the compiled kernels with the pure-Python reference at grid
number 10.

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

## Open

- The scheduler's own time is about 0.8 ms per job. A knot of grid number 8 is about 150 jobs, so
  most of its 240 ms is that. Grouping a grading's small boundary maps into one job would cut it.
- A Khovanov, Lee or Rasmussen calculation is one job; its build and per-grading reductions are
  not split.
- τ's rank and level search are one job each, on the compiled sparse reducer.
- Oracle validation calls run in the caller.
- `compute()` takes one knot; only the Floer engine has many-knot entry points.
- A result's provenance does not yet record how its jobs ran: which reducer each reduction used
  (reference, compiled, sparse or GPU) and where the scheduler placed each job. The run report
  holds it; passing it into the result is the next step of the audit trail.
- A Floer calculation's grid is built in the caller before its graph is scheduled
  (`homology-engine.md` section 9).
