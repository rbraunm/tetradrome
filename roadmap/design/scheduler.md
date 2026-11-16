# The scheduler

**One rule: every calculation is a scheduler job.** An engine describes its work as jobs; the
scheduler decides where and when each one runs. A calculation done in the caller's process,
outside a job, is a defect: nothing can place it, bound its memory, or run it beside other work.

Code: `src/tetradrome/scheduler/`. Engines enter through `scheduler.run_to_result`.

## Jobs

A job is a callable with its inputs, the jobs it depends on, and:

- **paths**: the ways it can run, each with the cores, RAM and VRAM it needs there;
- **cost**: its predicted work, in the unit of its kind (states for a slice, operations for a
  reduction);
- **kind**: the family of jobs that share that unit, so measurements of one family never
  predict another.

A job's result is held until its last consumer has it. A result can be partitioned into shards
so a consumer takes only its part, and a large result with several consumers is placed in shared
memory once per NUMA node.

## Handlers

| Path | Runs in | Costs | For |
|---|---|---|---|
| inline | the scheduler's own process | nothing to dispatch; holds up the loop while it runs | work shorter than a dispatch |
| CPU pinned | a worker on one NUMA node's cores, its memory first touched there | a fork | most work |
| CPU unpinned | a worker on any cores | a fork, no locality | work too large for one node |
| GPU, warm | the standing worker that holds the device context | queueing behind other warm jobs | small GPU jobs |
| GPU, fresh | its own process and context | creating a context | large GPU jobs |

Workers are forked from a template process that has already imported numpy and bound the
compiled grid kernels (`engines/floer/prewarm.py`), so a worker starts ready. On Windows workers
are spawned cold and each is charged a measured import baseline.

## What is measured

Each handler has a speed and a cost, both measured while jobs run and kept for the life of the
process (`scheduler/routing.py`, shared through `scheduler/session.py`):

- **work rate**: seconds per cost unit, per kind and device. The inline path and both CPU
  placements run the same code, so they share a rate; the GPU has its own.
- **dispatch overhead**: seconds to hand a job to a placement and reap it, beyond its run time.
- **memory**: a worker's private bytes are sampled; the ledger charges the larger of declared and
  measured, and logs a job that exceeds its declaration.

Nothing is stored between processes: a stored figure would drift against changed hardware.

## The planner

For each ready job (`scheduler/placement.py`):

1. Drop the paths this machine can never serve. If none is left the job is infeasible, reported
   before the run starts, and only its own component is abandoned.
2. Choose among the rest by measurement. Inline, when the predicted run time is no more than a
   worker's dispatch overhead. Otherwise the worker path with the smallest predicted run plus
   dispatch time. Where a figure is unmeasured, the order the builder declared.
3. Admit the job if the chosen path's cores and memory are free now, per NUMA node and against
   the global ceiling less a margin; otherwise it waits. A busy path never degrades a job to a
   slower one.

Under memory pressure the largest held results are spilled to disk rather than stalling.

## Failure

A job that raises, or whose worker exits without reporting (killed for memory, or unable to load
the job), fails. Its whole component is abandoned; other components finish. `run_to_result`
raises on any failure, so no partial answer is returned.

## What an engine does

- Builds a graph and calls `run_to_result`.
- Prices every job before it runs. The Floer engine first runs a histogram graph (generators per
  grading) and builds the main graph from its result.
- Keeps a serial pure-Python form of each calculation as the oracle its scheduled form is tested
  against (`grid_complexes`, `grading_histogram_reference`, `tau_reference`).

## Measured

On a 32-CPU, two-socket container, one knot of grid number 10 (3.6 million states):

| | Seconds |
|---|---|
| HFK-hat, end to end | 78 |
| of which: histogram, generation, merge, reduction (worker CPU) | 16, 130, 15, 93 |
| tau | 487 |
| one generation slice of 4096 states, in a ready worker | 0.044 |
| the same in a worker that binds the kernels itself | 0.63 |
| a determinant through `compute()`, against 0.04 ms called directly | 0.0003 |

## Not scheduled yet

- A Khovanov, Lee or Rasmussen calculation is one job; its build and per-grading reductions are
  not split.
- tau's rank and level search are two long single jobs on the set-based reducer.
- Oracle validation calls run in the caller.
- One knot per run: a roster is not submitted as one graph, so its knots do not share the machine.
