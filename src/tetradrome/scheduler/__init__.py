# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

"""A general, resource-aware compute scheduler.

Domain-agnostic and above the math layers: it knows the machine (cores and RAM per NUMA
node, GPUs, the real memory ceiling) and runs submitted jobs against it, memory-bounded and
parallel, with dependencies. The math engines submit work to it rather than each reinventing
parallel + memory-bounded execution. See roadmap for the design.
"""
from .executor import RunReport, Scheduler
from .graph import JobGraph
from .inventory import GPU, Machine, NumaNode, detect_machine
from .job import ComputePath, Job, Placement
from .ledger import Allocation, Ledger
from .placement import Decision, Outcome, Placed, plan_placement

__all__ = [
    "GPU",
    "Allocation",
    "ComputePath",
    "Decision",
    "Job",
    "JobGraph",
    "Ledger",
    "Machine",
    "NumaNode",
    "Outcome",
    "Placed",
    "Placement",
    "RunReport",
    "Scheduler",
    "detect_machine",
    "plan_placement",
]
