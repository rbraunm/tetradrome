# Outreach

How to talk about Tetradrome to mathematicians (SPEC 21, 22). The stance is
augmentation and peer research, not competition: existing tools are respected
instruments that Tetradrome orchestrates, validates against, and credits.

## Framing

Lead with humility and a narrow, concrete claim:

- A Python-first, reproducible workbench for knot-invariant experiments relevant to
  smooth 4-dimensional topology.
- The first target is a Conway-adjacent workflow: normalize inputs, orchestrate
  existing tools where appropriate, validate against known data, record conventions
  and versions, produce transparent reports.
- Native components and exact-algebra acceleration come later, only where they add
  clear value.
- It is **not** a claim of a new theorem and **not** an attempt to replace
  established tools.

Avoid: "I solved 4D knot theory in Python." Prefer: "a careful computational
workbench for a narrow, validated slice of the machinery."

## Short email template

A template; fill in the recipient and signature. The project specification can be
attached.

```text
Subject: Tetradrome - a Python workbench for reproducible Conway-adjacent knot
invariant workflows

Dear Professor [name],

I am building a Python project called Tetradrome: a readable, reproducible
computational workbench for knot-invariant experiments related to smooth
4-dimensional topology.

The initial target is intentionally narrow - a Conway-adjacent workflow that
normalizes knot inputs, orchestrates existing tools where appropriate, validates
outputs against known data, records conventions and backend versions, and produces
transparent reports. The goal is not to claim a new theorem or replace established
tools, but to build a disciplined apparatus that can reproduce and explore small
examples in a way that is easy to audit.

Longer term, I am interested in adding native Python/CUDA components for
exact-algebra workloads only where that adds clear value.

I have attached the specification in case it is of interest. I would be grateful
for any high-level warning signs, references, or suggestions about where such a
project is most likely to go wrong.

Best regards,
[your name]
```

## Who and what to ask for

The contributor most valued is a topologist (README, "Contributing"): help with
conventions, invariant definitions, the wording of obstruction / claim / strength,
and the trace and concordance machinery. Ask for high-level warning signs and
references, not endorsement. Sustained, high-quality contribution earns maintainer
status; that is the intended path, not the exception.
