# Outreach

How to describe Tetradrome to mathematicians. Existing tools are respected instruments
that Tetradrome validates against and credits. It does not replace them and does not
call them to produce its answers.

## Framing

- A Python-first, reproducible workbench that computes the invariants of smooth
  4-dimensional topology natively and validates each result against KnotInfo and the
  existing tools, with full provenance.
- The first target is a Conway-adjacent workflow: normalize inputs, compute natively,
  validate, record conventions and versions, produce transparent reports.
- It is **not** a claim of a new theorem and **not** an attempt to replace established
  tools.

Avoid: "I solved 4D knot theory in Python." Prefer: "a careful computational workbench
for a narrow, validated slice of the machinery."

## Short email template

Fill in the recipient and signature. The project specification can be attached.

```text
Subject: Tetradrome - a Python workbench for reproducible Conway-adjacent knot
invariant workflows

Dear Professor [name],

I am building a Python project called Tetradrome: a readable, reproducible workbench
that computes the invariants of smooth 4-dimensional topology natively and validates
them against known data.

The initial target is intentionally narrow - a Conway-adjacent workflow that
normalizes knot inputs, computes the invariants natively, validates the results
against KnotInfo and the existing tools, records conventions and versions, and
produces transparent reports. The goal is not to claim a new theorem or replace
established tools, which remain my validation reference, but to build a disciplined
apparatus that can reproduce and explore small examples in a way that is easy to
audit.

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
references, not endorsement.
