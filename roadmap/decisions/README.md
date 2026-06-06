# Decision Records

Short records of choices that must be settled before code, because the code keys
off them. Format: Status / Context / Decision / Consequences. These are
deliberately reversible where the SPEC says so; reversibility is noted per record.

- `0001-canonical-invariant-names.md` -- the canonical names every result and
  export uses (SPEC 12.4). Gates the normalizer.
- `0002-python-target.md` -- supported Python versions.
- `0003-native-coefficient-field.md` -- the field the first native engine works over.
- `0004-validate-by-default-error-policy.md` -- validate-by-default and the loud
  error set.
