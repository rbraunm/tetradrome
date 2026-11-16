# Existing Tools

Licences and roles of the external tools. None of them computes a returned value
(`SPEC.md` section 2). What each provisioned oracle computes is in `backend_matrix.md`. The
versions on a host are reported by `scripts/install_oracles.sh` and recorded in every
result, so no version is pinned here.

## Licences

| Tool | Licence | How it is used |
|---|---|---|
| KnotInfo (`database_knotinfo`) | GPL | data read at run time: name resolution, tabulated diagrams, tabulated answers. The one declared dependency. `roadmap/research/knotinfo.md` |
| `knot_floer_homology` | GPLv2+ | imported in the validation path only |
| Spherogram / SnapPy | GPLv2+ | measured in the benchmark; optional interop. `roadmap/research/backends-pip.md` |
| Regina, Khoca, KnotJob, knotkit, JavaKh, KhoHo, kht++, SageMath | not recorded here | validators or measured oracles, imported in the validation path or run as separate programs |

No third-party source or data is in the tree. The user installs the optional tools
(SPEC 9).

## Known, not provisioned

| Tool | Area | Note |
|---|---|---|
| KnotTheory` / Knot Atlas, FastKh | Mathematica Khovanov and knot tables | historical reference |
| pyknotid | knots as 3D space curves | not used |
| `ykh` (Rust, MIT) | Khovanov and Bar-Natan homology over Z, Q, F2, F3; reduced | candidate oracle |
| `yui-hfk` (Rust, MIT) | grid HFK-hat, genus, tau, Upsilon over F2 | candidate oracle; the only packaged Upsilon oracle found |
