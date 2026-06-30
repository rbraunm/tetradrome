#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

# install_oracles.sh -- bring a fresh Tetradrome sandbox to oracle-ready.
#
# The comparison artifact (scripts/comparison/) validates Tetradrome's native results
# against external gold-master oracles. Those oracles are opt-in validators ONLY
# (CLAUDE.md, decision 0006) -- never a runtime dependency of a computation. This script
# installs the ones that are runnable in an ephemeral sandbox so adapter work can be
# developed and checked here, and obtains (but does not build) the ones whose real run
# host is CT 250.
#
# It is the sandbox sibling of scripts/provision_runner.py (which stands up CT 250).
#
# Discovery contract (scripts/comparison/adapters.py):
#   snappy  -> python `import snappy`         (pip-installed)
#   knotjob -> `knotjob` executable on PATH   (a wrapper around `java -jar KnotJob.jar`)
#   sage    -> `sage` executable on PATH      (CT 250 only; sandbox reports absent)
#   khoca   -> `khoca` on PATH or importable  (obtain-only here; sandbox reports absent)
#
# Honest post-state in the sandbox: SnapPy + KnotJob available and smoke-tested;
# Khoca source obtained but not built (its prebuilt is Python-3.6 ABI-locked, dead on a
# modern interpreter); Sage repo reachability confirmed, install left to CT 250. An
# absent oracle is reported as absent by the adapter, never faked.
#
# Idempotent and fail-loud: re-running is safe, and any whitelist regression or failed
# smoke test aborts loudly with a nonzero exit.
#
#   scripts/install_oracles.sh            # full run
#   ORACLE_HOME=/home/claude/oracles scripts/install_oracles.sh
#
# Tunables (environment): ORACLE_HOME (default /opt/oracles), BIN_DIR (default
# /usr/local/bin), TEMURIN_MAJOR (default 25; KnotJob needs Java >= 23).

set -euo pipefail

ORACLE_HOME="${ORACLE_HOME:-/opt/oracles}"
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
TEMURIN_MAJOR="${TEMURIN_MAJOR:-25}"

KNOTJOB_URL="https://www.maths.dur.ac.uk/users/dirk.schuetz/KnotJob.zip"
KHOCA_REPO="https://github.com/LLewark/khoca.git"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Per-oracle outcome, printed in the closing summary.
STATUS_SNAPPY="not attempted"
STATUS_KNOTJOB="not attempted"
STATUS_KHOCA="not attempted"
STATUS_SAGE="not attempted"

log() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# ---- reachability preflight --------------------------------------------------
# A blocked domain comes back from the egress proxy with an x-deny-reason header.
# Catch a whitelist regression in seconds rather than mid-install.

probe() {
  # probe <url> -> 0 reachable, 1 blocked, 2 no HTTP response
  local url="$1" headers
  headers="$(curl -sS -L --max-time 30 -o /dev/null -D - "$url" 2>/dev/null)" || return 2
  if printf '%s' "$headers" | grep -qi 'x-deny-reason'; then
    printf '%s' "$headers" | grep -i 'x-deny-reason' >&2
    return 1
  fi
  printf '%s' "$headers" | grep -qiE '^HTTP/' || return 2
  return 0
}

preflight() {
  log "Reachability preflight"
  local blocked=0 url rc
  # The artifact-bearing domains the rest of the script depends on. lewark.de is
  # only a redirect stub to the ETH page, so it is intentionally not gated on here.
  local urls="
    ${KNOTJOB_URL}
    https://packages.adoptium.net/artifactory/api/gpg/key/public
    https://pypi.org/simple/snappy/
    https://files.pythonhosted.org/
    http://deb.debian.org/debian/dists/bookworm/Release
    http://security.debian.org/debian-security/dists/bookworm-security/Release
    https://github.com/LLewark/khoca
    https://codeload.github.com/LLewark/khoca/tar.gz/refs/heads/master
    https://people.math.ethz.ch/~llewark/khoca.php
  "
  for url in ${urls}; do
    set +e
    probe "${url}"
    rc=$?
    set -e
    case "${rc}" in
      0) info "ok       ${url}" ;;
      1) info "BLOCKED  ${url}"; blocked=$((blocked + 1)) ;;
      2) info "NO HTTP  ${url}"; blocked=$((blocked + 1)) ;;
    esac
  done
  if [ "${blocked}" -ne 0 ]; then
    die "${blocked} domain(s) not reachable. Whitelist them and retry in a NEW conversation
      (egress whitelist changes only take effect on a fresh sandbox)."
  fi
}

# ---- Temurin JDK (KnotJob needs Java >= 23) ----------------------------------

install_temurin() {
  log "Temurin ${TEMURIN_MAJOR} JDK"
  local pkg="temurin-${TEMURIN_MAJOR}-jdk"
  if dpkg -s "${pkg}" >/dev/null 2>&1; then
    info "${pkg} already installed"
    return
  fi
  . /etc/os-release
  curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
    | gpg --dearmor | ${SUDO} tee /usr/share/keyrings/adoptium.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb ${VERSION_CODENAME} main" \
    | ${SUDO} tee /etc/apt/sources.list.d/adoptium.list >/dev/null
  ${SUDO} apt-get update -qq
  DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y -qq "${pkg}" >/dev/null
  info "installed $(java -version 2>&1 | head -1)"
}

# ---- KnotJob: download jar + put a `knotjob` wrapper on PATH ------------------

install_knotjob() {
  log "KnotJob"
  local dir="${ORACLE_HOME}/knotjob"
  ${SUDO} mkdir -p "${dir}"
  # Rolling artifact: always fetch fresh so a new session gets the current build.
  curl -fsSL -o /tmp/KnotJob.zip "${KNOTJOB_URL}"
  ${SUDO} unzip -o -q /tmp/KnotJob.zip 'KnotJob/KnotJob.jar' -d "${dir}"
  local jar="${dir}/KnotJob/KnotJob.jar"
  [ -f "${jar}" ] || die "KnotJob.jar not found in the downloaded archive."

  # The adapter discovers KnotJob via `shutil.which("knotjob")`, so expose a wrapper
  # on PATH. KnotJob's CLI is `java -jar KnotJob.jar <files> <commands>`; no GUI when
  # args are present. Heap left at the JVM default (sandbox is small); raise with
  # -Xmx on the real run host if a knot needs it.
  ${SUDO} tee "${BIN_DIR}/knotjob" >/dev/null <<EOF
#!/usr/bin/env bash
exec java -jar "${jar}" "\$@"
EOF
  ${SUDO} chmod +x "${BIN_DIR}/knotjob"
  command -v knotjob >/dev/null || die "knotjob wrapper not on PATH (${BIN_DIR})."
  info "jar at ${jar}"
  info "wrapper at ${BIN_DIR}/knotjob"
}

smoke_knotjob() {
  log "KnotJob smoke test (trefoil)"
  local work="/tmp/knotjob-smoke"
  rm -rf "${work}"; mkdir -p "${work}"
  # Right-handed trefoil PD. The parser keeps only digits and commas, so the
  # bracketed form is fine verbatim.
  printf 'PD[X[1,4,2,5],X[3,6,4,1],X[5,2,6,3]]\n' > "${work}/trefoil.txt"
  ( cd "${work}" && knotjob trefoil.txt -kb0 -s0 >/dev/null 2>&1 )
  local out="${work}/trefoil.txt_s0_kb0"
  [ -f "${out}" ] || die "KnotJob wrote no output file -- run path broken."
  # The trefoil's s-invariant is +/-2 (sign tracks chirality of the input PD).
  if grep -qE 'S-Invariant mod 0 : -?2$' "${out}"; then
    info "s-invariant = $(grep -oE '\-?2$' "${out}" | head -1)  (expected magnitude 2: PASS)"
  else
    info "--- output ---"; cat "${out}" >&2
    die "KnotJob smoke FAILED: trefoil s-invariant not +/-2."
  fi
  STATUS_KNOTJOB="installed + smoke PASS (runnable)"
}

# ---- SnapPy: pip install + hyperbolic-volume smoke ---------------------------

install_snappy() {
  log "SnapPy"
  # The topology tool `snappy`, NOT python-snappy. pip is idempotent. This also
  # pulls knot_floer_homology (the kfh oracle) in as a dependency.
  pip install --break-system-packages -q snappy
  info "$(python -c 'import snappy; print("snappy", snappy.__version__)')"
}

smoke_snappy() {
  log "SnapPy smoke test (figure-eight volume)"
  python - <<'PY' || die "SnapPy smoke FAILED."
import snappy
v = snappy.Manifold("4_1").volume()
expected = 2.0298832128
assert abs(v - expected) < 1e-6, f"4_1 volume {v} != {expected}"
print(f"   4_1 volume = {v}  (expected {expected}: PASS)")
PY
  STATUS_SNAPPY="installed + smoke PASS (runnable)"
}

# ---- Khoca: obtain source only (build is a CT 250 / Docker concern) ----------

obtain_khoca() {
  log "Khoca (obtain source only)"
  local dir="${ORACLE_HOME}/khoca/src"
  ${SUDO} mkdir -p "${ORACLE_HOME}/khoca"
  if [ -d "${dir}/.git" ]; then
    info "source present; refreshing"
    ${SUDO} git -C "${dir}" pull --ff-only --quiet
  else
    ${SUDO} git clone --depth 1 --quiet "${KHOCA_REPO}" "${dir}"
  fi
  [ -f "${dir}/khoca.py" ] || die "khoca.py missing from clone -- source not obtained."
  info "source at ${dir}"
  info "NOT built: prebuilt binary is Python-3.6 ABI-locked; build from source or Docker on CT 250."
  STATUS_KHOCA="source obtained (build = CT 250)"
}

# ---- Sage: reachability only (multi-GB apt tree installs on CT 250) ----------

check_sage() {
  log "SageMath (Debian repo reachability only)"
  # Confirmed in preflight; just state the position. bookworm is now oldstable, so
  # verify the sagemath candidate when provisioning CT 250.
  info "deb.debian.org reachable; install is CT 250 work (do not apt-install Sage here)."
  STATUS_SAGE="repo reachable (install = CT 250)"
}

# ---- summary -----------------------------------------------------------------

summary() {
  log "Summary"
  printf '   %-9s %s\n' "snappy"  "${STATUS_SNAPPY}"
  printf '   %-9s %s\n' "knotjob" "${STATUS_KNOTJOB}"
  printf '   %-9s %s\n' "khoca"   "${STATUS_KHOCA}"
  printf '   %-9s %s\n' "sage"    "${STATUS_SAGE}"
  printf '\n   ORACLE_HOME=%s\n' "${ORACLE_HOME}"
  info "Verify what the adapter sees:  python -c \"from scripts.comparison import adapters\""
}

main() {
  preflight
  install_temurin
  install_knotjob
  smoke_knotjob
  install_snappy
  smoke_snappy
  obtain_khoca
  check_sage
  summary
  log "Oracle setup complete."
}

main "$@"
