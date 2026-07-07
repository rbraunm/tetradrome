#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Randy Braunm

# install_oracles.sh -- bring a host to oracle-ready for the Tetradrome comparison artifact.
#
# The comparison artifact (scripts/comparison/) validates Tetradrome's native results against
# external gold-master oracles, and deliberately runs EVERY overlapping engine on the same
# invariant so the artifact can compare wall-time and exact output (decision 0006: oracles are
# opt-in validators only, never a runtime dependency of a computation). This script installs and
# builds all of them and smoke-gates each, so those comparisons never have to be re-derived by
# hand in a later session.
#
# It is portable across the two hosts we use and is the single source of truth for the recipes:
#   * the ephemeral sandbox (Ubuntu noble) -- run it directly;
#   * CT 250 (Debian bookworm) -- scripts/provision_runner.py runs THIS SAME script inside the
#     LXC with PIP pointed at the container venv and INSTALL_SAGE=1 (SageMath is bookworm-only).
#
# Every apt line survives a base image that carries an unrelated broken third-party apt source
# (the noble sandbox ships a dead NodeSource list): the apt index refresh is scoped to exactly
# the sources this script relies on (the base distro + the Adoptium repo we add), so a rogue
# source elsewhere cannot break it -- while a genuine failure of a source we DO need still aborts.
#
# Discovery contract (scripts/comparison/adapters.py) -- import for python packages, a PATH
# executable (a wrapper) for the compiled tools:
#   kfh      import knot_floer_homology     (pip; also pulled by snappy)
#   snappy   import snappy                   (pip)
#   regina   import regina                   (pip wheel: cp312 noble / cp311 bookworm)
#   khoca    import khoca                    (pip wheel + runtime deps: py sympy)
#   knotjob  `knotjob`  -> java -jar KnotJob.jar
#   javakh   `javakh`   -> java -cp ... org.katlas.JavaKh.JavaKh   (built from source)
#   khtpp    `khtpp`    -> the kht++ binary                        (built from source)
#   knotkit  `kk`       -> the knotkit binary                      (built from source)
#   khoho    `khoho`    -> gp with KhoHo preloaded                 (built from source)
#   sage     `sage`     (apt sagemath; INSTALL_SAGE=1 -- CT 250 only, multi-GB apt tree)
#
# Empirically-derived build recipes captured here (why each deviation from upstream defaults):
#   javakh   upstream build.sh is patched for JDK 17, but it compiles clean under Temurin 25
#            (only finalize() deprecation warnings), so one JDK covers both javakh and KnotJob.
#   kht++    upstream targets clang; g++ 12/13 needs `#include <cstdint>` in Coefficients.h for
#            int_fast64_t, and the header path for Eigen is the system one, not ../libraries/Eigen.
#   knotkit  ships bison-2.4.3-generated parser files; the system bison 3.x regenerates an
#            incompatible parser, so we RESTORE the shipped parser and touch it newer than the
#            grammar to keep make from regenerating, and build with g++ + system GMP (the Makefile
#            hard-codes clang/libc++ and MacPorts paths).
#   khoho    builds .so modules against system PARI; KhoHo_gvars calls reset_all() at load time,
#            but reset_all is defined in KhoHo_data and needs MAX_DIAGRAM_NUM from the main KhoHo
#            file, so the loader pre-seeds DStore as a vector (the guard then skips that premature
#            reset_all), reads every module, then calls reset_all() once at the end. gp must run
#            from the source dir (the install() paths to the .so are relative).
#
# Idempotent, update-aware, and fail-loud. Re-running converges: an oracle already present at the
# current upstream is left alone (no recompile), an oracle with an upstream update is rebuilt, and a
# missing oracle is installed. A converged re-run does no source build and no network apt call and
# ends with an unambiguous "good to proceed" line, so a later session knows the host is ready without
# re-running or troubleshooting. Any whitelist regression or failed smoke aborts loudly with a nonzero
# exit. This script is a first-class project artifact (ADR 0013): it is how any consumer -- not just
# our sandbox -- keeps their oracles current, and it reports the exact version of every oracle it
# converges to, because the recorded version is the reproducibility (ADR 0013).
#
#   scripts/install_oracles.sh              # converge to present + current, apply updates, verify
#   scripts/install_oracles.sh --check      # dry run: report each oracle's version and update state,
#                                           #          change nothing (exit nonzero only if any missing)
#   VERIFY=1 scripts/install_oracles.sh     # additionally re-run every smoke even when converged
#   ORACLE_HOME=/home/claude/oracles scripts/install_oracles.sh
#   PIP="/opt/tetradrome/venv/bin/pip" PIP_INSTALL_FLAGS="" PY="/opt/tetradrome/venv/bin/python" \
#       INSTALL_SAGE=1 scripts/install_oracles.sh    # how provision_runner.py invokes it on CT 250
#
# Tunables (environment): ORACLE_HOME (default /opt/oracles), BIN_DIR (default /usr/local/bin),
# TEMURIN_MAJOR (default 25; KnotJob needs Java >= 23), PIP (default 'pip'; a venv pip on CT 250),
# PIP_INSTALL_FLAGS (default '--break-system-packages'; '' for a venv pip), PY (default 'python';
# the venv python on CT 250), INSTALL_SAGE (default 0), VERIFY (default 0), and the --check flag.

set -euo pipefail

# A guaranteed-present UTF-8 locale, so apt maintainer scripts (perl, apt-listchanges) stop
# emitting "Setting locale failed ... falling back to C" noise. C.UTF-8 always exists; stderr is
# otherwise left intact so a genuine apt warning still surfaces.
export LC_ALL=C.UTF-8 LANG=C.UTF-8

ORACLE_HOME="${ORACLE_HOME:-/opt/oracles}"
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
TEMURIN_MAJOR="${TEMURIN_MAJOR:-25}"
PIP="${PIP:-pip}"
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---break-system-packages}"
PY="${PY:-python}"
INSTALL_SAGE="${INSTALL_SAGE:-0}"
DEBUG="${DEBUG:-}"          # non-empty: stream each source build live instead of hiding it in a log
VERIFY="${VERIFY:-}"        # non-empty: re-run every smoke even for oracles that were already current
CHECK_ONLY=0               # set by --check: report each oracle's version and update state, change nothing

KNOTJOB_URL="https://www.maths.dur.ac.uk/users/dirk.schuetz/KnotJob.zip"
JAVAKH_REPO="https://github.com/geometer/JavaKh-v2.git"
KHTPP_REPO="https://github.com/cbz20/khtpp.git"
KHOHO_REPO="https://github.com/AShumakovitch/KhoHo.git"
KNOTKIT_REPO="https://github.com/cseed/knotkit.git"

# Trefoil PD (right-handed); the parser in each tool keeps only digits/commas, so the bracketed
# form is fine verbatim. Used by the KnotJob and JavaKh smokes.
TREFOIL_PD='PD[X[1,4,2,5],X[3,6,4,1],X[5,2,6,3]]'

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Per-oracle outcome + resolved version, printed in the closing summary. STATUS is what happened this
# run (installed / built / up to date / updated); VERSION is the exact version we converged to and is
# the value that matters for reproducibility (ADR 0013).
STATUS_KFH="not attempted";     VERSION_KFH="-"
STATUS_SNAPPY="not attempted";  VERSION_SNAPPY="-"
STATUS_REGINA="not attempted";  VERSION_REGINA="-"
STATUS_KHOCA="not attempted";   VERSION_KHOCA="-"
STATUS_KNOTJOB="not attempted"; VERSION_KNOTJOB="-"
STATUS_JAVAKH="not attempted";  VERSION_JAVAKH="-"
STATUS_KHTPP="not attempted";   VERSION_KHTPP="-"
STATUS_KNOTKIT="not attempted"; VERSION_KNOTKIT="-"
STATUS_KHOHO="not attempted";   VERSION_KHOHO="-"
STATUS_SAGE="not attempted";    VERSION_SAGE="-"

# Per-oracle "acted this run" flags: a smoke runs when its oracle was installed/built/updated this run
# (or when VERIFY is set). A converged oracle is left un-smoked so a no-change re-run stays fast.
ACT_PIP=0; ACT_KNOTJOB=0; ACT_JAVAKH=0; ACT_KHTPP=0; ACT_KNOTKIT=0; ACT_KHOHO=0
ACTED=0                    # any oracle installed/built/updated this run (drives the closing verdict)

log() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# ---- version probes: one source of truth for the summary and (later) runtime provenance ----------
# Each returns the resolved version string, or "absent" when the oracle is not installed. These never
# mutate anything, so they are safe to call in --check.

pip_version() {
  # pip_version <distribution> -> installed version | "absent"
  ${PY} - "$1" <<'PY' 2>/dev/null || echo absent
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("absent")
PY
}

git_version() {
  # git_version <dir> -> short HEAD sha | "absent"   (the version of a source-built oracle)
  [ -d "$1/.git" ] || { echo absent; return; }
  ${SUDO} git -C "$1" rev-parse --short HEAD 2>/dev/null || echo absent
}

file_hash() {
  # file_hash <path> -> 12-char sha256 | "absent"    (the version of the rolling KnotJob jar)
  [ -f "$1" ] || { echo absent; return; }
  sha256sum "$1" 2>/dev/null | cut -c1-12
}

git_remote_current() {
  # git_remote_current <dir> -> 0 if local HEAD matches upstream, 1 if an update is available.
  # Non-mutating (ls-remote, no fetch); used only by --check.
  local dir="$1" localsha remotesha
  localsha="$(${SUDO} git -C "${dir}" rev-parse HEAD 2>/dev/null)" || return 1
  remotesha="$(${SUDO} git -C "${dir}" ls-remote origin HEAD 2>/dev/null | awk '{print $1}')" || return 0
  [ -n "${remotesha}" ] && [ "${localsha}" != "${remotesha}" ] && return 1
  return 0
}

want_smoke() {
  # want_smoke <act-flag-value> -> 0 (smoke) if the oracle acted this run or VERIFY is set.
  [ "$1" = "1" ] || [ -n "${VERIFY}" ]
}

# Run a source build quietly: a clean build prints nothing here (only the caller's section header
# and its "wrapper at" line remain), while a failed build dumps the log tail and aborts. DEBUG
# streams it live instead. <command> is eval'd, so the caller's in-scope vars (dir, GMP_INC,
# PARI_INC, SUDO) resolve. The thousands of compiler warnings from these third-party trees are the
# bulk of what this hides; a real error still surfaces via the tail and the nonzero exit.
quiet_build() {
  # quiet_build <description> <command>
  local desc="$1" cmd="$2" logpath
  if [ -n "${DEBUG}" ]; then
    eval "${cmd}"
    return
  fi
  logpath="$(mktemp)"
  if ! eval "${cmd}" >"${logpath}" 2>&1; then
    tail -n 40 "${logpath}" >&2
    die "${desc} failed -- full build log at ${logpath}."
  fi
  rm -f "${logpath}"
}

# OS identity (VERSION_CODENAME picks the Adoptium suite; ID picks the base apt mirror to probe).
. /etc/os-release
OS_ID="${ID:-}"
OS_CODENAME="${VERSION_CODENAME:-}"
[ -n "${OS_CODENAME}" ] || die "cannot determine VERSION_CODENAME from /etc/os-release."

# ---- reachability preflight --------------------------------------------------
# A blocked domain comes back from an egress proxy with an x-deny-reason header. Catch a
# whitelist regression in seconds rather than mid-install. One retry on a transient no-HTTP so a
# blip on a healthy host does not abort (a genuine block returns the deny header every time).

probe() {
  # probe <url> -> 0 reachable, 1 blocked, 2 no HTTP response
  local url="$1" headers
  headers="$(curl -sS -L --max-time 30 -o /dev/null -D - "$url" 2>/dev/null)" || return 2
  if printf '%s' "${headers}" | grep -qi 'x-deny-reason'; then
    printf '%s' "${headers}" | grep -i 'x-deny-reason' >&2
    return 1
  fi
  printf '%s' "${headers}" | grep -qiE '^HTTP/' || return 2
  return 0
}

preflight() {
  log "Reachability preflight"
  local base_apt
  case "${OS_ID}" in
    ubuntu) base_apt="http://archive.ubuntu.com/ubuntu/dists/${OS_CODENAME}/Release" ;;
    debian) base_apt="http://deb.debian.org/debian/dists/${OS_CODENAME}/Release" ;;
    *) die "unsupported base OS '${OS_ID}' (expected ubuntu or debian)." ;;
  esac
  local urls="
    ${base_apt}
    https://packages.adoptium.net/artifactory/api/gpg/key/public
    https://pypi.org/simple/snappy/
    https://files.pythonhosted.org/
    ${KNOTJOB_URL}
    https://github.com/geometer/JavaKh-v2
    https://codeload.github.com/cbz20/khtpp/tar.gz/refs/heads/main
  "
  local blocked=0 url rc
  for url in ${urls}; do
    set +e
    probe "${url}"; rc=$?
    if [ "${rc}" -eq 2 ]; then sleep 2; probe "${url}"; rc=$?; fi   # one retry on transient
    set -e
    case "${rc}" in
      0) info "ok       ${url}" ;;
      1) info "BLOCKED  ${url}"; blocked=$((blocked + 1)) ;;
      2) info "NO HTTP  ${url}"; blocked=$((blocked + 1)) ;;
    esac
  done
  if [ "${blocked}" -ne 0 ]; then
    die "${blocked} domain(s) not reachable. Whitelist them and retry in a NEW conversation
      (an egress whitelist change only takes effect on a fresh sandbox)."
  fi
}

# ---- apt: Adoptium repo + build deps, with a scoped index refresh ------------

apt_setup() {
  log "apt: Adoptium repo + build dependencies (Temurin ${TEMURIN_MAJOR}, GMP, Eigen, PARI, bison/flex)"
  # Add the Adoptium key + source (idempotent overwrite).
  curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
    | gpg --dearmor | ${SUDO} tee /usr/share/keyrings/adoptium.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb ${OS_CODENAME} main" \
    | ${SUDO} tee /etc/apt/sources.list.d/adoptium.list >/dev/null

  # Refresh ONLY the sources we depend on -- the base distro sources plus Adoptium -- via a
  # temp sources dir, so an unrelated broken source in the base image cannot fail the update.
  local tmp_src copied=0 f
  tmp_src="$(mktemp -d)"
  for f in /etc/apt/sources.list \
           /etc/apt/sources.list.d/ubuntu.sources \
           /etc/apt/sources.list.d/debian.sources; do
    if [ -s "${f}" ]; then ${SUDO} cp "${f}" "${tmp_src}/"; copied=$((copied + 1)); fi
  done
  ${SUDO} cp /etc/apt/sources.list.d/adoptium.list "${tmp_src}/"
  [ "${copied}" -gt 0 ] || die "no base apt sources found to refresh (looked for sources.list, ubuntu.sources, debian.sources)."
  ${SUDO} apt-get update -qq \
    -o Dir::Etc::sourceparts="${tmp_src}" \
    -o Dir::Etc::sourcelist="/dev/null" \
    -o APT::Get::List-Cleanup="0"
  ${SUDO} rm -rf "${tmp_src}"

  local pkgs="ca-certificates curl gnupg unzip git build-essential bison flex \
    libgmp-dev libgmpxx4ldbl libeigen3-dev libpari-dev pari-gp temurin-${TEMURIN_MAJOR}-jdk"
  if [ "${INSTALL_SAGE}" = "1" ]; then
    # bookworm is oldstable; verify the candidate exists rather than trusting the name.
    if ${SUDO} apt-cache policy sagemath 2>/dev/null | grep -q 'Candidate: [0-9]'; then
      pkgs="${pkgs} sagemath"
    else
      die "INSTALL_SAGE=1 but no 'sagemath' candidate in the apt sources for ${OS_CODENAME}."
    fi
  fi
  DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y -qq ${pkgs} >/dev/null
  command -v java >/dev/null || die "java not on PATH after install."
  command -v g++  >/dev/null || die "g++ not on PATH after install."
  command -v gp   >/dev/null || die "gp (PARI/GP) not on PATH after install."
  info "installed $(java -version 2>&1 | head -1)"
  detect_headers
}

detect_headers() {
  # Multiarch header dirs for the source builds (parent dir so <gmp.h>/<pari/pari.h> resolve).
  # Pure dpkg lookups, no network -- safe to call whether or not apt_setup ran this session.
  GMP_INC="$(dirname "$(dpkg -L libgmp-dev  | grep -m1 '/gmp\.h$')")"
  PARI_INC="$(dirname "$(dirname "$(dpkg -L libpari-dev | grep -m1 '/pari\.h$')")")"
  [ -d "${GMP_INC}" ]  || die "could not locate gmp.h include dir."
  [ -d "${PARI_INC}" ] || die "could not locate pari.h include dir."
  info "gmp include: ${GMP_INC}   pari include: ${PARI_INC}"
}

apt_ready() {
  # True when the toolchain the source builds need is already present, so a converged re-run makes
  # no network apt call. Checks the commands plus the -dev packages that carry the build headers.
  command -v java >/dev/null && command -v g++ >/dev/null && command -v gp >/dev/null || return 1
  dpkg -s libgmp-dev  >/dev/null 2>&1 && dpkg -s libpari-dev >/dev/null 2>&1 || return 1
  if [ "${INSTALL_SAGE}" = "1" ]; then command -v sage >/dev/null || return 1; fi
  return 0
}

ensure_apt() {
  if apt_ready; then
    log "apt: toolchain already present -- skipping index refresh and install"
    detect_headers
  else
    apt_setup
  fi
}

# ---- python oracles: kfh (via snappy), snappy, regina, khoca -----------------

install_pip_oracles() {
  log "pip oracles: snappy (pulls kfh), regina, khoca (+ py sympy) -- applying any updates"
  # Apply updates rather than pin (ADR 0013): -U upgrades to the current release; the version we end
  # up on is recorded below and is what makes the run reproducible. Capture versions before and after
  # so a run that changes nothing skips the (otherwise wasted) smoke.
  local before_kfh before_snappy before_regina before_khoca
  before_kfh="$(pip_version knot_floer_homology)"
  before_snappy="$(pip_version snappy)"
  before_regina="$(pip_version regina)"
  before_khoca="$(pip_version khoca)"
  ${PIP} install ${PIP_INSTALL_FLAGS} -q -U snappy
  ${PIP} install ${PIP_INSTALL_FLAGS} -q -U regina khoca py sympy
  VERSION_KFH="$(pip_version knot_floer_homology)"
  VERSION_SNAPPY="$(pip_version snappy)"
  VERSION_REGINA="$(pip_version regina)"
  VERSION_KHOCA="$(pip_version khoca)"
  for v in "${VERSION_KFH}" "${VERSION_SNAPPY}" "${VERSION_REGINA}" "${VERSION_KHOCA}"; do
    [ "${v}" = "absent" ] && die "pip oracle missing after install (kfh=${VERSION_KFH} snappy=${VERSION_SNAPPY} regina=${VERSION_REGINA} khoca=${VERSION_KHOCA})."
  done
  if [ "${before_kfh}:${before_snappy}:${before_regina}:${before_khoca}" \
     = "${VERSION_KFH}:${VERSION_SNAPPY}:${VERSION_REGINA}:${VERSION_KHOCA}" ] \
     && [ "${before_snappy}" != "absent" ]; then
    STATUS_KFH="up to date"; STATUS_SNAPPY="up to date"
    STATUS_REGINA="up to date"; STATUS_KHOCA="up to date"
  else
    STATUS_KFH="installed/updated"; STATUS_SNAPPY="installed/updated"
    STATUS_REGINA="installed/updated"; STATUS_KHOCA="installed/updated"
    ACT_PIP=1; ACTED=1
  fi
}

smoke_pip_oracles() {
  log "pip oracle smokes (kfh import, SnapPy fig-8 volume, regina Jones, khoca trefoil sl2)"
  ${PY} - <<'PY' || die "pip-oracle smoke FAILED."
import importlib
# kfh
importlib.import_module("knot_floer_homology")
print("   kfh: import knot_floer_homology OK")
# SnapPy figure-eight volume
import snappy
v = snappy.Manifold("4_1").volume(); expected = 2.0298832128
assert abs(v - expected) < 1e-6, f"4_1 volume {v} != {expected}"
print(f"   snappy: 4_1 volume = {v} (PASS)")
# regina trefoil Jones
import regina
j = str(regina.ExampleLink.trefoil().jones())
assert j == "-x^8 + x^6 + x^2", f"regina trefoil jones {j!r}"
print(f"   regina: trefoil jones = {j} (PASS)")
# khoca trefoil sl2 (reduced part decodes to t^-3 q^8 + t^-2 q^6 + t^0 q^2)
import khoca
raw = str(khoca.InteractiveCalculator(0, "0.0", 0)("braidaaa"))
for tok in ("[-3, 8", "[-2, 6", "[0, 2"):
    assert tok in raw, f"khoca trefoil sl2 missing {tok}: {raw[:120]}"
print("   khoca: trefoil sl2 reduced part = t^-3 q^8 + t^-2 q^6 + t^0 q^2 (PASS)")
PY
  STATUS_KFH="installed + smoke PASS (import, wired run)"
  STATUS_SNAPPY="installed + smoke PASS"
  STATUS_REGINA="installed + smoke PASS"
  STATUS_KHOCA="installed + smoke PASS"
}

# ---- KnotJob: download jar + `knotjob` wrapper -------------------------------

install_knotjob() {
  log "KnotJob"
  local dir="${ORACLE_HOME}/knotjob"
  local jar="${dir}/KnotJob/KnotJob.jar"
  ${SUDO} mkdir -p "${dir}"
  # Rolling artifact with no version string: fetch to a temp file and compare its hash to the
  # installed jar. Swap (and rebuild the wrapper) only when the content actually differs -- that
  # applies an upstream update without recompiling identical bytes every run.
  curl -fsSL -o /tmp/KnotJob.zip "${KNOTJOB_URL}"
  ${SUDO} unzip -o -q /tmp/KnotJob.zip 'KnotJob/KnotJob.jar' -d /tmp/knotjob-fetch
  local fetched="/tmp/knotjob-fetch/KnotJob/KnotJob.jar"
  [ -f "${fetched}" ] || die "KnotJob.jar not found in the downloaded archive."
  local new_hash old_hash
  new_hash="$(file_hash "${fetched}")"
  old_hash="$(file_hash "${jar}")"
  if [ "${new_hash}" != "${old_hash}" ] || [ ! -x "${BIN_DIR}/knotjob" ]; then
    ${SUDO} mkdir -p "${dir}/KnotJob"
    ${SUDO} cp "${fetched}" "${jar}"
    ${SUDO} tee "${BIN_DIR}/knotjob" >/dev/null <<EOF
#!/usr/bin/env bash
exec java -jar "${jar}" "\$@"
EOF
    ${SUDO} chmod +x "${BIN_DIR}/knotjob"
    STATUS_KNOTJOB="installed/updated"; ACT_KNOTJOB=1; ACTED=1
    info "jar installed (${new_hash})"
  else
    STATUS_KNOTJOB="up to date"
    info "jar ${new_hash} unchanged"
  fi
  rm -rf /tmp/knotjob-fetch /tmp/KnotJob.zip
  command -v knotjob >/dev/null || die "knotjob wrapper not on PATH (${BIN_DIR})."
  VERSION_KNOTJOB="sha256:${new_hash}"
  info "wrapper at ${BIN_DIR}/knotjob"
}

smoke_knotjob() {
  log "KnotJob smoke (trefoil s-invariant)"
  local work="/tmp/knotjob-smoke"
  rm -rf "${work}"; mkdir -p "${work}"
  printf '%s\n' "${TREFOIL_PD}" > "${work}/trefoil.txt"
  ( cd "${work}" && knotjob trefoil.txt -kb0 -s0 >/dev/null 2>&1 )
  local out="${work}/trefoil.txt_s0_kb0"
  [ -f "${out}" ] || die "KnotJob wrote no output file -- run path broken."
  if grep -qE 'S-Invariant mod 0 : -?2$' "${out}"; then
    info "s-invariant magnitude 2 (PASS)"
  else
    cat "${out}" >&2; die "KnotJob smoke FAILED: trefoil s-invariant not +/-2."
  fi
  STATUS_KNOTJOB="installed + smoke PASS"
}

# ---- clone-or-refresh helper for the source-built oracles --------------------

clone_or_pull() {
  # clone_or_pull <repo-url> <dir>  -- sets SRC_CHANGED=1 iff the checked-out HEAD actually moved
  # (a fresh clone counts as changed), so a caller rebuilds only on a genuine upstream update.
  local repo="$1" dir="$2" before after
  SRC_CHANGED=0
  if [ -d "${dir}/.git" ]; then
    before="$(${SUDO} git -C "${dir}" rev-parse HEAD 2>/dev/null || echo none)"
    ${SUDO} git -C "${dir}" fetch --depth 1 origin --quiet
    ${SUDO} git -C "${dir}" reset --hard --quiet '@{upstream}'
    after="$(${SUDO} git -C "${dir}" rev-parse HEAD)"
    if [ "${before}" = "${after}" ]; then
      info "source at ${after} unchanged upstream"
    else
      info "source updated ${before} -> ${after}"; SRC_CHANGED=1
    fi
  else
    ${SUDO} git clone --depth 1 --quiet "${repo}" "${dir}"
    SRC_CHANGED=1
  fi
}

# ---- JavaKh-v2: build under Temurin, `javakh` wrapper ------------------------

install_javakh() {
  log "JavaKh-v2"
  local dir="${ORACLE_HOME}/javakh"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${JAVAKH_REPO}" "${dir}"          # sets SRC_CHANGED
  if [ "${SRC_CHANGED}" = "0" ] && [ -d "${dir}/build" ] && [ -x "${BIN_DIR}/javakh" ]; then
    STATUS_JAVAKH="up to date"
    info "up to date; skipping build"
  else
    # javac over all sources; JDK 25 is fine (only finalize() deprecation warnings, now in the log).
    quiet_build "JavaKh build" '( cd "${dir}" && ${SUDO} sh build.sh )'
    [ -d "${dir}/build" ] || die "JavaKh build produced no build/ dir."
    local cp="${dir}/build:${dir}/jars/log4j-1.2.12.jar:${dir}/jars/junit-4.12.jar:${dir}/jars/commons-logging-1.1.jar:${dir}/jars/commons-io-1.2.jar:${dir}/jars/commons-cli-1.0.jar"
    ${SUDO} tee "${BIN_DIR}/javakh" >/dev/null <<EOF
#!/usr/bin/env bash
exec java -cp "${cp}" org.katlas.JavaKh.JavaKh "\$@"
EOF
    ${SUDO} chmod +x "${BIN_DIR}/javakh"
    command -v javakh >/dev/null || die "javakh wrapper not on PATH (${BIN_DIR})."
    STATUS_JAVAKH="built/updated"; ACT_JAVAKH=1; ACTED=1
    info "wrapper at ${BIN_DIR}/javakh"
  fi
  VERSION_JAVAKH="git:$(git_version "${dir}")"
}

smoke_javakh() {
  log "JavaKh smoke (trefoil rational Khovanov, PD on stdin)"
  local out
  if ! out="$(printf '%s\n' "${TREFOIL_PD}" | timeout 60 javakh -Q 2>&1)"; then
    printf '%s\n' "${out}" >&2; die "JavaKh smoke: invocation failed (nonzero exit or timeout)."
  fi
  # trefoil Khovanov over Q has generators in three homological degrees (t^-3, t^-2, t^0).
  if printf '%s' "${out}" | grep -q 't\^-3' \
     && printf '%s' "${out}" | grep -q 't\^-2' \
     && printf '%s' "${out}" | grep -q 't\^0'; then
    info "rational Khovanov spans t^-3,t^-2,t^0 (PASS)"
  else
    printf '%s\n' "${out}" >&2; die "JavaKh smoke FAILED: trefoil Khovanov shape unexpected."
  fi
  STATUS_JAVAKH="built + smoke PASS"
}

# ---- kht++: build (cstdint patch + system Eigen), `khtpp` wrapper ------------

install_khtpp() {
  log "kht++"
  local dir="${ORACLE_HOME}/khtpp"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KHTPP_REPO}" "${dir}"           # sets SRC_CHANGED
  if [ "${SRC_CHANGED}" = "0" ] && [ -x "${dir}/kht++" ] && [ -x "${BIN_DIR}/khtpp" ]; then
    STATUS_KHTPP="up to date"
    info "up to date; skipping build"
  else
    # g++ needs cstdint for int_fast64_t; add it once (idempotent).
    local hdr="${dir}/sources/headers/Coefficients.h"
    [ -f "${hdr}" ] || die "kht++ Coefficients.h missing -- upstream layout changed."
    if ! grep -q '#include <cstdint>' "${hdr}"; then
      ${SUDO} sed -i 's|#include <vector>|#include <vector>\n#include <cstdint>|' "${hdr}"
    fi
    quiet_build "kht++ build" '( cd "${dir}" && ${SUDO} make -s PATH_EIGEN=/usr/include/eigen3 CXX=g++ )'
    [ -x "${dir}/kht++" ] || die "kht++ build produced no binary."
    # kht++ resolves its input path relative to the working directory (it strips a leading slash),
    # so a caller cd's to the directory holding the .kht file and passes a relative path.
    ${SUDO} tee "${BIN_DIR}/khtpp" >/dev/null <<EOF
#!/usr/bin/env bash
exec "${dir}/kht++" "\$@"
EOF
    ${SUDO} chmod +x "${BIN_DIR}/khtpp"
    command -v khtpp >/dev/null || die "khtpp wrapper not on PATH (${BIN_DIR})."
    STATUS_KHTPP="built/updated"; ACT_KHTPP=1; ACTED=1
    info "wrapper at ${BIN_DIR}/khtpp"
  fi
  VERSION_KHTPP="git:$(git_version "${dir}")"
}

smoke_khtpp() {
  log "kht++ smoke (shipped 3_1 trefoil example)"
  local dir="${ORACLE_HOME}/khtpp" out
  [ -f "${dir}/examples/tests/3_1.kht" ] || die "kht++ shipped example missing under ${dir}."
  # kht++ mangles an absolute path (strips the leading slash), so it must be given a path
  # relative to its working directory -- run from the khtpp dir with the relative example path.
  if ! out="$( cd "${dir}" && timeout 60 khtpp examples/tests/3_1.kht 2>&1 )"; then
    printf '%s\n' "${out}" >&2; die "kht++ smoke: invocation failed (nonzero exit or timeout)."
  fi
  if printf '%s' "${out}" | grep -q "Computation for" \
     && ! printf '%s' "${out}" | grep -qi "cannot read"; then
    info "computed the shipped 3_1 example (PASS)"
  else
    printf '%s\n' "${out}" >&2; die "kht++ smoke FAILED on the shipped 3_1 example."
  fi
  STATUS_KHTPP="built + smoke PASS"
}

# ---- knotkit: build using the SHIPPED bison-2 parser, `kk` wrapper -----------

install_knotkit() {
  log "knotkit"
  local dir="${ORACLE_HOME}/knotkit"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KNOTKIT_REPO}" "${dir}"         # sets SRC_CHANGED
  if [ "${SRC_CHANGED}" = "0" ] && [ -x "${dir}/kk" ] && [ -x "${BIN_DIR}/kk" ]; then
    STATUS_KNOTKIT="up to date"
    info "up to date; skipping build"
  else
    # The system bison 3.x regenerates a parser incompatible with knotkit's 2012 grammar. Restore
    # the shipped bison-2.4.3 parser and stamp the generated files newer than their sources so make
    # never re-runs bison/flex. (This clean rebuild runs only when the source changed or the binary
    # is missing -- never on a converged re-run.)
    quiet_build "knotkit build" \
      '( cd "${dir}"
         ${SUDO} git checkout -- .
         ${SUDO} find . -name \*.o -delete
         ${SUDO} touch knot_parser/knot_parser.cc knot_parser/knot_parser.hh knot_parser/knot_scanner.cc \
                       rd_parser/rd_parser.cc rd_parser/rd_parser.hh rd_parser/rd_scanner.cc
         ${SUDO} make -s CXX=g++ OPTFLAGS="-O2 -g -std=c++14" \
                      INCLUDES="-I. -I${GMP_INC}" LDFLAGS="" LIBS="-lgmp -lgmpxx -lz" kk )'
    [ -x "${dir}/kk" ] || die "knotkit build produced no kk binary."
    ${SUDO} tee "${BIN_DIR}/kk" >/dev/null <<EOF
#!/usr/bin/env bash
exec "${dir}/kk" "\$@"
EOF
    ${SUDO} chmod +x "${BIN_DIR}/kk"
    command -v kk >/dev/null || die "kk wrapper not on PATH (${BIN_DIR})."
    STATUS_KNOTKIT="built/updated"; ACT_KNOTKIT=1; ACTED=1
    info "wrapper at ${BIN_DIR}/kk"
  fi
  VERSION_KNOTKIT="git:$(git_version "${dir}")"
}

smoke_knotkit() {
  log "knotkit smoke (trefoil Rasmussen s)"
  local out
  if ! out="$(timeout 60 kk s "T(2,3)" 2>&1)"; then
    printf '%s\n' "${out}" >&2; die "knotkit smoke: invocation failed (nonzero exit or timeout)."
  fi
  if printf '%s' "${out}" | grep -qE 's\(T\(2,3\).*= *-?2$'; then
    info "s(T(2,3)) magnitude 2 (PASS)"
  else
    printf '%s\n' "${out}" >&2; die "knotkit smoke FAILED: trefoil s not +/-2."
  fi
  STATUS_KNOTKIT="built + smoke PASS"
}

# ---- KhoHo: build .so against PARI, `khoho` wrapper (gp with KhoHo preloaded) -

install_khoho() {
  log "KhoHo"
  local dir="${ORACLE_HOME}/khoho"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KHOHO_REPO}" "${dir}"           # sets SRC_CHANGED
  if [ "${SRC_CHANGED}" = "0" ] && ${SUDO} test -f "${dir}/print_ranks.so" && [ -x "${BIN_DIR}/khoho" ]; then
    STATUS_KHOHO="up to date"
    info "up to date; skipping build"
  else
    quiet_build "KhoHo build" '( cd "${dir}" && ${SUDO} make -s PARI_INPUT="-I${PARI_INC}" )'
    ${SUDO} test -f "${dir}/print_ranks.so" || die "KhoHo build produced no print_ranks.so."

    # The GP layer must load with DStore pre-seeded (so KhoHo_gvars' load-time reset_all() does not
    # fire before MAX_DIAGRAM_NUM, defined in the main KhoHo file, exists), then reset_all() once all
    # modules are read. gp runs from this dir because the install() paths to the .so are relative.
    ${SUDO} tee "${dir}/preamble.gp" >/dev/null <<'GP'
allocatemem(512*10^6);
DStore = vector(1);
read("KhoHo_gvars");
read("KhoHo_data");
read("KhoHo_diagr");
read("KhoHo_chain");
read("KhoHo_reduce");
read("KhoHo_print");
read("KhoHo");
reset_all();
GP

    ${SUDO} tee "${BIN_DIR}/khoho" >/dev/null <<EOF
#!/usr/bin/env bash
# khoho: run PARI/GP with KhoHo preloaded. Reads a gp program on stdin, appends quit.
set -euo pipefail
cd "${dir}"
tmp="\$(mktemp)"
{ cat "${dir}/preamble.gp"; cat; printf '\nquit;\n'; } > "\${tmp}"
gp -q "\${tmp}" </dev/null
rc=\$?
rm -f "\${tmp}"
exit \${rc}
EOF
    ${SUDO} chmod +x "${BIN_DIR}/khoho"
    command -v khoho >/dev/null || die "khoho wrapper not on PATH (${BIN_DIR})."
    STATUS_KHOHO="built/updated"; ACT_KHOHO=1; ACTED=1
    info "wrapper at ${BIN_DIR}/khoho"
  fi
  VERSION_KHOHO="git:$(git_version "${dir}")"
}

smoke_khoho() {
  log "KhoHo smoke (trefoil rational Khovanov polynomial)"
  local out
  if ! out="$(printf 'print(KhPol_Q(torus(2,3)));\n' | timeout 90 khoho 2>&1)"; then
    printf '%s\n' "${out}" >&2; die "KhoHo smoke: invocation failed (nonzero exit or timeout)."
  fi
  # trefoil unreduced Khovanov over Q: q + q^3 + q^5 t^2 + q^9 t^3 -- gate on the top term.
  if printf '%s' "${out}" | grep -q 'q\^9\*t\^3'; then
    info "KhPol_Q(trefoil) includes q^9*t^3 (PASS)"
  else
    printf '%s\n' "${out}" >&2; die "KhoHo smoke FAILED: trefoil KhPol_Q unexpected."
  fi
  STATUS_KHOHO="built + smoke PASS"
}

# ---- SageMath (CT 250 only; installed above when INSTALL_SAGE=1) -------------

check_sage() {
  if [ "${INSTALL_SAGE}" = "1" ]; then
    log "SageMath smoke"
    sage --version 2>/dev/null | grep -qi "SageMath" || die "sage --version did not report SageMath."
    info "$(sage --version 2>/dev/null | head -1)"
    STATUS_SAGE="installed + smoke PASS"
    VERSION_SAGE="$(sage --version 2>/dev/null | head -1 | sed 's/^SageMath version //; s/,.*//')"
  else
    log "SageMath (skipped)"
    info "INSTALL_SAGE=0: Sage is a CT 250 install (multi-GB apt tree); the sandbox reports it absent."
    STATUS_SAGE="skipped (CT 250 only; set INSTALL_SAGE=1)"
  fi
}

# ---- summary -----------------------------------------------------------------

summary() {
  log "Summary"
  printf '   %-9s %-30s %s\n' "oracle" "version" "state"
  printf '   %-9s %-30s %s\n' "kfh"     "${VERSION_KFH}"     "${STATUS_KFH}"
  printf '   %-9s %-30s %s\n' "snappy"  "${VERSION_SNAPPY}"  "${STATUS_SNAPPY}"
  printf '   %-9s %-30s %s\n' "regina"  "${VERSION_REGINA}"  "${STATUS_REGINA}"
  printf '   %-9s %-30s %s\n' "khoca"   "${VERSION_KHOCA}"   "${STATUS_KHOCA}"
  printf '   %-9s %-30s %s\n' "knotjob" "${VERSION_KNOTJOB}" "${STATUS_KNOTJOB}"
  printf '   %-9s %-30s %s\n' "javakh"  "${VERSION_JAVAKH}"  "${STATUS_JAVAKH}"
  printf '   %-9s %-30s %s\n' "khtpp"   "${VERSION_KHTPP}"   "${STATUS_KHTPP}"
  printf '   %-9s %-30s %s\n' "knotkit" "${VERSION_KNOTKIT}" "${STATUS_KNOTKIT}"
  printf '   %-9s %-30s %s\n' "khoho"   "${VERSION_KHOHO}"   "${STATUS_KHOHO}"
  printf '   %-9s %-30s %s\n' "sage"    "${VERSION_SAGE}"    "${STATUS_SAGE}"
  printf '\n   ORACLE_HOME=%s   BIN_DIR=%s\n' "${ORACLE_HOME}" "${BIN_DIR}"
  info "Verify what the adapter sees:  PYTHONPATH=. ${PY} -c \"from scripts.comparison import adapters; [print(o.key, o.available()) for o in adapters.ORACLES]\""
}

# ---- --check: report state, change nothing --------------------------------------------------------

check_source() {
  # check_source <key> <dir> <artifact> <wrapper>  -- report a source oracle's version + update state
  local key="$1" dir="$2" artifact="$3" bin="$4" sha state
  if [ ! -e "${artifact}" ] || [ ! -x "${bin}" ]; then
    printf '   %-9s %-22s %s\n' "${key}" "-" "MISSING"
    return 1
  fi
  sha="$(git_version "${dir}")"
  if git_remote_current "${dir}"; then state="current"; else state="update available"; fi
  printf '   %-9s %-22s %s\n' "${key}" "git:${sha}" "present (${state})"
  return 0
}

check_mode() {
  log "Oracle check (dry run -- nothing is installed, built, or modified)"
  printf '   %-9s %-22s %s\n' "oracle" "version" "state"
  local missing=0 pair key dist v
  for pair in "kfh:knot_floer_homology" "snappy:snappy" "regina:regina" "khoca:khoca"; do
    key="${pair%%:*}"; dist="${pair#*:}"
    v="$(pip_version "${dist}")"
    if [ "${v}" = "absent" ]; then
      printf '   %-9s %-22s %s\n' "${key}" "-" "MISSING"; missing=$((missing + 1))
    else
      printf '   %-9s %-22s %s\n' "${key}" "${v}" "present"
    fi
  done
  v="$(file_hash "${ORACLE_HOME}/knotjob/KnotJob/KnotJob.jar")"
  if [ "${v}" = "absent" ] || [ ! -x "${BIN_DIR}/knotjob" ]; then
    printf '   %-9s %-22s %s\n' "knotjob" "-" "MISSING"; missing=$((missing + 1))
  else
    printf '   %-9s %-22s %s\n' "knotjob" "sha256:${v}" "present"
  fi
  check_source javakh  "${ORACLE_HOME}/javakh"  "${ORACLE_HOME}/javakh/build"          "${BIN_DIR}/javakh" || missing=$((missing + 1))
  check_source khtpp   "${ORACLE_HOME}/khtpp"   "${ORACLE_HOME}/khtpp/kht++"           "${BIN_DIR}/khtpp"  || missing=$((missing + 1))
  check_source knotkit "${ORACLE_HOME}/knotkit" "${ORACLE_HOME}/knotkit/kk"            "${BIN_DIR}/kk"     || missing=$((missing + 1))
  check_source khoho   "${ORACLE_HOME}/khoho"   "${ORACLE_HOME}/khoho/print_ranks.so"  "${BIN_DIR}/khoho"  || missing=$((missing + 1))
  if command -v sage >/dev/null 2>&1; then
    printf '   %-9s %-22s %s\n' "sage" "$(sage --version 2>/dev/null | head -1 | sed 's/^SageMath version //; s/,.*//')" "present"
  else
    printf '   %-9s %-22s %s\n' "sage" "-" "absent (CT 250 only unless INSTALL_SAGE=1)"
  fi
  printf '\n'
  if [ "${missing}" -ne 0 ]; then
    info "${missing} oracle(s) missing -- run 'scripts/install_oracles.sh' (no --check) to converge."
    return 1
  fi
  info "All oracles present -- good to proceed. Run without --check to apply any upstream updates."
  return 0
}

# ---- verdict -------------------------------------------------------------------------------------

verdict() {
  if [ "${ACTED}" -eq 0 ]; then
    log "All oracles present and current -- good to proceed."
  else
    log "Oracle setup complete (changes applied this run) -- good to proceed."
  fi
}

main() {
  for arg in "$@"; do
    case "${arg}" in
      --check) CHECK_ONLY=1 ;;
      *) die "unknown argument '${arg}' (accepted: --check)." ;;
    esac
  done
  if [ "${CHECK_ONLY}" -eq 1 ]; then
    check_mode; exit $?
  fi

  preflight
  ensure_apt              # installs the toolchain, or skips (no network) when already present
  install_pip_oracles     # applies updates; sets ACT_PIP if a version changed
  install_knotjob         # swaps the rolling jar only if its hash changed
  install_javakh          # each source oracle rebuilds only on an upstream change or missing artifact
  install_khtpp
  install_knotkit
  install_khoho

  # Smoke only what was installed/built/updated this run (or everything under VERIFY=1); a converged
  # oracle was already proven on the run that built it, so a no-change re-run stays fast.
  want_smoke "${ACT_PIP}"     && smoke_pip_oracles || info "pip oracles unchanged; smoke skipped (VERIFY=1 to force)"
  want_smoke "${ACT_KNOTJOB}" && smoke_knotjob     || info "knotjob unchanged; smoke skipped (VERIFY=1 to force)"
  want_smoke "${ACT_JAVAKH}"  && smoke_javakh      || info "javakh unchanged; smoke skipped (VERIFY=1 to force)"
  want_smoke "${ACT_KHTPP}"   && smoke_khtpp       || info "khtpp unchanged; smoke skipped (VERIFY=1 to force)"
  want_smoke "${ACT_KNOTKIT}" && smoke_knotkit     || info "knotkit unchanged; smoke skipped (VERIFY=1 to force)"
  want_smoke "${ACT_KHOHO}"   && smoke_khoho       || info "khoho unchanged; smoke skipped (VERIFY=1 to force)"
  check_sage
  summary
  verdict
}

main "$@"
