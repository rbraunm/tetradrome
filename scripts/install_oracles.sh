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
#   khoho    builds .so modules against system PARI; the GP layer must be loaded data-before-gvars
#            with DStore pre-seeded so KhoHo_gvars' load-time reset_all() sees MAX_DIAGRAM_NUM, and
#            gp must run from the source dir (the install() paths to the .so are relative).
#
# Idempotent and fail-loud: re-running is safe, and any whitelist regression or failed smoke
# aborts loudly with a nonzero exit.
#
#   scripts/install_oracles.sh
#   ORACLE_HOME=/home/claude/oracles scripts/install_oracles.sh
#   PIP="/opt/tetradrome/venv/bin/pip" PIP_INSTALL_FLAGS="" PY="/opt/tetradrome/venv/bin/python" \
#       INSTALL_SAGE=1 scripts/install_oracles.sh    # how provision_runner.py invokes it on CT 250
#
# Tunables (environment): ORACLE_HOME (default /opt/oracles), BIN_DIR (default /usr/local/bin),
# TEMURIN_MAJOR (default 25; KnotJob needs Java >= 23), PIP (default 'pip'; a venv pip on CT 250),
# PIP_INSTALL_FLAGS (default '--break-system-packages'; '' for a venv pip), PY (default 'python';
# the venv python on CT 250), INSTALL_SAGE (default 0).

set -euo pipefail

ORACLE_HOME="${ORACLE_HOME:-/opt/oracles}"
BIN_DIR="${BIN_DIR:-/usr/local/bin}"
TEMURIN_MAJOR="${TEMURIN_MAJOR:-25}"
PIP="${PIP:-pip}"
PIP_INSTALL_FLAGS="${PIP_INSTALL_FLAGS:---break-system-packages}"
PY="${PY:-python}"
INSTALL_SAGE="${INSTALL_SAGE:-0}"

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

# Per-oracle outcome, printed in the closing summary.
STATUS_KFH="not attempted"
STATUS_SNAPPY="not attempted"
STATUS_REGINA="not attempted"
STATUS_KHOCA="not attempted"
STATUS_KNOTJOB="not attempted"
STATUS_JAVAKH="not attempted"
STATUS_KHTPP="not attempted"
STATUS_KNOTKIT="not attempted"
STATUS_KHOHO="not attempted"
STATUS_SAGE="not attempted"

log() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

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

  # Multiarch header dirs for the source builds (parent dir so <gmp.h>/<pari/pari.h> resolve).
  GMP_INC="$(dirname "$(dpkg -L libgmp-dev  | grep -m1 '/gmp\.h$')")"
  PARI_INC="$(dirname "$(dirname "$(dpkg -L libpari-dev | grep -m1 '/pari\.h$')")")"
  [ -d "${GMP_INC}" ]  || die "could not locate gmp.h include dir."
  [ -d "${PARI_INC}" ] || die "could not locate pari.h include dir."
  info "gmp include: ${GMP_INC}   pari include: ${PARI_INC}"
}

# ---- python oracles: kfh (via snappy), snappy, regina, khoca -----------------

install_pip_oracles() {
  log "pip oracles: snappy (pulls kfh), regina, khoca (+ py sympy)"
  ${PIP} install ${PIP_INSTALL_FLAGS} -q snappy
  ${PIP} install ${PIP_INSTALL_FLAGS} -q regina khoca py sympy
  STATUS_KFH="installed (import knot_floer_homology)"
  STATUS_SNAPPY="installed"
  STATUS_REGINA="installed"
  STATUS_KHOCA="installed"
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
  ${SUDO} mkdir -p "${dir}"
  curl -fsSL -o /tmp/KnotJob.zip "${KNOTJOB_URL}"        # rolling artifact: always fetch fresh
  ${SUDO} unzip -o -q /tmp/KnotJob.zip 'KnotJob/KnotJob.jar' -d "${dir}"
  local jar="${dir}/KnotJob/KnotJob.jar"
  [ -f "${jar}" ] || die "KnotJob.jar not found in the downloaded archive."
  ${SUDO} tee "${BIN_DIR}/knotjob" >/dev/null <<EOF
#!/usr/bin/env bash
exec java -jar "${jar}" "\$@"
EOF
  ${SUDO} chmod +x "${BIN_DIR}/knotjob"
  command -v knotjob >/dev/null || die "knotjob wrapper not on PATH (${BIN_DIR})."
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
  # clone_or_pull <repo-url> <dir>
  local repo="$1" dir="$2"
  if [ -d "${dir}/.git" ]; then
    info "source present; refreshing"
    ${SUDO} git -C "${dir}" fetch --depth 1 origin --quiet
    ${SUDO} git -C "${dir}" reset --hard --quiet '@{upstream}'
  else
    ${SUDO} git clone --depth 1 --quiet "${repo}" "${dir}"
  fi
}

# ---- JavaKh-v2: build under Temurin, `javakh` wrapper ------------------------

install_javakh() {
  log "JavaKh-v2 (build)"
  local dir="${ORACLE_HOME}/javakh"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${JAVAKH_REPO}" "${dir}"
  ( cd "${dir}" && ${SUDO} sh build.sh )                 # javac over all sources; JDK 25 is fine
  [ -d "${dir}/build" ] || die "JavaKh build produced no build/ dir."
  local cp="${dir}/build:${dir}/jars/log4j-1.2.12.jar:${dir}/jars/junit-4.12.jar:${dir}/jars/commons-logging-1.1.jar:${dir}/jars/commons-io-1.2.jar:${dir}/jars/commons-cli-1.0.jar"
  ${SUDO} tee "${BIN_DIR}/javakh" >/dev/null <<EOF
#!/usr/bin/env bash
exec java -cp "${cp}" org.katlas.JavaKh.JavaKh "\$@"
EOF
  ${SUDO} chmod +x "${BIN_DIR}/javakh"
  command -v javakh >/dev/null || die "javakh wrapper not on PATH (${BIN_DIR})."
  info "wrapper at ${BIN_DIR}/javakh"
}

smoke_javakh() {
  log "JavaKh smoke (trefoil rational Khovanov, PD on stdin)"
  local out
  out="$(printf '%s\n' "${TREFOIL_PD}" | timeout 60 javakh -Q 2>/dev/null || true)"
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
  log "kht++ (build)"
  local dir="${ORACLE_HOME}/khtpp"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KHTPP_REPO}" "${dir}"
  # g++ needs cstdint for int_fast64_t; add it once (idempotent).
  local hdr="${dir}/sources/headers/Coefficients.h"
  [ -f "${hdr}" ] || die "kht++ Coefficients.h missing -- upstream layout changed."
  if ! grep -q '#include <cstdint>' "${hdr}"; then
    ${SUDO} sed -i 's|#include <vector>|#include <vector>\n#include <cstdint>|' "${hdr}"
  fi
  ( cd "${dir}" && ${SUDO} make -s PATH_EIGEN=/usr/include/eigen3 CXX=g++ )
  [ -x "${dir}/kht++" ] || die "kht++ build produced no binary."
  # kht++ resolves its input path relative to the working directory (it strips a leading slash),
  # so a caller cd's to the directory holding the .kht file and passes a relative path.
  ${SUDO} tee "${BIN_DIR}/khtpp" >/dev/null <<EOF
#!/usr/bin/env bash
exec "${dir}/kht++" "\$@"
EOF
  ${SUDO} chmod +x "${BIN_DIR}/khtpp"
  command -v khtpp >/dev/null || die "khtpp wrapper not on PATH (${BIN_DIR})."
  info "wrapper at ${BIN_DIR}/khtpp"
}

smoke_khtpp() {
  log "kht++ smoke (shipped 3_1 trefoil example)"
  local dir="${ORACLE_HOME}/khtpp" out
  [ -f "${dir}/examples/tests/3_1.kht" ] || die "kht++ shipped example missing under ${dir}."
  # kht++ mangles an absolute path (strips the leading slash), so it must be given a path
  # relative to its working directory -- run from the khtpp dir with the relative example path.
  out="$( cd "${dir}" && timeout 60 khtpp examples/tests/3_1.kht 2>&1 || true )"
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
  log "knotkit (build)"
  local dir="${ORACLE_HOME}/knotkit"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KNOTKIT_REPO}" "${dir}"
  # The system bison 3.x regenerates a parser incompatible with knotkit's 2012 grammar. Restore
  # the shipped bison-2.4.3 parser and stamp the generated files newer than their sources so make
  # never re-runs bison/flex.
  ( cd "${dir}"
    ${SUDO} git checkout -- .
    ${SUDO} find . -name '*.o' -delete
    ${SUDO} touch knot_parser/knot_parser.cc knot_parser/knot_parser.hh knot_parser/knot_scanner.cc \
                  rd_parser/rd_parser.cc rd_parser/rd_parser.hh rd_parser/rd_scanner.cc
    ${SUDO} make -s CXX=g++ OPTFLAGS="-O2 -g -std=c++14" \
                 INCLUDES="-I. -I${GMP_INC}" LDFLAGS="" LIBS="-lgmp -lgmpxx -lz" kk )
  [ -x "${dir}/kk" ] || die "knotkit build produced no kk binary."
  ${SUDO} tee "${BIN_DIR}/kk" >/dev/null <<EOF
#!/usr/bin/env bash
exec "${dir}/kk" "\$@"
EOF
  ${SUDO} chmod +x "${BIN_DIR}/kk"
  command -v kk >/dev/null || die "kk wrapper not on PATH (${BIN_DIR})."
  info "wrapper at ${BIN_DIR}/kk"
}

smoke_knotkit() {
  log "knotkit smoke (trefoil Rasmussen s)"
  local out
  out="$(timeout 60 kk s "T(2,3)" 2>&1 || true)"
  if printf '%s' "${out}" | grep -qE 's\(T\(2,3\).*= *-?2$'; then
    info "s(T(2,3)) magnitude 2 (PASS)"
  else
    printf '%s\n' "${out}" >&2; die "knotkit smoke FAILED: trefoil s not +/-2."
  fi
  STATUS_KNOTKIT="built + smoke PASS"
}

# ---- KhoHo: build .so against PARI, `khoho` wrapper (gp with KhoHo preloaded) -

install_khoho() {
  log "KhoHo (build)"
  local dir="${ORACLE_HOME}/khoho"
  ${SUDO} mkdir -p "${ORACLE_HOME}"
  clone_or_pull "${KHOHO_REPO}" "${dir}"
  ( cd "${dir}" && ${SUDO} make -s PARI_INPUT="-I${PARI_INC}" )
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
  info "wrapper at ${BIN_DIR}/khoho"
}

smoke_khoho() {
  log "KhoHo smoke (trefoil rational Khovanov polynomial)"
  local out
  out="$(printf 'print(KhPol_Q(torus(2,3)));\n' | timeout 90 khoho 2>/dev/null || true)"
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
  else
    log "SageMath (skipped)"
    info "INSTALL_SAGE=0: Sage is a CT 250 install (multi-GB apt tree); the sandbox reports it absent."
    STATUS_SAGE="skipped (CT 250 only; set INSTALL_SAGE=1)"
  fi
}

# ---- summary -----------------------------------------------------------------

summary() {
  log "Summary"
  printf '   %-9s %s\n' "kfh"     "${STATUS_KFH}"
  printf '   %-9s %s\n' "snappy"  "${STATUS_SNAPPY}"
  printf '   %-9s %s\n' "regina"  "${STATUS_REGINA}"
  printf '   %-9s %s\n' "khoca"   "${STATUS_KHOCA}"
  printf '   %-9s %s\n' "knotjob" "${STATUS_KNOTJOB}"
  printf '   %-9s %s\n' "javakh"  "${STATUS_JAVAKH}"
  printf '   %-9s %s\n' "khtpp"   "${STATUS_KHTPP}"
  printf '   %-9s %s\n' "knotkit" "${STATUS_KNOTKIT}"
  printf '   %-9s %s\n' "khoho"   "${STATUS_KHOHO}"
  printf '   %-9s %s\n' "sage"    "${STATUS_SAGE}"
  printf '\n   ORACLE_HOME=%s   BIN_DIR=%s\n' "${ORACLE_HOME}" "${BIN_DIR}"
  info "Verify what the adapter sees:  PYTHONPATH=. ${PY} -c \"from scripts.comparison import adapters; [print(o.key, o.available()) for o in adapters.ORACLES]\""
}

main() {
  preflight
  apt_setup
  install_pip_oracles
  install_knotjob
  install_javakh
  install_khtpp
  install_knotkit
  install_khoho
  smoke_pip_oracles
  smoke_knotjob
  smoke_javakh
  smoke_khtpp
  smoke_knotkit
  smoke_khoho
  check_sage
  summary
  log "Oracle setup complete."
}

main "$@"
