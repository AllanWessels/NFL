#!/usr/bin/env bash
# Weekly retrain, as cron runs it.
#
# Tuesday morning is the slot: Monday Night Football has finished, nflverse has
# published the week, and there is a full six days before the next Sunday to
# notice that the gate refused.
#
# Everything interesting is decided by `nfl retrain` itself. This wrapper only
# does the three things a cron job needs and an interactive shell gives you for
# free: run from the repo root, keep a dated log, and exit with a status the
# scheduler can act on.

set -euo pipefail

# Resolve the repo root from this script's own location, not from $PWD: cron
# starts in $HOME and a relative path would silently retrain nothing.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${NFL_LOG_DIR:-${REPO_ROOT}/data/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/retrain-${STAMP}.log"

# uv lives in ~/.local/bin or ~/.cargo/bin; cron's PATH has neither.
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# rich wraps to the terminal width, and cron has no terminal.
export COLUMNS="${COLUMNS:-200}"
export NO_COLOR="${NO_COLOR:-1}"

{
  echo "=== nfl retrain @ ${STAMP} (${REPO_ROOT}) ==="
  uv --version || true
  echo
} >>"${LOG_FILE}" 2>&1

set +e
uv run nfl retrain "$@" >>"${LOG_FILE}" 2>&1
STATUS=$?
set -e

{
  echo
  echo "=== exit ${STATUS} @ $(date -u +%Y%m%dT%H%M%SZ) ==="
} >>"${LOG_FILE}"

# The tail goes to stdout so cron's MAILTO carries the verdict rather than a
# bare exit code; the whole run stays in the log file.
if [[ ${STATUS} -ne 0 ]]; then
  echo "nfl retrain FAILED (exit ${STATUS}). Last 40 lines of ${LOG_FILE}:" >&2
  tail -n 40 "${LOG_FILE}" >&2
else
  echo "nfl retrain ok. Log: ${LOG_FILE}"
  tail -n 20 "${LOG_FILE}"
fi

exit "${STATUS}"
