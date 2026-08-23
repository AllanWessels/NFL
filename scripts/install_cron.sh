#!/usr/bin/env bash
# Install (or remove) the crontab entry that runs the weekly retrain.
#
#   scripts/install_cron.sh            # show what would be installed
#   scripts/install_cron.sh --install  # install it
#   scripts/install_cron.sh --remove   # take it out again
#
# The slot is Tuesday 06:00 **local time** — cron has no timezone of its own, it
# uses the system's, which is what you want here: "Tuesday morning" is a fact
# about when the football was played, not about UTC.
#
# Installing is idempotent: the entry is tagged with a marker comment, and an
# existing tagged entry is replaced rather than duplicated. Nothing else in the
# crontab is touched.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/weekly_retrain.sh"

MARKER="# nfl-weekly-retrain (${REPO_ROOT})"
SCHEDULE="${NFL_CRON_SCHEDULE:-0 6 * * 2}"
ENTRY="${SCHEDULE} ${RUNNER} >> ${REPO_ROOT}/data/logs/cron.log 2>&1"

usage() {
  sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

action="${1:---show}"

case "${action}" in
  --show | -n | "")
    echo "Would install into the crontab of $(id -un):"
    echo
    echo "${MARKER}"
    echo "${ENTRY}"
    echo
    echo "Run '${BASH_SOURCE[0]} --install' to install it."
    exit 0
    ;;
  --install)
    if [[ ! -x "${RUNNER}" ]]; then
      echo "error: ${RUNNER} is not executable — run: chmod +x ${RUNNER}" >&2
      exit 1
    fi
    mkdir -p "${REPO_ROOT}/data/logs"
    # `crontab -l` exits non-zero when there is no crontab yet; that is not an
    # error, it is the first install.
    current="$(crontab -l 2>/dev/null || true)"
    # Drop any previous entry of ours (the marker line and the line after it).
    filtered="$(printf '%s\n' "${current}" | grep -vF "${MARKER}" | grep -vF "${RUNNER}" || true)"
    {
      printf '%s\n' "${filtered}" | sed '/^$/d'
      printf '%s\n' "${MARKER}"
      printf '%s\n' "${ENTRY}"
    } | crontab -
    echo "installed:"
    crontab -l | grep -A1 -F "${MARKER}"
    ;;
  --remove)
    current="$(crontab -l 2>/dev/null || true)"
    if [[ -z "${current}" ]]; then
      echo "no crontab for $(id -un); nothing to remove"
      exit 0
    fi
    printf '%s\n' "${current}" | grep -vF "${MARKER}" | grep -vF "${RUNNER}" | crontab -
    echo "removed the nfl weekly retrain entry"
    ;;
  --help | -h)
    usage
    ;;
  *)
    echo "unknown option ${action}" >&2
    usage >&2
    exit 2
    ;;
esac
