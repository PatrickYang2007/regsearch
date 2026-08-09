#!/usr/bin/env bash
# One-command session start: activate the venv, bring Postgres up, show the
# corpus counts.
#
# MUST BE SOURCED, not executed:
#
#     source scripts/start.sh
#
# Activating a virtualenv means mutating the *calling* shell's PATH. A child
# process cannot do that to its parent, so running this as ./scripts/start.sh
# would start Postgres and then exit, leaving your prompt exactly as unactivated
# as it was. The guard below catches that rather than letting it look like it
# worked.
#
# Deliberately no `set -euo pipefail`: those options would persist in your
# interactive shell after sourcing, and `set -e` there means the next command
# that returns non-zero closes your terminal.

# The file is chmod +x so that `./scripts/start.sh` reaches this guard and gets
# the explanation. Left non-executable it fails with a bare "Permission denied"
# (exit 126) instead, which reads as a broken checkout rather than as "you used
# it wrong" -- the guard may as well not exist.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: this script must be sourced, not executed." >&2
  echo "       run:  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

# BASH_SOURCE, not $0 or $PWD: when sourced, $0 is the calling shell's name and
# the caller may be in any directory. This resolves the project root from the
# script's own location, so `source /abs/path/to/scripts/start.sh` works from
# anywhere.
_regsearch_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Every failure path below unsets _regsearch_root before returning. A bare
# `return 1` would leave the variable set in the CALLER's interactive shell --
# sourcing has no scope of its own, so anything this file forgets to clean up
# is permanent for that terminal.
cd "${_regsearch_root}" || { unset _regsearch_root; return 1; }

if [[ ! -f .venv/bin/activate ]]; then
  echo "error: no virtualenv at ${_regsearch_root}/.venv" >&2
  echo "       run: uv sync --extra dev" >&2
  unset _regsearch_root
  return 1
fi

source .venv/bin/activate || { unset _regsearch_root; return 1; }

# Idempotent: prints and returns 0 if a server is already up on this node.
./scripts/pg_start.sh || {
  echo "error: Postgres failed to start; see data/run/postgres.log" >&2
  unset _regsearch_root
  return 1
}

echo
regsearch stats

cat <<'EOF'

Expected: 19,791 docs / 99,567 passages / 0 unembedded / 8,075 contexts.
Anything else, stop and find out why before running more.

  regsearch search "chromatin looping" --arm dense
  regsearch eval --split test --origin citation --out docs/ablation.md

EOF

unset _regsearch_root
