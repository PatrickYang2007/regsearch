#!/usr/bin/env bash
# Open psql against the containerised Postgres. Extra args pass through:
#   scripts/pg_psql.sh -c "SELECT count(*) FROM passages;"
#
# Picks the transport automatically: the Unix socket when running on the node
# that hosts Postgres, TCP otherwise. A socket file on shared storage is NOT
# connectable from another host, so a Slurm job on a different node must go
# over TCP -- getting this wrong looks like "server not running" from the
# compute node while psql works fine on the login node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="${REGSEARCH_RUN_DIR:-${ROOT}/data/run}"
PORT="${REGSEARCH_PG_PORT:-5432}"
USER_NAME="${REGSEARCH_PG_USER:-regsearch}"
DB_NAME="${REGSEARCH_PG_DB:-regsearch}"

PG_HOST_FILE="${RUNDIR}/pg_host"
PW_FILE="${RUNDIR}/pg_password"

if [[ ! -s "${PG_HOST_FILE}" ]]; then
  echo "error: ${PG_HOST_FILE} missing. Run scripts/pg_start.sh first." >&2
  exit 1
fi

RECORDED_HOST="$(cat "${PG_HOST_FILE}")"
APPTAINER_ARGS=(--cleanenv --bind "${ROOT}:${ROOT}")

if [[ "$(hostname -f)" == "${RECORDED_HOST}" ]]; then
  CONN_HOST="${RUNDIR}"          # same node: socket
else
  CONN_HOST="${RECORDED_HOST}"   # different node: TCP
  if [[ -s "${PW_FILE}" ]]; then
    APPTAINER_ARGS+=(--env "PGPASSWORD=$(tr -d '\n' <"${PW_FILE}")")
  fi
fi

exec apptainer exec "${APPTAINER_ARGS[@]}" "${ROOT}/containers/pgvector.sif" \
  psql -h "${CONN_HOST}" -p "${PORT}" -U "${USER_NAME}" -d "${DB_NAME}" "$@"
