#!/usr/bin/env bash
# Stop the Postgres started by pg_start.sh (fast shutdown, then SIGKILL fallback).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${ROOT}/containers/pgvector.sif"
PGDATA="${REGSEARCH_PGDATA_DIR:-${ROOT}/data/pgdata}"
RUNDIR="${REGSEARCH_RUN_DIR:-${ROOT}/data/run}"
PIDFILE="${RUNDIR}/postgres.pid"

if [[ -f "${SIF}" ]]; then
  # -m fast: roll back open transactions, checkpoint, exit. Avoids the recovery
  # pass a SIGKILL would force on next start.
  apptainer exec --cleanenv --bind "${ROOT}:${ROOT}" "${SIF}" \
    pg_ctl -D "${PGDATA}" -m fast stop 2>/dev/null || true
fi

if [[ -s "${PIDFILE}" ]]; then
  PID="$(cat "${PIDFILE}")"
  for _ in $(seq 1 20); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "${PID}" 2>/dev/null; then
    echo "warn: pid ${PID} still alive after fast stop; sending SIGKILL" >&2
    kill -9 "${PID}" 2>/dev/null || true
  fi
  rm -f "${PIDFILE}"
fi

echo "postgres stopped"
