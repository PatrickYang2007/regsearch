#!/usr/bin/env bash
# Start Postgres 17 + pgvector inside Apptainer, as the calling user.
#
# Why not the container's docker-entrypoint.sh: it expects to run as root (to
# chown PGDATA and su to `postgres`). Under Apptainer we run as our own uid,
# which has no entry in the container's /etc/passwd, so the entrypoint aborts.
# Calling initdb/postgres directly sidesteps all of that -- Postgres only cares
# that PGDATA is owned by the running uid, which it is.
#
# Connections go over a Unix socket in data/run, not TCP. On a shared cluster
# that avoids port collisions with other users and exposes no listening socket.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${ROOT}/containers/pgvector.sif"
PGDATA="${REGSEARCH_PGDATA_DIR:-${ROOT}/data/pgdata}"
RUNDIR="${REGSEARCH_RUN_DIR:-${ROOT}/data/run}"
LOGFILE="${RUNDIR}/postgres.log"
PIDFILE="${RUNDIR}/postgres.pid"

PGUSER_NAME="${REGSEARCH_PG_USER:-regsearch}"
PGDB_NAME="${REGSEARCH_PG_DB:-regsearch}"

mkdir -p "${PGDATA}" "${RUNDIR}"
chmod 700 "${PGDATA}"

if [[ ! -f "${SIF}" ]]; then
  echo "error: ${SIF} not found. Run: scripts/pull_image.sh" >&2
  exit 1
fi

# NOTE: /vast is NFS. Postgres is supported on NFS only with working locks
# (NFSv4 here) and a single concurrent postmaster. That holds for this
# single-writer prototype. If you ever see lock errors, point
# REGSEARCH_PGDATA_DIR at node-local scratch instead and re-ingest.

# Bind the project root so PGDATA/RUNDIR resolve identically inside the container.
APPTAINER_ARGS=(--cleanenv --bind "${ROOT}:${ROOT}")

if [[ -s "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
  echo "postgres already running (pid $(cat "${PIDFILE}")), socket at ${RUNDIR}"
  exit 0
fi

if [[ ! -f "${PGDATA}/PG_VERSION" ]]; then
  echo "==> initdb (first run)"
  # trust auth on the local socket: the socket lives in a 0700 dir we own, so
  # the filesystem is already the access control. No password to leak into env.
  apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    initdb -D "${PGDATA}" \
      --username="${PGUSER_NAME}" \
      --auth-local=trust \
      --auth-host=reject \
      --encoding=UTF8 --locale=C >/dev/null
fi

echo "==> starting postgres"
nohup apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  postgres -D "${PGDATA}" \
    -c listen_addresses='' \
    -c unix_socket_directories="${RUNDIR}" \
    -c shared_buffers=1GB \
    -c work_mem=64MB \
    -c maintenance_work_mem=2GB \
    -c max_parallel_workers_per_gather=2 \
    >"${LOGFILE}" 2>&1 &
echo $! >"${PIDFILE}"

echo "==> waiting for readiness"
for _ in $(seq 1 60); do
  if apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
      pg_isready -h "${RUNDIR}" -U "${PGUSER_NAME}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    pg_isready -h "${RUNDIR}" -U "${PGUSER_NAME}" >/dev/null 2>&1; then
  echo "error: postgres did not become ready. Tail of ${LOGFILE}:" >&2
  tail -30 "${LOGFILE}" >&2
  exit 1
fi

# Idempotent database + extension setup.
if ! apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    psql -h "${RUNDIR}" -U "${PGUSER_NAME}" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${PGDB_NAME}'" | grep -q 1; then
  apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    createdb -h "${RUNDIR}" -U "${PGUSER_NAME}" "${PGDB_NAME}"
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  psql -h "${RUNDIR}" -U "${PGUSER_NAME}" -d "${PGDB_NAME}" -v ON_ERROR_STOP=1 \
  -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "postgres ready"
echo "  socket : ${RUNDIR}"
echo "  db     : ${PGDB_NAME}  user: ${PGUSER_NAME}"
echo "  log    : ${LOGFILE}"
echo "  psql   : scripts/pg_psql.sh"
