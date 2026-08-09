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
# Must match Settings.pg_port: over a Unix socket this selects the socket
# filename (.s.PGSQL.<port>), so a mismatch means the client finds nothing.
PGPORT_NUM="${REGSEARCH_PG_PORT:-5432}"

mkdir -p "${PGDATA}" "${RUNDIR}"
chmod 700 "${PGDATA}"
# RUNDIR holds the Unix socket, and local socket connections use trust auth --
# no password. The filesystem IS the access control for that path, so this
# chmod is load-bearing, not tidiness. It was missing: the directory inherited
# 2755 from its parent, leaving password-free superuser access to anyone in the
# group who could reach the node. Only the group-restricted parent above it was
# actually keeping others out, which is not what the trust setting below
# assumes.
chmod 700 "${RUNDIR}"

# Slurm jobs land on other nodes and must reach this database. A Unix socket
# cannot serve them: it is local IPC, and a socket file on shared storage is
# still not connectable from another host -- the compute node sees the inode
# and gets ECONNREFUSED. So we listen on TCP and record where.
#
# TCP on a shared cluster means real auth: scram-sha-256 with a generated
# password, never trust. Local (same-node) connections stay on the socket.
PWFILE="${RUNDIR}/pg_password"
if [[ ! -s "${PWFILE}" ]]; then
  umask 077
  # tr -dc can exit 141 (SIGPIPE) when head closes the pipe early; `|| true`
  # keeps `set -o pipefail` from aborting the script on a successful read.
  { LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32 || true; } >"${PWFILE}"
  echo >>"${PWFILE}"
fi
chmod 600 "${PWFILE}"
PGPASS_VALUE="$(tr -d '\n' <"${PWFILE}")"

# Record the host serving this database so clients on other nodes can find it.
hostname -f >"${RUNDIR}/pg_host"

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
      --auth-host=scram-sha-256 \
      --encoding=UTF8 --locale=C >/dev/null
fi

# Allow compute nodes in. Restricted to RFC1918 ranges -- the cluster's internal
# network -- rather than 0.0.0.0/0, so this is never reachable from off-cluster
# even if the node has a public interface.
if ! grep -q 'regsearch-cluster-access' "${PGDATA}/pg_hba.conf" 2>/dev/null; then
  cat >>"${PGDATA}/pg_hba.conf" <<'HBA'

# regsearch-cluster-access
host    all    all    10.0.0.0/8       scram-sha-256
host    all    all    172.16.0.0/12    scram-sha-256
host    all    all    192.168.0.0/16   scram-sha-256
HBA
fi

echo "==> starting postgres"
nohup apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  postgres -D "${PGDATA}" \
    -p "${PGPORT_NUM}" \
    -c listen_addresses='*' \
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
      pg_isready -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    pg_isready -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" >/dev/null 2>&1; then
  echo "error: postgres did not become ready. Tail of ${LOGFILE}:" >&2
  tail -30 "${LOGFILE}" >&2
  exit 1
fi

# Idempotent database + extension setup.
if ! apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    psql -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='${PGDB_NAME}'" | grep -q 1; then
  apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
    createdb -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" "${PGDB_NAME}"
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  psql -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" -d "${PGDB_NAME}" -v ON_ERROR_STOP=1 \
  -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

# Sync the role password to the generated one. Passed via a psql variable, not
# string-interpolated into SQL, and over the local socket so it never crosses
# the network.
# Fed on stdin, not via -c: psql expands :'var' only when reading a script,
# never for a -c command string (it would send the literal ":'pw'" and error).
apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  psql -h "${RUNDIR}" -p "${PGPORT_NUM}" -U "${PGUSER_NAME}" -d "${PGDB_NAME}" \
  -v ON_ERROR_STOP=1 -v pw="${PGPASS_VALUE}" \
  >/dev/null <<SQL
ALTER ROLE ${PGUSER_NAME} WITH PASSWORD :'pw';
SQL

echo "postgres ready"
echo "  host   : $(cat "${RUNDIR}/pg_host")  port: ${PGPORT_NUM}"
echo "  socket : ${RUNDIR}  (same-node clients)"
echo "  db     : ${PGDB_NAME}  user: ${PGUSER_NAME}"
echo "  log    : ${LOGFILE}"
echo "  psql   : scripts/pg_psql.sh"
