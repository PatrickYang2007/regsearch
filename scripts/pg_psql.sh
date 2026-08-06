#!/usr/bin/env bash
# Open psql against the containerised Postgres. Extra args pass through:
#   scripts/pg_psql.sh -c "SELECT count(*) FROM passages;"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec apptainer exec --cleanenv --bind "${ROOT}:${ROOT}" "${ROOT}/containers/pgvector.sif" \
  psql -h "${REGSEARCH_RUN_DIR:-${ROOT}/data/run}" \
       -U "${REGSEARCH_PG_USER:-regsearch}" \
       -d "${REGSEARCH_PG_DB:-regsearch}" "$@"
