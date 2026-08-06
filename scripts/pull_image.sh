#!/usr/bin/env bash
# Pull the Postgres 17 + pgvector image into containers/.
# Cache goes under the project (not $HOME, which is quota-limited on the cluster).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export APPTAINER_CACHEDIR="${ROOT}/.apptainer_cache"
mkdir -p "${APPTAINER_CACHEDIR}" "${ROOT}/containers"
apptainer pull --force "${ROOT}/containers/pgvector.sif" docker://pgvector/pgvector:pg17
echo "pulled -> ${ROOT}/containers/pgvector.sif"
