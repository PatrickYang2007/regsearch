#!/usr/bin/env bash
#SBATCH --job-name=regsearch-embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/embed-%j.out
#SBATCH --error=slurm/logs/embed-%j.err

# Encode every passage lacking an embedding, then build the HNSW index.
#
# Postgres runs on the node that ran scripts/pg_start.sh, not here, so this job
# connects over plain TCP: pg_start.sh records its hostname in
# data/run/pg_host, config.Settings.resolved_host reads it, and the DSN dials
# host:5432 with scram-sha-256 (password in data/run/pg_password, 0600).
# pg_hba is restricted to RFC1918 ranges, so it is reachable from compute nodes
# and not from off-cluster.
#
# NOT a Unix socket. The socket file lives on /vast and is visible from every
# node, but a Unix socket is local IPC and cannot be connected to across hosts
# -- that misreading cost two failed GPU jobs in session 1. Only same-node
# clients use the socket.
#
# The database must still be up when this job starts: scripts/pg_start.sh
# (idempotent), before sbatch.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export UV_CACHE_DIR="${ROOT}/.uv_cache"
export HF_HOME="${ROOT}/data/hf"          # keep model weights off the home quota
export TOKENIZERS_PARALLELISM=false        # silences a fork warning under DataLoader

echo "node   : $(hostname)"
echo "gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

if ! ./scripts/pg_psql.sh -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "error: cannot reach Postgres. Run scripts/pg_start.sh first." >&2
  exit 1
fi

uv run --extra embed regsearch embed --verbose
uv run regsearch build-index

echo "done"
uv run regsearch stats
