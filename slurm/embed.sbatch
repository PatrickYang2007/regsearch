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
# Postgres runs on the submitting node, not here, so this job connects over the
# shared-filesystem Unix socket. That works because /vast is visible from every
# node -- but it does mean the database must still be up when this job starts.
# Check with: scripts/pg_start.sh (idempotent).

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
