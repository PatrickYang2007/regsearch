#!/usr/bin/env bash
#SBATCH --job-name=regsearch-rerank-ft
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/rerank-ft-%j.out
#SBATCH --error=slurm/logs/rerank-ft-%j.err

# Fine-tune the cross-encoder reranker on citation-derived WEAK supervision and
# write the checkpoint to data/models/reranker/. rerank.py prefers that path and
# only falls back to the public ms-marco checkpoint when config.json is absent
# there, so the hybrid_rerank arm switches over as soon as this job finishes --
# re-run `regsearch eval` afterwards to get the trained row into the table.
#
# HOW THIS JOB REACHES POSTGRES
# -----------------------------
# Postgres runs on the node that ran scripts/pg_start.sh, which is NOT this
# node. The connection is plain TCP: pg_start.sh records its own hostname in
# data/run/pg_host, config.Settings.resolved_host reads that file, and the DSN
# dials host:5432 with scram-sha-256 (password in data/run/pg_password, 0600).
# pg_hba is restricted to RFC1918 ranges, so it is reachable from compute nodes
# and not from off-cluster.
#
# It specifically does NOT go over the Unix socket. A Unix socket is local IPC:
# the socket file sits on /vast and is visible from here, but connecting to it
# from another host is impossible, and trying cost two failed GPU jobs in
# session 1. Same-node clients still use the socket; cross-node ones cannot.
#
# The database must already be up when this starts -- the preflight below exits
# rather than burning GPU time on a job that cannot read its training data.
# Run scripts/pg_start.sh (idempotent) BEFORE sbatch, not after.
#
# WHERE THE TIME GOES
# -------------------
# Hard-negative mining runs one real hybrid search per training query (619 of
# them) to collect the candidates the reranker will actually face at inference.
# `hybrid` p50 is ~1.4 s on this corpus, dominated by the lexical arm, so
# expect ~15-25 min of Postgres-bound wall clock before the first gradient
# step. The training itself is minutes on an A100. The 2 h limit is mostly
# headroom for the mining phase.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export UV_CACHE_DIR="${ROOT}/.uv_cache"
export HF_HOME="${ROOT}/data/hf"          # keep model weights off the home quota
export TOKENIZERS_PARALLELISM=false        # silences a fork warning under DataLoader

# Overridable at submit time (sbatch exports the submitting environment), e.g.
#   EPOCHS=2 MAX_NEG=16 sbatch slurm/finetune_rerank.sbatch
: "${EPOCHS:=1}"
: "${BATCH_SIZE:=32}"
: "${LR:=2e-5}"
: "${MAX_NEG:=8}"
: "${MAX_POS:=8}"
: "${CANDIDATE_K:=50}"
: "${SKIP_TOP_N:=0}"     # drop the head of the pool; false negatives cluster there
: "${LIMIT:=}"           # empty = all 619 training queries

# `: "${X:=default}"` assigns a SHELL variable, which a child process does not
# inherit. The training block below is a separate `python` process reading
# os.environ, so without this export it died on KeyError: 'EPOCHS' -- after
# passing preflight and allocating a GPU, which is the expensive way to fail.
export EPOCHS BATCH_SIZE LR MAX_NEG MAX_POS CANDIDATE_K SKIP_TOP_N LIMIT

echo "node   : $(hostname)"
echo "gpu    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "pg host: $(cat data/run/pg_host 2>/dev/null || echo '<unset>')"
echo "params : epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LR} neg=${MAX_NEG} pos=${MAX_POS} k=${CANDIDATE_K}"

if ! ./scripts/pg_psql.sh -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "error: cannot reach Postgres. Run scripts/pg_start.sh first." >&2
  exit 1
fi

# Invoked as a script rather than `regsearch finetune-rerank` because the CLI
# command is being wired separately; swap this block for the CLI call once it
# lands. The Python entry point (train_reranker) is stable either way.
uv run --extra embed python - <<'PY'
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from regsearch.retrieve.train_rerank import train_reranker

limit = os.environ.get("LIMIT") or None

meta = train_reranker(
    epochs=int(os.environ["EPOCHS"]),
    batch_size=int(os.environ["BATCH_SIZE"]),
    lr=float(os.environ["LR"]),
    max_negatives_per_query=int(os.environ["MAX_NEG"]),
    limit=int(limit) if limit else None,
    max_positives_per_query=int(os.environ["MAX_POS"]),
    candidate_k=int(os.environ["CANDIDATE_K"]),
    skip_top_n_negatives=int(os.environ["SKIP_TOP_N"]),
)
print(json.dumps(meta, indent=2))
PY

echo "done -- checkpoint at data/models/reranker"
ls -la data/models/reranker

# Sanity check that the arm will actually pick the new checkpoint up rather than
# silently falling back to the public one. Cheap, and the failure mode it
# catches (a save that wrote no config.json) is otherwise invisible until the
# next eval quietly reports baseline numbers again.
uv run --extra embed python -c "
from regsearch.retrieve.rerank import model_path
p = model_path()
print('reranker in use:', p)
assert 'data/models/reranker' in p, 'fine-tuned checkpoint NOT picked up'
"

echo
echo "next: regsearch eval --split test --origin citation --out docs/ablation.md"
