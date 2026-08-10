#!/usr/bin/env bash
#SBATCH --job-name=regsearch-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/eval-%j.out
#SBATCH --error=slurm/logs/eval-%j.err

# Run the four-arm ablation off the interactive node and write the markdown
# table to docs/ablation.md (override with OUT=).
#
# WHY THIS IS A BATCH JOB
# -----------------------
# `regsearch eval --split test --origin citation` takes ~15 min on the 1-CPU
# OnDemand allocation and pins that CPU for the whole time, which blocks every
# other thing you want to do in the session. The cost is concentrated in one
# arm: hybrid_rerank runs a cross-encoder forward pass over rerank_topk=100
# candidates for each of the 171 test queries, i.e. ~17,100 pairs. That is
# torch matmul work and it is the only part of the eval that scales with cores.
#
# GPU, AS OF THE cu126 FIX
# ------------------------
# This was deliberately a CPU job for as long as the venv carried a cu130 torch
# that this cluster's 570.x driver (CUDA 12.8) rejects at init -- `is_available()`
# returned False, so a GPU allocation would have sat idle while the cross-encoder
# ran on CPU anyway. pyproject now pins the cu126 build of the same torch
# version, verified on a real A5000 (job 8907907, evidence in
# docs/agent-notes/evidence/): `torch.cuda.is_available() -> True`, and the
# fine-tuned CrossEncoder loads on cuda:0.
#
# rerank.py:53 picks its device with exactly that call, so this arm now runs on
# the GPU. cpus-per-task drops from 8 to 2: the cross-encoder forward passes are
# no longer the CPU's problem.
#
# LATENCY NUMBERS ARE NOT COMPARABLE ACROSS THIS CHANGE. The p50 columns in
# docs/ablation.md were measured on 8 CPU cores; a GPU run will beat them for
# reasons that have nothing to do with retrieval quality. Recall/nDCG/MRR are
# device-independent and remain comparable.
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
# from another host is impossible, and trying cost two failed jobs in session
# 1. Same-node clients still use the socket; cross-node ones cannot.
#
# The database must already be up when this starts, AND it must stay up for the
# whole run -- every arm queries it live. If the allocation hosting Postgres
# ends mid-job the eval dies with a connection error. Run scripts/pg_start.sh
# (idempotent) BEFORE sbatch, not after.
#
# HF_HOME IS DELIBERATELY NOT SET HERE
# ------------------------------------
# Unlike embed.sbatch and finetune_rerank.sbatch, this file does not export
# HF_HOME. src/regsearch/config.py line 29 does
# `os.environ.setdefault("HF_HOME", PROJECT_ROOT/"data"/"hf")` at import time,
# and every path into the models imports regsearch.config first, so the weights
# resolve to data/hf without help. Verified:
#
#   $ .venv/bin/python -c 'import os, regsearch.db.client; print(os.environ["HF_HOME"])'
#   /vast/palmer/pi/garg/Patrick/regsearch/data/hf
#
# Both models are already cached there, so this runs offline on a compute node.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export UV_CACHE_DIR="${ROOT}/.uv_cache"
export TOKENIZERS_PARALLELISM=false        # silences a fork warning under DataLoader

# Overridable at submit time (sbatch exports the submitting environment), e.g.
#   ARMS=hybrid,hybrid_rerank OUT=docs/ablation_rerank.md sbatch slurm/eval.sbatch
#   CANONICALIZE=false OUT=docs/ablation_nocanon.md sbatch slurm/eval.sbatch
: "${SPLIT:=test}"
: "${ORIGIN:=citation}"
: "${ARMS:=fts,dense,hybrid,hybrid_rerank}"
: "${OUT:=docs/ablation.md}"
: "${CANONICALIZE:=true}"   # false -> --no-canonicalize (twins score as misses)
: "${VERBOSE:=false}"

# `: "${X:=default}"` assigns a SHELL variable, which a child process does not
# inherit. These particular five are interpolated into the argv of the command
# below, so they would survive without exporting -- but OMP_NUM_THREADS and
# friends below are read out of os.environ by a child python, and finetune_rerank
# .sbatch died on KeyError: 'EPOCHS' after allocating a GPU by getting exactly
# that wrong. Export everything that crosses the boundary; the argv ones cost
# nothing to include and stop the next person from re-learning it.
export SPLIT ORIGIN ARMS OUT CANONICALIZE VERBOSE

# Thread count for the cross-encoder. torch's default is "however many cores I
# can see", which on a shared compute node is the whole machine, not this job's
# cgroup -- oversubscribing to 64 threads on 8 allocated cores is measurably
# slower than 8 and irritates everyone else on the node. Pin it to the
# allocation. Falls back to 1 when run outside Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

case "${CANONICALIZE}" in
  true|1|yes)  CANON_FLAG="--canonicalize" ;;
  false|0|no)  CANON_FLAG="--no-canonicalize" ;;
  *) echo "error: CANONICALIZE must be true or false, got '${CANONICALIZE}'" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
case "${VERBOSE}" in
  true|1|yes) EXTRA_ARGS+=(--verbose) ;;
esac

mkdir -p "$(dirname "${OUT}")"

echo "node    : $(hostname)"
echo "cpus    : ${SLURM_CPUS_PER_TASK:-<none>} (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
echo "pg host : $(cat data/run/pg_host 2>/dev/null || echo '<unset>')"
echo "params  : split=${SPLIT} origin=${ORIGIN} arms=${ARMS} ${CANON_FLAG}"
echo "out     : ${OUT}"

if ! ./scripts/pg_psql.sh -tAc "SELECT 1" >/dev/null 2>&1; then
  echo "error: cannot reach Postgres. Run scripts/pg_start.sh first." >&2
  echo "       (it must stay up for the whole run -- every arm queries it live)" >&2
  exit 1
fi

# Announce which reranker the hybrid_rerank row will actually mean. The
# fine-tuned checkpoint is picked up silently the moment data/models/reranker/
# contains a config.json, so a table generated before and after a fine-tune are
# not comparable and nothing in the table itself says so.
uv run --extra embed python -c "
from regsearch.retrieve.rerank import model_path
print('reranker in use:', model_path())
"

# --extra embed, not a bare `uv run`: the dense and hybrid_rerank arms need
# sentence-transformers, which lives in the optional extra. Without it the eval
# gets partway through `fts` and then fails on the import.
uv run --extra embed regsearch eval \
  --split "${SPLIT}" \
  --origin "${ORIGIN}" \
  --arms "${ARMS}" \
  --out "${OUT}" \
  "${CANON_FLAG}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "done -- wrote ${OUT}"
cat "${OUT}"

echo
echo "next: update the same table in README.md and drop any stale markers."
