# Agent notes: `slurm/eval.sbatch` + `scripts/start.sh`

Session 4, 2026-08-08. Files touched: `slurm/eval.sbatch` (new),
`scripts/start.sh` (fixed + chmod +x). **Nothing submitted, nothing committed.**

---

## 1. Why the eval moved to a batch job

`regsearch eval --split test --origin citation` costs ~15 min on the 1-CPU
OnDemand allocation and holds that CPU the whole time. Essentially all of it is
one arm: `hybrid_rerank` runs a cross-encoder over `rerank_topk=100` candidates
for each of 171 test queries (~17,100 pairs). That is the only part of the eval
that scales with cores, so 8 CPUs on a `day` node is the right shape.

## 2. No GPU — verified, not assumed

The instruction said torch cannot init CUDA here. Confirmed on this node:

```
$ .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080).
torch 2.13.0+cu130
cuda available: False
torch.version.cuda 13.0
```

`retrieve/rerank.py` picks its device with exactly `torch.cuda.is_available()`,
so on a GPU node the CrossEncoder would still be built on CPU and the GPU would
sit idle. `#SBATCH --gpus=1` is deliberately absent and the reasoning is written
into the file so nobody "fixes" it later.

## 3. HF_HOME claim — verified true, so it is NOT set

`src/regsearch/config.py:29` runs
`os.environ.setdefault("HF_HOME", PROJECT_ROOT/"data"/"hf")` at import, and
every path to the models imports `regsearch.config` first:

```
$ .venv/bin/python -c "import os, regsearch.db.client; print(os.environ['HF_HOME'])"
/vast/palmer/pi/garg/Patrick/regsearch/data/hf
```

So unlike `embed.sbatch` / `finetune_rerank.sbatch`, this file does not export
it. The divergence is intentional and commented. (Those two are not wrong,
merely redundant — and they predate the config change, commit `62647dc`.)

## 4. The `export` trap

`finetune_rerank.sbatch` originally set `: "${EPOCHS:=1}"` without exporting,
and its heredoc `python` died on `KeyError: 'EPOCHS'` *after* allocating a GPU.

In `eval.sbatch` the five tunables (`SPLIT ORIGIN ARMS OUT CANONICALIZE`) are
interpolated into **argv**, so they would technically survive unexported. They
are exported anyway, alongside `OMP_NUM_THREADS`/`MKL_NUM_THREADS`, which *are*
read out of `os.environ` by the child. Verified with a stub `uv` on `PATH` that
prints its argv and environment — every variable arrives.

## 5. Thread pinning

torch's default thread count is "cores I can see", which on a shared compute
node is the whole machine rather than the cgroup. Oversubscribing 64 threads
onto 8 allocated cores is slower than 8 and hurts co-tenants, so
`OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}`. (On the interactive node
`torch.get_num_threads()` already reports 1 — the cgroup there really is 1 CPU.)

## 6. `--extra embed` is mandatory

`uv run regsearch eval` without it gets through the `fts` arm and then dies on
the `sentence_transformers` import: it lives in the optional `embed` extra
(`pyproject.toml`). Both `uv` invocations in the file carry it.

## 7. Test results (stubbed `uv`, real preflight)

| test | result |
|---|---|
| `bash -n slurm/eval.sbatch` | OK |
| defaults → argv | `eval --split test --origin citation --arms fts,dense,hybrid,hybrid_rerank --out docs/ablation.md --canonicalize` |
| all overrides → argv | `--split train --origin manual --arms dense,hybrid_rerank --out <path> --no-canonicalize --verbose` |
| `CANONICALIZE=maybe` | `error: CANONICALIZE must be true or false, got 'maybe'`, exit 2 |
| Postgres unreachable (simulated `REGSEARCH_RUN_DIR`) | preflight message, exit 1, **before** any `uv` call |
| empty `EXTRA_ARGS` under `set -u` | fine — uses `${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}` |

### Gotcha found while testing

The OnDemand interactive shell already carries
`SLURM_SUBMIT_DIR=/var/www/ood/apps/sys/dashboard`. Running the file directly
(`bash slurm/eval.sbatch`) from that shell therefore dies on
`cd: /var/www/ood/apps/sys/dashboard: No such file or directory`. Not a bug in
the job: `sbatch` overwrites `SLURM_SUBMIT_DIR` with the submission cwd. But it
means **you cannot smoke-test these sbatch files by executing them** in an
OnDemand terminal without unsetting it first. `embed.sbatch` and
`finetune_rerank.sbatch` share the behaviour.

## 8. `scripts/start.sh` — three bugs found and fixed

1. **File was mode 644.** `./scripts/start.sh` returned
   `Permission denied` (exit 126) instead of the "must be sourced" guard —
   i.e. the guard was unreachable by the exact mistake it exists to catch.
   Fixed with `chmod +x`; every other script in `scripts/` is already 755.
2. **`_regsearch_root` leaked into the caller's shell on every failure path.**
   Sourcing has no scope of its own, so a `return 1` before the trailing
   `unset` left the variable set in that terminal forever. All failure returns
   now unset first.
3. Nothing else. `set -euo pipefail` is genuinely absent (grep only hits the
   comment explaining why), the `BASH_SOURCE`-based root resolution genuinely
   works from a foreign cwd, and re-sourcing is genuinely idempotent —
   all three verified by execution, not by reading.

Verified after fixing: sourced from `/tmp` → lands in the project root, venv
active, `regsearch` on `PATH` exactly once even after a double source,
`errexit/nounset/pipefail` all still `off` in the caller, `_regsearch_root`
unset, and a subsequent `false` does not kill the shell.

## 9. Handover

**Not submitted.** Patrick submits it himself once the in-flight benchmark
finishes:

```bash
scripts/pg_start.sh            # must be up AND stay up for the whole run
sbatch slurm/eval.sbatch
```

Postgres must survive the entire job, not just the preflight — every arm queries
it live, so if the allocation hosting it ends mid-run the eval dies partway.
