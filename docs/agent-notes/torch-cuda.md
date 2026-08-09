# torch / CUDA driver mismatch on mccleary

Status: **diagnosed and fixed, fix verified end-to-end on a GPU node.**
Date: 2026-08-08. Cluster: Yale mccleary.

The deliverable in one line:

```
torch 2.13.0+cu126 on r107u12n01 (partition gpu, NVIDIA RTX A5000)
torch.cuda.is_available() -> True
```

Evidence: `.uv_cache/torchprobe/logs/verify-8867925.out`.

---

## 1. The facts

### Driver on the GPU nodes

Measured inside real Slurm jobs, not on the login node (this login node has no
GPU and no NVIDIA driver at all -- `nvidia-smi` there fails outright, so any
check run on it is meaningless).

| Job | Partition | Node | Driver | CUDA (nvidia-smi) | GPU |
|---|---|---|---|---|---|
| 8867695 | `gpu_devel` | r102u10n01 | 570.211.01 | 12.8 | RTX A5000, sm_86 |
| 8867925 | `gpu` | r107u12n01 | 570.211.01 | 12.8 | RTX A5000, sm_86 |

`NVRM version: NVIDIA UNIX x86_64 Kernel Module 570.211.01` (built 2025-11-25).

Driver 570.x is a **CUDA 12.8** driver. The `found version 12080` in torch's
warning is exactly this: 12080 = 12.8.0. The driver is *not* old or broken --
it is a current 12.x driver. The installed torch is simply from the next
major CUDA family.

### Why it breaks

CUDA guarantees *minor* version compatibility inside a major family: a binary
built against any 12.x runtime runs on any 12.x driver >= 525.60.13. It does
**not** guarantee compatibility across a major bump. `torch 2.13.0+cu130` is
built against CUDA **13.0**, which needs a **580.x or newer** driver. On a
570.x driver, `cudaGetDeviceCount` fails at init, torch swallows it as a
`UserWarning`, and `torch.cuda.is_available()` returns False. Nothing crashes;
the job just silently trains on CPU. That is how `training_meta.json` ended up
with `"device": "cpu"` after being submitted to a GPU node.

### How cu130 got installed in the first place

`uv.lock` resolves torch from the default index:

```
name = "torch"
version = "2.13.0"
source = { registry = "https://pypi.org/simple" }
```

`pyproject.toml` only says `torch>=2.3` in the `embed` extra, with no index
pinning. The **default PyPI linux wheel for torch 2.13.0 is the cu130 build**.
So a plain `uv sync --extra embed` silently installs a CUDA 13 torch. There is
no bug in the project's install step -- upstream's default just moved past this
cluster's driver.

### GPU inventory (all of it is fine for cu126)

```
gpu / gpu_devel / priority_gpu:  rtx5000 (sm_75), rtx3090 (sm_86),
                                 a5000 (sm_86), a100 (sm_80)
```

No Blackwell/sm_100 anywhere, so a cu126 build (which ships sm_50..sm_90
kernels) covers every GPU on the cluster. This matters: it means the fix is not
A5000-specific and will still work if a job lands on the A100 nodes.

### CUDA modules do not change the answer

`module avail` tops out at `CUDA/12.8.0` (and `cuDNN/9.5.1.17-CUDA-12.6.0`).
There is no CUDA 13 module, which is consistent with a site-wide 570.x driver.

More importantly, **modules are irrelevant here**. The pip torch wheels bundle
their own CUDA runtime (`nvidia-*-cu12` packages) and do not use a system CUDA
toolkit. Do *not* add `module load CUDA/...` to the sbatch scripts hoping to fix
this -- it will not, and it can only cause library-path confusion. The only
thing that matters is the driver, and the driver is fixed at 570.211.01.

---

## 2. The fix

Install the **cu126** build of the **same** torch version. Upstream publishes
`torch-2.13.0+cu126-cp311-cp311-manylinux_2_28_x86_64.whl`, so this is a pure
CUDA-build swap with **no torch version change**:

| index | newest cp311 torch |
|---|---|
| cu130 | 2.13.0 (currently installed -- too new for the driver) |
| **cu126** | **2.13.0** <- same version, correct CUDA family |
| cu128 | 2.11.0 only |

cu128 would match the driver's 12.8 exactly, but its newest wheel is torch
2.11.0 -- that would force a torch *downgrade* and drag `transformers`/
`sentence-transformers` resolution with it. cu126 avoids all of that and is
guaranteed to run on the 570.x driver by CUDA minor-version compatibility.
This was not taken on faith; it was measured (section 3).

### pyproject.toml change (NOT APPLIED -- for you to apply)

Keep `torch>=2.3` in the `embed` extra exactly as it is. Append two blocks:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]

# This cluster's GPU nodes run driver 570.211.01 (CUDA 12.8). The default PyPI
# wheel for torch >=2.13 is built against CUDA 13.0 and needs a 580.x driver, so
# it silently falls back to CPU here. Pin the CUDA build, not the version.
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

`explicit = true` is load-bearing: it means *only* torch is drawn from the
PyTorch index; every other dependency keeps coming from PyPI.

Then re-lock and sync:

```bash
uv lock
uv sync --extra embed
```

I ran that `uv lock` against a **throwaway copy** of `pyproject.toml`/`uv.lock`
in `/tmp` (the real files are untouched). It resolves cleanly:

```
Updated torch v2.13.0 -> v2.13.0+cu126
source = { registry = "https://download.pytorch.org/whl/cu126" }
```

Full diff of the resulting lock vs. the current one: **torch, plus its CUDA
runtime deps flipping from the `*-cu13` family to the `*-cu12` family**. That
is all. `sentence-transformers` (5.6.1), `transformers`, `numpy`, `scipy`,
`scikit-learn` and every non-CUDA package are byte-identical. There is no
sentence-transformers caveat -- the concern that pinning torch might drag it
backwards does not materialise, because the torch version does not move.

### One-liner alternative (if you prefer not to touch pyproject.toml)

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  "torch==2.13.0+cu126"
```

This works but is not reproducible -- the next `uv sync` will silently pull
cu130 back and put you right back on CPU. Prefer the pyproject change.

---

## 3. How the fix was verified

Constraint: the project `.venv` must not be touched (a benchmark and another
agent are live on it). So the whole check ran in a throwaway venv:

```
/vast/palmer/pi/garg/Patrick/regsearch/.uv_cache/torchprobe/venv
```

built with its own `UV_CACHE_DIR`, containing `torch==2.13.0+cu126` and
`sentence-transformers==5.6.1`. The project `.venv`, `pyproject.toml` and
`uv.lock` were never written to.

The check cannot run on the login node -- it has no GPU -- so it ran as a real
Slurm job on the `gpu` partition (job 8867925, 1 GPU, 10 min limit, finished in
well under that). Output, verbatim:

```
r107u12n01.mccleary.ycrc.yale.edu
name, driver_version
NVIDIA RTX A5000, 570.211.01
torch: 2.13.0+cu126
compiled cuda: 12.6
torch.cuda.is_available() -> True
device count: 1
device name: NVIDIA RTX A5000
capability: (8, 6)
matmul OK, 10x4096^3 in 0.149s, checksum 260041.6250
CrossEncoder param device: cuda:0
scores: [1.0562892 6.528191 ]
ALL GPU CHECKS PASSED
```

The test deliberately goes past `is_available()`, because that boolean alone
would not catch a wheel that loads but has no usable kernels for sm_86:

1. `torch.cuda.is_available()` -> True.
2. A real 4096x4096 matmul on the device, 10 iterations in 0.149 s, result
   summed back to host -- so kernels actually launch and produce numbers.
3. The project's own fine-tuned cross-encoder at `data/models/reranker` loaded
   via `sentence_transformers.CrossEncoder(..., device="cuda")`, parameters
   confirmed on `cuda:0`, and a real `predict()` on two query/passage pairs.
   The scores are sane and correctly ordered (the on-topic ATAC-seq passage
   scores 6.53, the off-topic GNN passage 1.06), so the fine-tuned weights load
   and score correctly under the cu126 build. The model directory was read
   only; `HF_HUB_OFFLINE=1` was set so nothing was fetched or rewritten.

## 4. Follow-ups for whoever applies this

- `slurm/eval.sbatch` (owned by another agent) carries a comment block at lines
  24-34 explaining why it deliberately has no `#SBATCH --gpus=1`, ending with
  "If the torch/driver pair is [fixed]...". That premise is now false and the
  `--gpus=1` line can be added once the sync lands. I did not edit that file.
- The reranker fine-tune that produced `"device": "cpu"` (275 steps / 17 min) is
  worth re-running on GPU, both for the speedup and because CPU and GPU runs are
  not bitwise comparable.
- Nothing needs to be escalated to YCRC. The driver is current and healthy; this
  was purely a client-side wheel selection problem.

### Cleanup

The throwaway venv is ~6.2 GB against the group quota. Once you have applied and
sync'd the fix, delete it:

```bash
rm -rf /vast/palmer/pi/garg/Patrick/regsearch/.uv_cache/torchprobe
```

Keep it until then if you want to reproduce the check yourself -- rebuilding it
means re-downloading torch. The Slurm logs under `.uv_cache/torchprobe/logs/`
are the evidence for everything above.
