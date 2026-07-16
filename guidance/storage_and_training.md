# Storage and Training Operations for SD3.5 GenEval

This guide defines where Flow-Factory stores code, software, models, datasets,
caches, checkpoints, logs, and secrets for the SD3.5 DiffusionNFT GenEval
experiments on Pluto. It also follows a training job from a blank node through
checkpoint recovery.

Two designs appear in this document:

- **Current fallback:** durable inputs live on Sensei FS and
  [`prepare_flowfactory_runtime.sh`](../scripts/prepare_flowfactory_runtime.sh)
  stages a runnable copy on local SSD.
- **Production target:** a versioned image or Pluto Code Distribution supplies
  code and dependencies, approved artifact storage supplies models and data,
  local SSD remains the active workspace, and S3 stores completed checkpoints.

Think of the storage tiers as a workshop. The image is the toolbox, Sensei FS
is a shared library, local SSD is the workbench, and S3 is the archive. A
training process should work on the bench, not in the library or the archive.

## Decision

The current local-SSD design is the right runtime approach for SD3.5. Running
everything directly from Sensei FS would remove the staging step, but it would
put model loading, Python imports, Arrow preprocessing, logs, and checkpoint
writes on a shared filesystem that the platform does not recommend for those
access patterns.

| Current choice | Assessment |
|---|---|
| Copy SD3.5 and OpenCLIP to local SSD | Recommended |
| Build Arrow embeddings on local SSD | Required by the configured distributed local preprocessing mode |
| Run staged code and installed packages locally | Recommended for the current fallback |
| Write checkpoints locally, then publish them to S3 | Recommended |
| Keep durable source models and datasets on Sensei FS temporarily | Acceptable fallback |
| Keep production code and a production Conda environment on Sensei FS | Not recommended |
| Run the whole experiment directly from Sensei FS | Avoid |

The platform's
[`file-system.md`](/sensei-fs-3/users/yunfeix/ai-platform-docs/file-system.md)
assigns S3 to checkpoints and backups, local SSD to active model files and
training datasets, the root EBS filesystem to code and executable builds, and
Sensei FS to shared datasets and collaboration. Its Sensei FS best-practices
section warns against code, Conda environments, checkpoints, executables,
shared libraries, and large collections of small files on Sensei FS.

Sensei FS is FSx for Lustre with an S3-backed cache. It is not Ceph. Its backing
S3 is an implementation detail and is not the run-specific S3 prefix used to
publish Flow-Factory checkpoints.

## Storage contract

This table is the canonical assignment of storage responsibilities.

| Tier | Use it for | Do not rely on it for | Lifetime |
|---|---|---|---|
| Versioned Docker image | OS, CUDA-compatible runtime, pinned Python packages, MMCV build, and optionally immutable code | Datasets, generated caches, logs, secrets, or checkpoints in the writable container layer | Reusable image layers; container writes are disposable |
| Root EBS filesystem | Interactive builds or code when local SSD is not appropriate | Unpublished run results | Until termination, unless a job snapshot is explicitly saved; `/code` from Code Distribution is not snapshotted |
| Local SSD at `/mnt/localssd` | Active code, environment, model snapshots, datasets, Arrow embeddings, Torch and Triton caches, temporary files, W&B files, logs, and in-progress checkpoints | The only copy of anything needed after node loss | Until the job terminates |
| Sensei FS | Shared source datasets, collaboration, and temporary durable bootstrap inputs | Production code, environments, active caches, frequent checkpoint writes, or personal copies of shared datasets | Persistent, quota-limited, with cold-file recall after eviction |
| S3 or approved artifact storage | Completed checkpoints, model and dataset backups, immutable source artifacts | Direct high-frequency training I/O | Durable |
| Vault and Pluto resource-access roles | W&B tokens and cloud credentials | Secret values committed to `.env`, images, code, or checkpoints | Managed outside the repository |
| Logging FS | Platform-managed job logs only | User data of any kind | Platform-managed |

The root filesystem can be useful, but this repository's fallback uses local
SSD for the active code and environment so all high-frequency reads share the
same node-local workspace. In production, a versioned image can eliminate most
of that bootstrap work.

## What goes where

| Asset | Current durable source | Active location | Production source | Must survive node loss? |
|---|---|---|---|---|
| Flow-Factory code | Personal Sensei checkout | `/mnt/localssd/flowfactory/code/Flow-Factory` | Versioned image or Pluto GitHub Code Distribution | Yes, through Git or the image |
| Python packages | Pinned Miniforge installer plus repository requirement files | `/mnt/localssd/flowfactory/miniforge` and `env` | Versioned certified image | Yes, through the image or lock files |
| CUDA libraries | Pinned PyTorch packages in the local environment | Resolved from `/mnt/localssd/flowfactory/env` plus the platform driver | Certified image and platform driver | Yes, through the image |
| MMCV CUDA operations | Prebuilt `sm80-sm90` wheel in the source checkout | Installed into the local environment | Built into and tested with the versioned image | Yes, through the image or artifact store |
| SD3.5 Medium | Personal Sensei Hugging Face cache | `/mnt/localssd/flowfactory/cache/huggingface` | Approved S3 or model artifact | Yes, at the canonical source |
| OpenCLIP ViT-L/14 | Personal Sensei Hugging Face cache | `/mnt/localssd/flowfactory/cache/huggingface` | Approved S3 or model artifact | Yes, at the canonical source |
| GenEval prompt data | Personal Sensei dataset path | `/mnt/localssd/flowfactory/data` | Sensei tenant directory or approved S3 dataset artifact | Yes |
| No-rewrite control data | Regenerated from the staged GenEval data | `/mnt/localssd/flowfactory/data` | Versioned dataset artifact or deterministic regeneration | Its source and manifest must survive |
| Arrow embeddings | None; deterministic cache | `/mnt/localssd/flowfactory/cache/flow_factory/datasets` | Rebuild on each node or stage a compatible versioned cache | No |
| Mask2Former detector weights | Pinned OpenMMLab URL and SHA-256 | Local Torch cache | Pre-staged approved artifact | Yes, at the canonical source |
| W&B cache, media, and offline files | None | `/mnt/localssd/flowfactory/cache/wandb` and `runs/wandb` | Same local paths, with online sync when permitted | Only records not already synced |
| Logs and temporary files | None | `/mnt/localssd/flowfactory/logs` and `tmp` | Same local paths | Only logs needed for diagnosis |
| Training checkpoints | None | `/mnt/localssd/flowfactory/runs/<run>/checkpoints` | Same local write path, then a run-specific S3 prefix | Yes |

Personal Sensei paths are fallback inputs, not the production ownership model.
Shared datasets belong in a tenant directory, not under
`/sensei-fs-3/users/<name>`.

## Current node-local layout

The prepared node has this logical shape:

```text
/mnt/localssd/flowfactory/
├── code/Flow-Factory
├── miniforge
├── env
├── python-overlay
├── cache/
│   ├── huggingface
│   ├── flow_factory/datasets
│   ├── torch
│   ├── triton
│   ├── wandb
│   ├── pip
│   ├── bootstrap
│   └── conda/pkgs
├── data
├── build/geneval
├── runs
│   └── wandb
├── logs
├── tmp
├── home
└── .ready and component readiness files
```

`build/geneval` is used only if a compatible MMCV wheel is unavailable.
`python-overlay` is reserved for local overrides. The launcher puts it before
the staged source tree on `PYTHONPATH`.

The `.env` file is the routing table. It identifies the durable input paths,
the local runtime root, the selected experiment config, cache locations,
offline behavior, and checkpoint publication settings. It must not contain
long-lived cloud credentials or API tokens.

## End-to-end training pipeline

### 1. Allocate a node

A new Pluto node starts with no prepared Flow-Factory workspace on its local
SSD. The SSD is fast and disposable. On current platform instance types, the
documented capacity is 478 GB for H100 and 530 GB for A100 nodes.

Before staging, confirm that the chosen experiment fits. Historical planning
measurements for this setup were:

| Item | Planning size |
|---|---:|
| SD3.5 and OpenCLIP model cache | 37.2 GB measured |
| Rewritten training Arrow cache | About 105 GB measured in an earlier build |
| No-rewrite Arrow cache | Up to about 105 GB estimated |
| Stock GenEval Arrow cache | About 69 GB estimated |

One rewritten run may therefore occupy about 142 GB before the environment,
checkpoints, logs, W&B files, and temporary space. Keeping all three Arrow
variants could approach 316 GB before those additions. These values are
observations and estimates, not stable format guarantees. Check free space on
the target node before building a second or third variant.

### 2. Prepare the cold node

Run the preparation script from the durable checkout:

```bash
cd /sensei-fs-3/users/yunfeix/origin_flowfactory/Flow-Factory
./scripts/prepare_flowfactory_runtime.sh
```

The script performs these operations:

1. Loads `.env` and validates the pinned Miniforge installer, model snapshots,
   datasets, CUDA settings, and the MMCV wheel configuration.
2. Refuses to replace the staged runtime while a Flow-Factory trainer is
   active.
3. Creates `/mnt/localssd/flowfactory` and redirects `HOME`, XDG, pip, Torch,
   Triton, W&B, and temporary writes below it.
4. Copies the checkout to `code/Flow-Factory`, excluding `.git`, `.scratch`,
   prior saves, W&B history, bytecode, and `__pycache__`. It also copies `.env`.
5. Downloads and verifies a pinned 105 MB Miniforge installer, installs it as
   `miniforge`, and uses that local Conda to create a Python 3.12 `env`.
6. Installs pinned packages and the editable staged checkout into `env`, then
   installs or builds MMCV and validates a real MMCV CUDA operation.
7. Copies SD3.5 and OpenCLIP into the local Hugging Face cache, downloads only
   the tokenizer/processor files (not redundant model weights) for PickScore's
   LAION processor repository, and downloads the pinned reward checkpoints.
8. Copies the rewritten GenEval dataset, verifies source hashes, generates the
   row-preserving no-rewrite control, and builds the row-aligned Pick-a-Pic
   rewrite JSONL with a validated manifest.
9. Writes component readiness files and writes `.ready` last.

The environment build, both model copies, dataset copy, and detector download
run in parallel. This turns preparation into a prewarm barrier: the slowest
independent task determines most of the wall time instead of adding every copy
and installation serially.

The code replacement uses a temporary directory followed by removal and rename.
It prevents a partially copied tree from becoming the final tree, but it is not
an atomic directory exchange. Do not run preparation concurrently with a
launcher.

The fallback software layer is now self-contained for training-time imports.
`sys.prefix`, `sys.base_prefix`, the Python standard library, installed
packages, editable checkout, and configured CUDA library path all resolve
under `/mnt/localssd/flowfactory`. The Sensei CUDA toolkit remains a preparation
input only if MMCV must be built from source; the supplied wheel avoids that
build on normal nodes.

Model readiness files currently prove that a copy completed, not that every
model file matches a content hash. Dataset and dependency readiness checks are
stronger because they use hashes or fingerprints.

### 3. Validate durability before launch

`CHECKPOINT_S3_URI` must point to an authorized, run-specific prefix before a
run can be considered recoverable from node loss. Credentials should come from
the Pluto resource-access role. Do not place AWS keys in `.env`.

The launcher deliberately permits a blank S3 URI for local validation, but it
prints a warning. Local-only checkpoints disappear when the node terminates.
The configured local retention policy still runs when S3 publication is
disabled, so a blank URI combined with retention of two checkpoints leaves only
the newest two local recovery points.

### Standardized prewarm and launch workflow

Treat node setup like warming an oven: do it once after allocation, verify it,
then run as many recipes as fit the same software and asset contract.

```bash
cd /sensei-fs-3/users/yunfeix/origin_flowfactory/Flow-Factory

# One-time prewarm on an empty node; idempotent on a warm node.
./scripts/prepare_flowfactory_runtime.sh

# Read-only proof that code, Python, stdlib, models, and data are local.
./scripts/check_flowfactory_runtime.sh

# A maintained experiment wrapper.
./scripts/train_nft_geneval_baseline.sh
```

Preparation must be rerun after code or dependency changes. On the same node,
unchanged model and dataset readiness markers prevent another 38 GB copy, and
the dependency fingerprint prevents unnecessary environment rebuilds.

The preparer still refreshes the small staged checkout on every invocation.
On this Sensei mount, reading that roughly 38 MB/440-file tree took more than
three minutes during the measured cold run. Do not put the preparer in every
training launch: run it once as a node prewarm barrier, use the read-only
checker before each experiment, and rerun preparation only when inputs change.

For another Flow-Factory YAML, reuse the same runtime without writing a new
bootstrap script:

```bash
./scripts/run_in_flowfactory_runtime.sh -- \
  ff-train examples/another/experiment.yaml
```

For the AdvantageFlow paper-aligned Pick-a-Pic multi-reward DiffusionNFT run,
the maintained wrapper is:

```bash
./scripts/train_nft_pickapic_multi_reward.sh
```

### Privileged-prompt distillation (PPD) arms

The PPD arms (see the PPD section of
[`algorithms.md`](algorithms.md)) add one storage input on top of a prepared
runtime: a row-keyed privileged-prompt records file per baseline. Records are
staged by a dedicated idempotent step that runs **after** the preparer,
because the stock-GenEval records are derived from the staged repository's
`dataset/geneval/train.jsonl` and both files must live on local SSD before a
launcher may start:

```bash
# After prepare + check. Rerun after any preparer rerun (staging is cheap).
./scripts/prepare_ppd_records.sh

# Auxiliary arms (CONFIG=... selects the matched rho=0 control instead).
./scripts/train_nft_geneval_stock_ppd.sh
./scripts/train_nft_pickapic_multi_reward_ppd.sh
```

The staged records land under `data/geneval_stock_ppd_pairs/` (built by
`scripts/build_geneval_stock_ppd_records.py` with a SHA-256 manifest; 100%
prompt coverage, 37.99% changed rows) and `data/pickapic_balanced_v0_pairs/`
(the legacy balanced_v0 pairs normalized to PPD schema v1; 87.37% changed
rows). Both are tens of megabytes — negligible next to the Arrow caches. The
PPD launchers fail fast when records are missing, and trainer initialization
independently validates every record against the training prompts.

PPD adds no new durable outputs: checkpoints, logs, and W&B artifacts follow
the same local-SSD-then-publish contract as the baselines. The matched-arm
discipline is operational, not just statistical — run the `rho: 0.0` control
with the same seed and confirm `ppd/control_zero == 0` at every step before
trusting an auxiliary/control comparison.

Its config is
[`pickapic_multi_reward.yaml`](../examples/nft/lora/sd3_5/pickapic_multi_reward.yaml).
It uses the original 25,432/2,048-prompt AdvantageFlow Pick-a-Pic split, SD3.5
Medium, equal-weight PickScore + HPSv2.1 + CLIPScore, 32 prompts with four
images each, 10-step ODE sampling, nine NFT training timesteps, 40-step eval,
rank-32/alpha-64 LoRA, and one 128-sample optimizer update per outer epoch.

After the baseline is running, the controlled balanced-v0 treatment uses:

```bash
# Run preparation after the baseline stops so the new code and dataset can be
# staged safely. The preparer refuses to modify an active training runtime.
./scripts/prepare_flowfactory_runtime.sh
./scripts/train_nft_pickapic_multi_reward_rewrite.sh
```

The treatment config is
[`pickapic_multi_reward_rewrite_balanced_v0.yaml`](../examples/nft/lora/sd3_5/pickapic_multi_reward_rewrite_balanced_v0.yaml).
It is intentionally identical to the baseline config except for `dataset_dir`
and the W&B run name. The builder reads the frozen 25,432-row artifact at
`PICKAPIC_REWRITE_DATASET_SOURCE`, writes each rewritten conditioning prompt as
`prompt`, and stores its row-aligned original text as `reward_prompt` metadata.
SD3.5 therefore generates from the rewrite while PickScore, HPSv2.1, and
CLIPScore all evaluate the original prompt. The unchanged 2,048-row baseline
test split is copied into the treatment so evaluation remains comparable.

The preparer treats the three reward-model snapshots, pinned HPSv2.1
checkpoint, HPS tokenizer vocabulary, and original prompt split as cold-node
inputs. Readiness markers make them one-time SSD prewarm work. The training
wrapper keeps Hugging Face offline, so a run fails before launch instead of
silently downloading multi-gigabyte reward assets during distributed startup.

The framework uses one microbatch size for both rollout and optimization. The
prior launcher sampled 16 images/GPU in one batch and optimized 8/GPU with two
accumulations. The translated config rolls out 8/GPU in two batches and uses
the same two-batch accumulation, preserving 128 rollouts and one update while
staying within Flow-Factory's reusable batch contract.

For a custom shell or Python entrypoint, put it in the repository, rerun
preparation so the staged checkout contains it, and execute it through the same
wrapper:

```bash
./scripts/run_in_flowfactory_runtime.sh -- bash scripts/my_experiment.sh
./scripts/run_in_flowfactory_runtime.sh -- python tools/my_experiment.py
```

The wrapper standardizes `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`, Hugging Face,
Torch, Triton, W&B, temporary files, and the working directory. A different
experiment that needs additional packages must add them to the pinned runtime
requirements and rerun preparation; the wrapper does not silently install
undeclared dependencies.

### 4. Launch the experiment

For the rewritten treatment:

```bash
./scripts/train_nft_geneval_baseline.sh
```

For the stock comparison:

```bash
CONFIG=examples/nft/lora/sd3_5/geneval_stock.yaml \
  ./scripts/train_nft_geneval_baseline.sh
```

For the no-rewrite control:

```bash
./scripts/train_nft_geneval_no_rewrite.sh
```

The launcher refuses to start without `.ready`, the staged repository, the
local Python executable, the selected YAML, and the local Hugging Face cache.
It also refuses to contend with another Flow-Factory trainer on the node.

The current GenEval YAML launches eight BF16 processes with Accelerate and
DeepSpeed ZeRO-2. It writes the experiment output, logs, framework caches, and
W&B data under the local runtime root.

### 5. Build or reuse Arrow embeddings

The configured preprocessing mode is `local`. Each distributed rank encodes its
partition into Arrow shards under a temporary build directory. The local
orchestrator consolidates the shards, validates the result, and renames the
completed build into the fingerprinted cache directory.

This cache must remain on node-local storage for the current implementation.
Putting it on a shared filesystem can let independent nodes or jobs write the
same build directory and race during consolidation. The implementation treats
that pattern as unsafe.

The fingerprint lets a warm node reuse a compatible completed cache. A changed
dataset, model, or preprocessing configuration creates a different cache. Old
fingerprints are disposable after confirming that no active or planned run
needs them.

Copying the 37.2 GB model snapshot once also means the eight workers load from
the node's SSD. Pointing `HF_HOME` at Sensei FS would make startup depend on a
shared mount and could trigger cold-file recall for data not accessed in seven
days.

The real maintained launcher was measured to its first `Epoch 0 Sampling`
event on the same A100 node and with the same completed 134 GB Arrow cache:

| Measurement | Previous copied venv | Self-contained local Conda | Change |
|---|---:|---:|---:|
| Empty-root preparation | 479.858 s | 389.979 s | 89.879 s faster (18.7%) |
| Cached real baseline launch | 800 s | 453 s | 347 s faster (43.4%) |

In the new run, launcher/model/reward initialization reached the initial
evaluation in about 81 seconds. The remaining roughly 372 seconds generated
and scored the baseline's 2,212-image initial evaluation; that is real
experiment work rather than filesystem startup. An earlier direct-Sensei
attempt never became active and failed after roughly 33 minutes while leaving
an 89 GB partial preprocessing cache.

### 6. Train and evaluate

Each epoch samples prompts, generates SD3.5 images, scores them with the GenEval
reward, computes the DiffusionNFT objective, updates the LoRA parameters, and
updates the EMA policy. Evaluation and checkpointing are configured every 20
epochs. The GenEval reward uses Mask2Former and OpenCLIP; detector weights may
be downloaded into the local Torch cache on first use unless they were staged
beforehand.

The current configuration leaves `max_epochs` unset, so the operator decides
when to stop the job. Monitor local free space as Arrow caches, checkpoints,
W&B media, and logs accumulate.

### 7. Complete, publish, and retain checkpoints

Checkpoints are written below the run's local `checkpoints` directory. With
`save_model_only: false`, a checkpoint includes the model, optimizer, scheduler,
random-number state, and trainer progress needed for full recovery.

Publication follows this boundary:

```text
all ranks finish local save
        │
        ▼
local checkpoint-N/_COMPLETE
        │
        ├── S3 disabled: warn and continue local-only
        │
        └── S3 enabled: upload data with s5cmd --no-clobber
                            │
                            ▼
                     upload _COMPLETE last
        │
        ▼
prune older completed local checkpoints
```

Consumers must ignore a local or S3 checkpoint until `_COMPLETE` exists. S3
checkpoint prefixes are immutable, and the trainer does not delete old S3
checkpoints. Apply a reviewed bucket lifecycle or an explicit retention process
to the S3 destination.

The repository currently implements publication but not a complete S3 restore
workflow. `resume_path` accepts a local checkpoint path or a Hugging Face
identifier. Recovery from S3 therefore requires manually copying a completed
S3 checkpoint to local SSD, verifying `_COMPLETE`, and pointing `resume_path`
at that local path. The final upload, download, and resumed-training integration
test remains required before this is a certified disaster-recovery path.

### 8. End the job

Before terminating the node:

1. Verify that every checkpoint to keep has `_COMPLETE` at the S3 destination.
2. Sync W&B offline records or export any local logs and media that matter.
3. Record the Git commit or image digest, YAML, dataset identity, and S3 prefix.
4. Delete disposable local caches only when the job no longer needs them.
5. Terminate the node only after durable publication is confirmed.

Nothing under `/mnt/localssd` should be treated as durable after termination.

## Warm-node and cold-node behavior

| Situation | Expected action |
|---|---|
| New node | Run the preparation script, build the needed Arrow fingerprint, then launch |
| Same node, unchanged inputs | Preparation reuses valid component caches; training reuses the completed Arrow fingerprint |
| Source code changed | Run preparation again to replace the staged code copy |
| Dependency files changed | Preparation rebuilds the local environment |
| Dataset changed | Preparation restages hashed source data; preprocessing builds a new fingerprint |
| Model snapshot changed in place | Remove its readiness file or use a versioned source; current sentinels do not hash model content |
| Node lost | Recreate the runtime from immutable sources, download a completed S3 checkpoint, and resume locally |

## Cleanup rules

Safe cleanup targets on an idle node include:

- obsolete fingerprinted directories under
  `cache/flow_factory/datasets`;
- W&B media already synced and no longer needed for diagnosis;
- old logs and temporary files;
- abandoned `*.tmp` preprocessing builds;
- old completed local checkpoints after their S3 copies are verified;
- the entire runtime root when the node is being retired and all durable outputs
  are confirmed.

Do not delete an active Arrow build, a checkpoint without a verified durable
copy, the only copy of an offline W&B run, or the runtime while training is
active. The preparation script checks for an active trainer before replacing
the staged runtime.

## Production migration

The current fallback is suitable for the upcoming validation work, but these
changes define the production path:

1. Build on a certified Pluto base image. Pin the SD3.5 training stack and the
   tested MMCV CUDA operations in a versioned image.
2. Supply code through that image or Pluto GitHub Code Distribution at a pinned
   commit. Stop treating a personal Sensei checkout as the production source.
3. Move canonical SD3.5, OpenCLIP, Mask2Former, and dataset artifacts to approved
   S3 or another governed artifact store. Use a Sensei tenant directory only
   where shared filesystem semantics are useful.
4. Extend preparation to download versioned source artifacts to local SSD. The
   current script accepts local source directories and uses `cp`; its S3 source
   comment describes a future design.
5. Keep active models, datasets, Arrow caches, logs, W&B files, temporary files,
   and checkpoint writes on local SSD.
6. Give each run an authorized S3 checkpoint prefix, complete the restore
   integration test, and automate local download plus marker validation.
7. Supply secrets through Vault and S3 permissions through Pluto resource-access
   roles.

The repository's generic [`Dockerfile`](../docker/docker-cuda/Dockerfile) is not the production
GenEval image. Its CUDA and Torch stack does not match the pinned GenEval
runtime, it does not install the full GenEval and MMCV path, and the current
Docker build context can include generated data. Do not store caches, datasets,
logs, or checkpoints in a container's writable layer.

## Operator checklist

Before launch:

- [ ] The node has enough SSD space for the selected model, Arrow variant,
      environment, checkpoints, logs, and safety margin.
- [ ] `.env` contains paths and non-secret settings only.
- [ ] `.ready` exists after a successful preparation run.
- [ ] MMCV CUDA validation passed on the current GPU architecture.
- [ ] The selected YAML points its dataset, Arrow cache, and save directory to
      local SSD.
- [ ] For PPD arms: `scripts/prepare_ppd_records.sh` has run after the latest
      preparer invocation, and the matched `rho: 0.0` control is scheduled
      alongside the auxiliary arm.
- [ ] `CHECKPOINT_S3_URI` is a unique, authorized run prefix for any run that
      must survive node loss.
- [ ] A completed S3 checkpoint is not assumed resumable until the restore path
      has been exercised.

Before termination:

- [ ] Required checkpoints have S3 `_COMPLETE` markers.
- [ ] Required W&B data and diagnostic logs are durable.
- [ ] The run records its code version, config, dataset identity, and checkpoint
      prefix.
- [ ] No local-only artifact is still needed.

## Implementation references

- [Cold-node preparation](../scripts/prepare_flowfactory_runtime.sh)
- [Training launcher](../scripts/train_nft_geneval_baseline.sh)
- [Rewritten GenEval config](../examples/nft/lora/sd3_5/geneval.yaml)
- [No-rewrite config](../examples/nft/lora/sd3_5/geneval_no_rewrite.yaml)
- [Stock GenEval config](../examples/nft/lora/sd3_5/geneval_stock.yaml)
- [Checkpoint publication](../src/flow_factory/utils/checkpoint_publish.py)
- [Platform filesystem policy](/sensei-fs-3/users/yunfeix/ai-platform-docs/file-system.md)
- [Platform Docker policy](/sensei-fs-3/users/yunfeix/ai-platform-docs/docker.md)

For algorithm behavior and the framework-wide stages, see
[`algorithms.md`](algorithms.md) and [`workflow.md`](workflow.md). This document
owns the storage placement, node lifecycle, and recovery contract.
