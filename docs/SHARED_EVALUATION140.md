# Shared WebShop evaluation: every method, both model sizes

All public GRPO, SDAR, RLSD, HINT-SD and RETHINK WebShop releases use the
**same ordered 140-task manifest**, not independently drawn 140-task subsets.
`configs/fixed_validation140.json`, `configs/task_split.json`, and
`configs/evaluation_protocol.json` are identical across the ten releases.

## Unchanged generation and native scoring

Temperature 0.4, sampling enabled, top-p 1.0, unrestricted top-k (-1 in vLLM),
one rollout per task; at most 15 environment actions, history length 2,
4096 prompt tokens and 512 response tokens per action. The native prompt,
action executor and reward function are not replaced. Native terminal score
1.0 is success. Report mean native score separately from success rate;
multiply either by 100 only for percentage-scale presentation.

The 1000-product synthetic-goal environment is retained. This is not the
official human500 evaluation and is not interchangeable with older 128-task
results. Training temperatures, losses, optimizer, training budgets, reward,
hint routing and validation cadence are not changed by this evaluation fix.

## Exact task and state identity

The existing shared set contains 28 objectives from each native category,
140 distinct target products, 123 queries and 126 product-category paths.
Worker i uses seed 1000+i and canonical filtered goal index i, for i=0..139.
These seeds are part of the task definition because native goal order and
price constraints depend on the worker seed. Do not replace them with a single
seed or a fresh per-method task draw.

The split is by objective content, not by per-worker shuffled index. All
matching held-out objective variants are removed from every RL training
worker, including constructor and automatic resets. This removes 145 native
goal entries; 6765 remain for training. Products themselves are not held out.
Runtime checks compare actual task identities and save ordered task IDs,
selection SHA256 and the initial-observation SHA256. A resume that changes
the evaluation identity is rejected, not silently mixed into the old curve.

Task manifest SHA256:
`706adb889929acc45bafd52fc144cd160b85ee7f752adf7b6dad8699f8c5ce4a`

Split manifest SHA256:
`6055ef05de812132425060ad959d11bd6ca1cbe40a94f9a195f9f0647ae9547c`

## Provenance and comparison boundaries

The shared manifests retain their canonical SDAR environment provenance
(`29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`). RETHINK retains its separately
pinned algorithm revision: matching prompt, memory, projector and WebShop
engine hashes are verified rather than upgrading its training algorithm.

This correction does not retrain or alter published checkpoints, rewrite old
scores as 140-task results, or establish that historical SFT/skill datasets
never contained these objectives. Audit all pretraining/SFT/skill provenance
before claiming an end-to-end held-out result. Start a new run directory after
upgrading a legacy 128-task RETHINK run; do not append to its old curve.

RETHINK's 200-H greedy free-generation diagnostic remains separately labeled:
it is not the shared140 benchmark. Formal cross-method evaluation uses the
native vLLM evaluation entry with the settings above and no hint API.

CPU/source checks do not constitute a CUDA/NCCL or full GPU training result.
