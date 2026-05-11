# Recipe: Partial Rollout (PRv3)

English | [简体中文](README_zh.md)

A partial-rollout pipeline for **synchronous RL training**, designed to reclaim GPU bubbles caused by **long-tail response lengths** via sample supplementation and mid-generation interruption with cross-step resume.

> ⚠️ **Don't confuse this with the fully-async framework's partial rollout.** This pipeline still runs the synchronous loop "rollout → wait for batch → one train step"; the only twist is that long-tail samples can be interrupted mid-rollout and resumed in a later step. Trainer and rollout phases remain serial. If you want the trainer/rollout fully decoupled and advancing concurrently, see `verl/experimental/fully_async_policy/` — not here.

> 📎 **Origin.** This implementation is based primarily on Tencent's APR proposal in [verl-recipe#58](https://github.com/verl-project/verl-recipe/pull/58) (Async Partial Rollout, SSIM + Rollout Caching, decoupled IS). PRv3 is a continuation of that work — same three-queue scheduling and partial-rollout semantics, with several follow-up changes detailed in [Continuation of verl-recipe#58](#continuation-of-verl-recipe58) below.

Background and method comparison (APRIL) live in [REFERENCE.md](REFERENCE.md). This README only covers **when to use it, the architecture, and how to run it**.

---

## When to use

Use it when:

- The dataset has a **long-tailed response length distribution** (a small fraction of very long samples drags down each step).
- Synchronous PPO/GRPO shows a visible GPU bubble waiting on those long-tail samples.
- Training tolerates **mild off-policy drift** — partial rollout inherently spans multiple weight versions; pair it with IS correction.
- Multi-turn / tool-call workloads — upstream vLLM ≥ 0.12's `pause_generation` + abort is enough; no recipe-side server fork.

Skip it when:

- Response lengths are uniform with no long-tail bubble — the plain synchronous trainer is simpler.
- Strict on-policy is required (every trajectory must come from the current weights).
- You need to mix this pipeline with the upstream sync trainer's batch-shape assumptions — PRv3-specific fields (dummy `gen_batch`, continuous-worker semantics) would break them.

---

## Architecture

```
                 trainer (PRv3RayPPOTrainer)
                        │
            push_batch  │  pull_batch
                        ▼
         ┌──────────────────────────────────┐
         │   RolloutPromptManager (Ray)     │
         │                                  │
         │   pending ─pull─► ongoing ─push─► done
         └──────────────────────────────────┘
                        │
            pull_prompts│push_prompts
                        ▼
              PRv3AgentLoopWorker  ×N   (run_continuous loop)
                        │
              llm_client│ generate (FullyLLMServerClient retries aborted)
                        ▼
                upstream vLLMReplica  ×replicas
                        ▲
              cancel/   │
              resume    │
                        │
              PRv3LLMServerManager  (pause_generation / resume_generation)
```

| Component | File | Role |
|---|---|---|
| `PRv3RayPPOTrainer` | `ray_trainer.py` | Main trainer loop. `_fit_generate` pushes prompts into the manager, awaits one full batch via `async_rollout_manager.generate_sequences`, runs log_prob / advantage / policy update, then `update_weights`. `fit()` calls `async_rollout_manager.shutdown()` before each return so workers join cleanly. |
| `RolloutPromptManager` | `prompt_manager.py` | Single-threaded Ray actor holding the three queues (pending / ongoing / done). `pull_batch` and `pull_prompts` both block on `asyncio.Event`s — no busy polling, no per-empty-pull RPCs. `stop()` flips `_stopped`; once set, `pull_prompts` returns `[]` which workers read as their shutdown signal. |
| `PRv3AgentLoopManager` / `PRv3AgentLoopWorker` | `agent_loop/agent_loop.py` | Worker runs a persistent loop (`run_continuous`) that keeps one pull RPC in flight alongside the rollout tasks, capped at `max_inflight_prompts`. Manager exposes `cancel()` / `resume()` (delegated to `PRv3LLMServerManager`) and `shutdown()` for trainer-side bracketing of `update_weights` and graceful exit. Per-prompt rollout dispatch is delegated to upstream `AgentLoopWorker.generate_sequences`. |
| `PRv3LLMServerManager` | `llm_server.py` | Thin override of upstream `LLMServerManager`: forces `get_client(fully_async=True)` so callers receive the retry-on-abort `FullyLLMServerClient`, and exposes `cancel` / `resume` that fan out to each `vLLMReplica.abort_all_requests` / `resume_generation`. Installed via a monkey-patch in `PRv3RayPPOTrainer.init_workers` because upstream has no FQN config knob for `LLMServerManager`. |

---

## Key invariants

1. **Prompt ownership during scheduling**: at any instant a prompt lives in exactly one of pending / ongoing / done. `pull_prompts` moves pending→ongoing; `push_prompts` is ongoing→done — terminal only. There is no aborted-back-to-pending re-queue path, because `FullyLLMServerClient.generate()` absorbs the abort/retry cycle inside one logical generate call.
2. **Cross-step abort/resume bracket**: `PRv3AgentLoopManager.generate_sequences` runs `await self.resume()` at entry and `await self.cancel()` after `pull_batch` returns. The naive `checkpoint_engine` backend (PRv3's default) short-circuits before its own abort, so the recipe wires this itself; workers' aborted `client.generate(...)` calls wait inside `FullyLLMServerClient`'s retry loop until the next step's `resume()`.
3. **Per-sample weight-version tracking** lives in `gen_batch.meta_info["global_steps"]` (set by the trainer) and `FullyLLMServerClient.generate()` (records the actual versions each retry submitted against). The worker does no per-call version tracking.
4. **Continuous worker loop**: each `PRv3AgentLoopWorker` runs `run_continuous` for the actor's lifetime, keeping one pull RPC in flight as an `asyncio.Task` alongside up to `max_inflight_prompts` rollouts. Pulls and completions share a single `asyncio.wait`, so completed rollouts get pushed back immediately without waiting on the pull RPC.
5. **The dummy `gen_batch` must carry `uid`**: when the dataloader is exhausted at the end of an epoch, the placeholder batch built to drain in-flight prompts still needs `non_tensor_batch["uid"]`, otherwise the manager can't compute the row count.
6. **Stateful dataloader resume**: `PRv3RayPPOTrainer.fit()` resumes via the stateful dataloader automatically — **don't** add manual skip-on-resume logic.
7. **Graceful shutdown**: `RolloutPromptManager.stop()` sets `_stop_event`. Each worker's `run_continuous` holds a `stop_task` awaiting `wait_until_stop.remote()` on a dedicated RPC channel; when it fires, the worker cancels every in-flight rollout and returns. The cancel is necessary because the last step's `cancel()` left vLLM paused — rollouts blocked inside `FullyLLMServerClient`'s retry-on-abort loop need `CancelledError` to exit (no next-step `resume()` is coming at training end).

---

## Quick start

### Partial-rollout runs

Single-turn:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo.sh
```

Multi-turn with tool calls — first generate the tool-agent dataset:
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
then launch:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### Baseline runs (vanilla GRPO, for A/B against partial rollout)

Same model / data / batch / hyperparameters; the only differences from the PR variants are the upstream entry, the upstream agent loop, and no IS correction:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_baseline.sh
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh
```

### PRv3-specific Hydra overrides

| Key | Value | Notes |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` / `tool_agent` | Use upstream agent loops directly — `FullyLLMServerClient` handles abort/retry, no PRv3 wrapper needed. |
| `+async_training.partial_rollout` | `True` | Gates the retry-on-abort path inside `FullyLLMServerClient`. Required for any PRv3 run; set via Hydra `+` (the key isn't pre-declared in the trainer config). |
| `algorithm.rollout_correction.rollout_is` | `token` (recommended) / `sequence` | Sequence-level ratios easily hit the `exp(±20)` safety clamp once rollouts span several weight versions. |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS upper bound; for IcePop pass `"0.5_5.0"`. |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True` *(tool variant only)* | Enables multi-turn. |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML path | Tool registry; shared across backends. |

---

## Continuation of verl-recipe#58

PRv3 keeps the SSIM (Sample Supplementation + Interruption) skeleton and the three-queue scheduling from the original APR proposal. The follow-up changes:

| Area | verl-recipe#58 (APR) | PRv3 (this directory) | Why |
|---|---|---|---|
| **vLLM cancel** | per-request `asyncio.Event` dict + `Lock`; `cancel()` walks the dict and sets each event, every in-flight `generate` then awaits its own cancel handle | Engine-level batch abort: `paused` flag + `inflight` counter + vLLM `abort_all_requests` (calls `pause_generation` on vLLM ≥ 0.12) in a yielding drain loop | Single engine-core call instead of fanning out per-request cancel handles through one Ray async actor; avoids actor-loop contention as the in-flight count grows. |
| **Inflight budget unit** | prompt count (`pull_pending_prompts(num_rollout_prompts)`) | trajectory count (`pull_prompts(traj_count)` + `get_unfinished_traj_count`) | A partially-rolled-out prompt with k aborted traj costs only k traj of budget, not n. Lets workers carry more partial prompts when the batch has many. |
| **Worker refill accounting** | refill by `len(done) × n` | refill by `sum(consumed_traj for done & fully-done)` recorded in `dict[Task, int]` | Matches the new traj-count budget; partial prompts that finish freed only their consumed traj, not n. |
| **Data flow** | manager owns `StatefulDataLoader`; `pull_pending_prompts` lazily pulls from it when pending is empty | trainer owns the dataloader; pushes batches into the manager via `push_batch` | Keeps the manager stateless w.r.t. epoch/iter cursor. Resume goes through the trainer's normal stateful-dataloader path; the manager doesn't replicate dataset/sampler logic. |
| **Staleness handling** | hard drop: `done_queue.append` only if `max(param_version_diff) < 10` | no drop; let off-policy correction handle it (`algorithm.rollout_correction.rollout_is=token`) | Avoids losing rollouts wholesale when the gate threshold hits; relies on token-level IS to bound divergence. |
| **Scheduling priority** | `get_scheduling_priority` sorts by `(unfinished_samples_num, finished_mean_response_length, max_staleness)` | FIFO with one twist — aborted prompts go to the head of `pending_queue` (`appendleft`) so the next pull resumes them while their KV cache may still be live on the rollout server | Cache-locality optimization for resume; we don't need a full scheduler so far. |

If you're tracking #58, the cancel-path rewrite and the traj-budget refactor are the two changes we'd flag as load-bearing — the rest are scope/data-flow choices that can be lifted back to the recipe layer.

---

## File layout

```
partial_rollout/
├── README.md / README_zh.md / REFERENCE.md
├── main_ppo.py                # @hydra entry; wraps PRv3TaskRunner
├── ray_trainer.py             # PRv3RayPPOTrainer
├── prompt_manager.py          # RolloutPromptManager (Ray actor)
├── llm_server.py              # PRv3LLMServerManager (force fully_async + cancel/resume)
├── agent_loop/
│   └── agent_loop.py          # PRv3AgentLoopManager / PRv3AgentLoopWorker
└── run/
    ├── gsm8k_tool_config.yaml                      # tool registry for the tool variants
    ├── run_qwen3-0.6b_gsm8k_grpo.sh                # PR, single-turn
    ├── run_qwen3-0.6b_gsm8k_grpo_tool.sh           # PR, tool-call
    ├── run_qwen3-0.6b_gsm8k_grpo_baseline.sh       # baseline, single-turn
    ├── run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh  # baseline, tool-call
    └── run_{dapomath,gsm8k}_{nopr,pr}_grpo_4b_*.sh # 4B Qwen3 ports of the recipe PR
```
