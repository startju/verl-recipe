# Recipe: Partial Rollout (PRv3)

[English](README.md) | 简体中文

**同步 RL 训练模式**下的 partial-rollout 流水线，针对**长尾响应长度**带来的 GPU 闲置问题做样本补充与中断续跑。

> ⚠️ **不要跟全异步框架的 partial rollout 混淆**：本流水线仍走「rollout → 等批次齐 → 一次性 train step」的同步循环，只是允许在 rollout 阶段中断长尾样本、跨 step 续跑；trainer 与 rollout 之间是顺序串行的。如果你需要的是 trainer / rollout 完全解耦异步推进的版本，看 `verl/experimental/fully_async_policy/`，不是这里。

> 📎 **来源说明**：本实现主要参考了腾讯在 [verl-recipe#58](https://github.com/verl-project/verl-recipe/pull/58) 中提出的 APR 方案（Async Partial Rollout，含 SSIM 与 Rollout Caching、decoupled IS）。PRv3 在 #58 基础上继续推进——保留三队列调度与 partial-rollout 语义，做了若干修改，详见下文 [接续 verl-recipe#58 的工作](#接续-verl-recipe58-的工作)。

学术背景与方法对照（APRIL）放在 [REFERENCE.md](REFERENCE.md)，本 README 只讲**何时用、整体架构、怎么跑**。

---

## 使用场景

适用：

- 数据集**响应长度分布长尾**（少量超长样本拖慢整批 step）
- 同步 PPO/GRPO 因等待长尾样本导致 GPU bubble 明显
- 训练对**轻微 off-policy** 容忍（partial rollout 必然引入 weight-version 跨越，需配合 IS 修正）
- 多轮 / tool-call 场景 —— 上游 vLLM ≥ 0.12 的 `pause_generation` + abort 已经够用，本 recipe 不 fork server

不适用：

- 响应长度均匀、没有 long-tail bubble — 同步 trainer 更简单
- 严格 on-policy 必须保证（每个 trajectory 只能由当前权重产出）
- 同时使用本流水线 + 上游 sync trainer 的 batch shape 假设（dummy gen_batch、continuous-worker 语义会破坏）

---

## 整体架构

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
              PRv3AgentLoopWorker  ×N   (run_continuous 持续循环)
                        │
              llm_client│ generate (FullyLLMServerClient retry aborted)
                        ▼
                upstream vLLMReplica  ×replicas
                        ▲
              cancel/   │
              resume    │
                        │
              PRv3LLMServerManager  (pause_generation / resume_generation)
```

- **`PRv3RayPPOTrainer`** (`ray_trainer.py`)：trainer 主循环。`_fit_generate` 把 prompt push 进 manager，调 `async_rollout_manager.generate_sequences` 等一个完整 batch 回来，再走 log_prob / advantage / policy update / `update_weights`。`fit()` 在每个 return 前调 `async_rollout_manager.shutdown()` 让 workers 干净退出。
- **`RolloutPromptManager`** (`prompt_manager.py`)：单线程 Ray actor，维护三个队列（pending / ongoing / done）。`pull_batch` 和 `pull_prompts` 都用 `asyncio.Event` 阻塞 —— 无 busy poll，无空 pull 的 RPC。`stop()` 翻 `_stopped`，之后 `pull_prompts` 立刻返回 `[]`，worker 据此识别 shutdown。
- **`PRv3AgentLoopManager` / `PRv3AgentLoopWorker`** (`agent_loop/agent_loop.py`)：worker 跑常驻 `run_continuous` 循环，把一个 pull RPC 当 task 跟 rollout 任务一起塞进同一个 `asyncio.wait`，capacity 上限是 `max_inflight_prompts`。完成的 rollout 不用等 pull 返回就能 push 回去。Manager 暴露 `cancel()` / `resume()`（委托给 `PRv3LLMServerManager`）让 trainer 包住 `update_weights`，以及 `shutdown()` 用于优雅退出。每个 prompt 的具体 rollout 直接调用上游 `AgentLoopWorker.generate_sequences`。
- **`PRv3LLMServerManager`** (`llm_server.py`)：上游 `LLMServerManager` 的薄壳子类。强制 `get_client(fully_async=True)`，让所有 caller 拿到带 retry-on-abort 的 `FullyLLMServerClient`；并暴露 `cancel` / `resume`，分别 fan out 到每个 `vLLMReplica.abort_all_requests` / `resume_generation`。因为上游没有 `LLMServerManager` 的 FQN 配置开关，所以在 `PRv3RayPPOTrainer.init_workers` 里 monkey-patch 替换。

---

## 关键不变量

1. **prompt 流转的所有权**：每个 prompt 同一时刻只在 pending / ongoing / done 之一。`pull_prompts` 移 pending→ongoing；`push_prompts` 只走 ongoing→done。没有 aborted-回 pending 的二次入队，因为 `FullyLLMServerClient.generate()` 在单次 generate 调用内部就把 abort/retry 吸收了。
2. **跨 step 的 abort/resume 包装**：`PRv3AgentLoopManager.generate_sequences` 开头 `await self.resume()`，`pull_batch` 拿到完整 batch 后 `await self.cancel()`。naive `checkpoint_engine` backend（PRv3 默认）会跳过自己的 abort，所以这一步必须由 recipe 自己来；worker 的 `client.generate(...)` 被 abort 后在 `FullyLLMServerClient` retry 循环里等下一个 step 的 `resume()`。
3. **per-sample weight-version 跟踪**：放在 `gen_batch.meta_info["global_steps"]`（trainer 写）和 `FullyLLMServerClient.generate()`（记录每次 retry 真正提交时的版本）里。worker 自己不做版本跟踪。
4. **持续 worker 循环**：每个 `PRv3AgentLoopWorker` 跑 `run_continuous` 直到 actor 销毁；把一个 pull RPC 当 `asyncio.Task` 跟最多 `max_inflight_prompts` 个 rollout 任务一起塞进同一个 `asyncio.wait`。完成的 rollout 不用等 pull 返回就能 push 回去。
5. **dummy gen_batch 也要带 uid**：epoch 末 dataloader 耗尽时构造的占位 batch 必须填 `non_tensor_batch["uid"]`，否则 manager 端取不到行数。
6. **stateful dataloader 续训**：`PRv3RayPPOTrainer.fit()` 走 stateful loader 自动恢复进度，**不要**手工加 skip-on-resume 逻辑。
7. **优雅退出**：`RolloutPromptManager.stop()` 翻 `_stopped`，之后 `pull_prompts` 立刻返回 `[]`。worker 的 `run_continuous` 收到 `[]` 就把还在跑的 rollout 全部 `task.cancel()` 然后返回。cancel 是必需的：最后一个 step 的 `cancel()` 把 vLLM 留在 paused 状态，rollout 卡在 `FullyLLMServerClient` retry 循环里等不到下一个 `resume()`，靠 `CancelledError` 才能把它们放出来。

---

## 快速上手

### 单轮 (single-turn)
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo.sh
```

### 多轮工具调用 (tool agent)
先生成 tool-agent 数据集：
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
再启动：
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### 关键 Hydra override

| 项 | 值 | 说明 |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `prv3_single_turn_agent` / `prv3_tool_agent` | 必须用 `prv3_` 开头的 agent loop（worker 里有 assert 防退化） |
| `algorithm.rollout_correction.rollout_is` | `token` 或 `sequence` | 推荐 `token`（partial rollout 跨权重版本，sequence 级 ratio 易被 clamp） |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS 上限；想用 IcePop 写 `"0.5_5.0"` |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True`（仅 tool 版） | 启用多轮 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML 路径 | tool registry，跨 backend 共用 |

---

## 接续 verl-recipe#58 的工作

PRv3 沿用了原 APR 提案的 SSIM（Sample Supplementation + Interruption）骨架和三队列调度，主要改动：

| 维度 | verl-recipe#58 (APR) | PRv3（本目录） | 原因 |
|---|---|---|---|
| **vLLM cancel** | 每个 request 一把 `asyncio.Event` + `Lock`；`cancel()` 遍历 dict set 每个 event，每个 in-flight `generate` 各自 await 自己的 cancel handle | engine-level 批量 abort：`paused` flag + `inflight` 计数 + vLLM `abort_all_requests`（vLLM ≥ 0.12 走 `pause_generation`），配合 yielding drain | 一次 engine-core 调用替代单 Ray async actor 上分发的 per-request cancel handle 风暴；in-flight 数量增长时避免 actor 事件循环抢占。 |
| **inflight 预算单位** | prompt 数（`pull_pending_prompts(num_rollout_prompts)`） | trajectory 数（`pull_prompts(traj_count)` + `get_unfinished_traj_count`） | partial prompt（剩余 k 个 aborted traj）只占 k 个 traj 预算，而不是固定 n。让 worker 在批次里 partial 较多时能多塞 prompt。 |
| **worker refill 算法** | `len(done) × n` 直接 refill | 用 `dict[Task, int]` 记录每个 task 拉时占的 traj 预算，refill 取 `sum(consumed_traj for done & fully-done)` | 配合新的 traj-count 预算：partial prompt 完成时只释放它当初占的 traj，而非整个 n。 |
| **数据流** | manager 内置 `StatefulDataLoader`，`pull_pending_prompts` 队列空时直接从 dataloader 拉 | trainer 持 dataloader，`push_batch` 注入 manager；manager 不直接拉数据 | manager 不持 epoch / iter 游标。续训走 trainer 的 stateful-dataloader 通道，manager 不复制 dataset/sampler 逻辑。 |
| **staleness 处理** | 硬丢：`max(param_version_diff) < 10` 才入 done_queue | 不丢，靠 off-policy 修正（`algorithm.rollout_correction.rollout_is=token`） | 阈值命中时 wholesale 丢 rollout 损失大；改用 token-level IS 控制偏移。 |
| **调度优先级** | `get_scheduling_priority` 按 `(unfinished_samples_num, finished_mean_response_length, max_staleness)` 排序 | FIFO，aborted prompt 入 `pending_queue` 头部（`appendleft`）让下次 pull 立即续跑（KV 缓存可能还活着） | resume 的 cache-locality 优化；目前还不需要完整调度器。 |

跟踪 #58 的话，cancel 路径重写和 traj 预算重构是两条 load-bearing 改动；其它属于 scope / 数据流选型，可以反向 port 回 recipe 层。
