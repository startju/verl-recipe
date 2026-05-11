# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig
from recipe.partial_rollout.prompt_manager import RolloutPrompt

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


@ray.remote
class PRv3AgentLoopWorker(AgentLoopWorker):
    """Continuous-loop worker. Delegates per-prompt rollout to upstream's
    `AgentLoopWorker.generate_sequences` (fires the n trajectories, runs
    `_run_agent_loop`, postprocesses, returns a DataProto). The PRv3
    additions are just the prompt-manager-driven pull/gen/push loop — no
    per-prompt dispatch logic, no per-call global_steps tracking (it flows
    through `gen_batch.meta_info` and `FullyLLMServerClient` records the
    actual weight versions used).
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        prompt_manager_handle: ray.actor.ActorHandle,
        teacher_client: Optional[dict[str, LLMServerClient]] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        self.prompt_manager_handle = prompt_manager_handle

    async def run_continuous(self, max_inflight_prompts: int) -> None:
        """Persistent worker loop fed by the prompt manager's blocking
        `pull_prompts` (await on an asyncio.Event — no sleep/poll). The
        abort/resume cycle around `update_weights` is owned by the upstream
        `checkpoint_engine` (bracketing the weight transfer), and any
        aborted `client.generate(...)` calls are retried with accumulated
        context inside `FullyLLMServerClient.generate()`. Loop ends when the
        actor is destroyed at training end.
        """
        running: set[asyncio.Task] = set()
        pull = self.prompt_manager_handle.pull_prompts.remote
        push = self.prompt_manager_handle.push_prompts.remote

        # Bootstrap: blocks until the trainer's first push_batch.
        for rp in await pull(max_inflight_prompts):
            running.add(asyncio.create_task(self._run_one(rp)))

        while True:
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            push_list: list[RolloutPrompt] = [t.result() for t in done]
            running -= done
            # Fire-and-forget; amortizes the RPC round-trip into the next pull.
            push(push_list)
            # Refill by exactly the count we just freed. Blocks if pending is
            # empty; resumes when push_batch sets _prompts_pending.
            for rp in await pull(len(push_list)):
                running.add(asyncio.create_task(self._run_one(rp)))

    async def _run_one(self, rp: RolloutPrompt) -> RolloutPrompt:
        rp.gen_batch_output = await super().generate_sequences(rp.gen_batch_output)
        return rp


class PRv3AgentLoopManager(AgentLoopManager):
    """PRv3 manager. Differences vs upstream:

    - Builds PRv3AgentLoopWorker with an extra `prompt_manager_handle` arg so
      cross-step partial-rollout state lives in a Ray actor instead of being
      threaded through generate_sequences kwargs.
    - Skips automatic worker creation in `create()` because workers need the
      prompt manager (and the trainer's `PRv3LLMServerManager` for cancel/
      resume), which the trainer wires in post-init via
      `init_agent_loop_workers(...)`.
    - Exposes `cancel` / `resume` that target replicas owned by the decoupled
      `LLMServerManager`. The trainer brackets `update_weights` with these so
      PRv3's continuously-running workers don't generate trajectories that
      straddle a weight update. Necessary because the naive checkpoint_engine
      backend (PRv3's default) short-circuits before its own abort/resume.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: Optional[dict[str, LLMServerClient]] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.agent_loop_workers_class = PRv3AgentLoopWorker
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)
        # Set by the trainer via init_agent_loop_workers; until then, calling
        # generate_sequences / cancel / resume is a programming error.
        self.rollout_prompt_manager: Optional[ray.actor.ActorHandle] = None
        self.llm_server_manager = None

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the manager but defer worker spawn.

        Upstream's create eagerly spawns AgentLoopWorkers; PRv3 can't because
        each worker needs `rollout_prompt_manager`, which the trainer only has
        after `RayPPOTrainer.init_workers` returns. The trainer calls
        `init_agent_loop_workers` immediately after, so the deferral window is
        a couple of statements wide.
        """
        return cls(*args, **kwargs)

    @auto_await
    async def init_agent_loop_workers(self, rollout_prompt_manager, llm_server_manager):
        self.rollout_prompt_manager = rollout_prompt_manager
        self.llm_server_manager = llm_server_manager
        await self._init_agent_loop_workers()
        # Spawn each worker's persistent loop fire-and-forget. Workers immediately
        # start polling `prompt_manager.pull_prompts` (which blocks on an
        # asyncio.Event until the trainer's first push_batch). Mid-trajectory
        # aborts triggered by the trainer's cancel() are retried with
        # accumulated context inside `FullyLLMServerClient.generate()`, so
        # workers don't need their own pause gate.
        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        num_workers = len(self.agent_loop_workers)
        max_inflight_prompts = (train_batch_size + num_workers - 1) // num_workers
        self._worker_loop_refs = [
            worker.run_continuous.remote(max_inflight_prompts) for worker in self.agent_loop_workers
        ]

    async def _init_agent_loop_workers(self):
        # Mirrors upstream `AgentLoopManager._init_agent_loop_workers` (verl
        # /experimental/agent_loop/agent_loop.py) but injects
        # `rollout_prompt_manager` between `llm_client` and `teacher_client`
        # in the PRv3AgentLoopWorker constructor.
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.rollout_prompt_manager,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Block until the prompt manager has a full training batch ready.

        Cross-step abort/resume cycle:

        - `resume()` at the start lifts the pause set by the previous step's
          `cancel()`. Workers' retried `client.generate(...)` calls — aborted
          last step and waiting inside `FullyLLMServerClient.generate()`'s
          retry loop — submit fresh against the just-updated weights.
        - `cancel()` after `pull_batch` aborts whatever the continuous
          workers are generating at the moment the trainer takes ownership
          of the batch. Without it, workers would keep producing trajectories
          across the upcoming forward/backward/update_weights with stale
          weights — wasted compute (`sleep_replicas` would block them anyway)
          and, if any token slipped through, off-policy contamination.
          Needed because the naive checkpoint_engine backend (PRv3's default)
          does not itself bracket abort/resume around `update_weights`.

        Per-sample weight-version tracking lives in `gen_batch.meta_info`
        (set by the trainer) and `FullyLLMServerClient.generate()`'s retry
        loop; no per-worker fan-out needed.
        """
        await self.resume()
        if prompts.meta_info.get("validate", False):
            # Validation uses the upstream `generate_sequences` path which
            # spawns its own per-call rollouts via the llm_client; the
            # continuous workers are unaffected and stay blocked in
            # `pull_prompts` until the next training-side `push_batch`.
            # No cancel at end — validation's per-call rollouts have already
            # completed when super() returns; canceling would only race with
            # the continuous workers' in-flight generations, which are still
            # legitimate work for the next training batch.
            return await super().generate_sequences(prompts)

        # pull_batch is async server-side and blocks on an internal
        # asyncio.Event until done_queue >= batch_size. One round-trip, no
        # manager-side polling, no per-step ~10k empty Ray RPCs.
        output = await self.rollout_prompt_manager.pull_batch.remote()
        await self.cancel()
        return output

    @auto_await
    async def cancel(self):
        await self.llm_server_manager.cancel()

    @auto_await
    async def resume(self):
        await self.llm_server_manager.resume()
