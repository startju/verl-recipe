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

from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


class PRv3LLMServerManager(LLMServerManager):
    """LLMServerManager override for PRv3.

    Two PRv3-specific additions on top of upstream:

    1. `get_client(fully_async=True)` is forced so every caller — including
       `RayPPOTrainer.init_workers`, which calls `get_client()` with no arg
       (defaults False) — receives the retry-on-abort `FullyLLMServerClient`.
       The retry loop is gated on `config.async_training.partial_rollout=True`,
       set in the run scripts via `+async_training.partial_rollout=True`.

    2. `cancel` / `resume` route to upstream `vLLMReplica.abort_all_requests`
       (vLLM `pause_generation` + abort) and `resume_generation`. The trainer
       brackets `update_weights` with these so PRv3's continuously-running
       workers don't generate trajectories that straddle a weight update.
       Upstream's `checkpoint_engine` only brackets abort/resume in non-naive
       backends (`base.py: if self.backend == "naive": return` short-circuits
       before the abort); PRv3 uses the naive backend by default, so it must
       wire this itself.

    The subclass is installed via a monkey-patch in `PRv3RayPPOTrainer
    .init_workers` because `RayPPOTrainer.init_workers` hardcodes
    `LLMServerManager.create(...)` and has no FQN config knob.
    """

    def get_client(self, fully_async: bool = False) -> LLMServerClient:
        return super().get_client(fully_async=True)

    async def cancel(self):
        await asyncio.gather(*[replica.abort_all_requests() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume_generation() for replica in self.rollout_replicas])
