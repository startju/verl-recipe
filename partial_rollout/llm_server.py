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

from omegaconf import DictConfig
from recipe.partial_rollout.vllm_rollout.vllm_async_server import PRv3vLLMReplica

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


class PRv3LLMServerManager(LLMServerManager):
    """LLMServerManager that:

    1. Swaps the replica class to `PRv3vLLMReplica` so each server exposes
       the Python-side `_resume_event`-gated `cancel`/`resume`. Setting
       `self.rollout_replica_class` before `super().__init__` short-circuits
       upstream's default lookup at `LLMServerManager.__init__`.
    2. Forces `get_client(fully_async=True)` so every caller — including
       `RayPPOTrainer.init_workers`, which calls `get_client()` with no arg
       (defaults False) — receives the retry-on-abort `FullyLLMServerClient`.
       The retry loop is gated on `config.async_training.partial_rollout=True`
       set in the run scripts via `+async_training.partial_rollout=True`.

    Installed via a monkey-patch in `PRv3RayPPOTrainer.init_workers` because
    upstream has no FQN config knob for `LLMServerManager`.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.rollout_replica_class = PRv3vLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)

    def get_client(self, fully_async: bool = False) -> LLMServerClient:
        return super().get_client(fully_async=True)

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])
