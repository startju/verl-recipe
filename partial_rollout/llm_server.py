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
from verl.workers.rollout.llm_server import LLMServerManager


class PRv3LLMServerManager(LLMServerManager):
    """LLMServerManager that swaps the replica class to PRv3vLLMReplica so
    replicas expose cancel()/resume() for cross-step partial rollout.

    Setting `self.rollout_replica_class` before `super().__init__` short-circuits
    the default lookup at LLMServerManager.__init__ (the upstream code only
    initializes it via `get_rollout_replica_class(...)` when the attribute is
    missing).
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.rollout_replica_class = PRv3vLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])
