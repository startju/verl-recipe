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

from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


class PRv3LLMServerManager(LLMServerManager):
    """LLMServerManager override that forces every `get_client()` caller to
    receive the retry-on-abort client (upstream `FullyLLMServerClient`).

    PRv3's abort/resume cycle around `update_weights` is owned by
    `verl/checkpoint_engine/base.py` (abort_all_requests + resume_generation
    bracket the weight transfer), and the retry-on-abort is gated on
    `config.async_training.partial_rollout=True` — set in the run scripts
    via `+async_training.partial_rollout=True`. There is no PRv3-side
    cancel/resume plumbing.

    This subclass exists only because `RayPPOTrainer.init_workers` calls
    `self.llm_server_manager.get_client()` with no `fully_async` arg
    (defaults False), and there's no upstream config knob to switch the
    default. The trainer monkey-patches `LLMServerManager` to this subclass
    for the duration of `init_workers` (ray_trainer.py).
    """

    def get_client(self, fully_async: bool = False) -> LLMServerClient:
        return super().get_client(fully_async=True)
