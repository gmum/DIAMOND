from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerDiscreteSchedulerOutputWithX0(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor


class FlowMatchEulerDiscreteSchedulerX0(FlowMatchEulerDiscreteScheduler):
    def _predict_x0(
        self,
        sample: torch.FloatTensor,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
    ) -> torch.FloatTensor:
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device=sample.device, dtype=sample.dtype)
        else:
            t = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
        t = t / self.config.num_train_timesteps
        while t.ndim < sample.ndim:
            t = t.view(-1, *([1] * (sample.ndim - 1)))
        return sample - t * model_output

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutputWithX0, Tuple]:
        x0 = self._predict_x0(sample, model_output, timestep)

        output = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            per_token_timesteps=per_token_timesteps,
            return_dict=return_dict,
        )

        if not return_dict:
            return (output[0], x0)

        return FlowMatchEulerDiscreteSchedulerOutputWithX0(
            prev_sample=output.prev_sample,
            pred_original_sample=x0,
        )
