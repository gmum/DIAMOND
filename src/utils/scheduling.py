import numpy as np
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from schedulers.flow_match_euler_discrete_x0 import FlowMatchEulerDiscreteSchedulerX0


def make_scheduler_and_timesteps(pipeline, latent, num_steps, device):
    scheduler = FlowMatchEulerDiscreteSchedulerX0.from_config(pipeline.scheduler.config)
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = calculate_shift(
        latent.shape[1],
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, device, None, sigmas, mu=mu)
    return scheduler, timesteps
