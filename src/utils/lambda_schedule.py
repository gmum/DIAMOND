import math


def build_lambda_schedule(cfg):
    schedule_type = cfg.type

    if schedule_type == "constant":
        value = float(getattr(cfg, "value", 0.0))
        return lambda step_idx, num_steps: value

    if schedule_type == "linear":
        start = float(getattr(cfg, "start", 12.0))
        end = float(getattr(cfg, "end", 1.0))
        return lambda step_idx, num_steps: start * (1 - step_idx / (num_steps - 1)) + end * (
            step_idx / (num_steps - 1)
        )

    if schedule_type == "power":
        start = float(getattr(cfg, "start", 12.0))
        end = float(getattr(cfg, "end", 1.0))
        power = float(getattr(cfg, "power", 2.0))
        return lambda step_idx, num_steps: end + (start - end) * ((1 - step_idx / (num_steps - 1)) ** power)

    if schedule_type == "cosine":
        start = float(getattr(cfg, "start", 12.0))
        end = float(getattr(cfg, "end", 1.0))
        return lambda step_idx, num_steps: end + (start - end) * 0.5 * (
            1 + math.cos(math.pi * step_idx / (num_steps - 1))
        )

    if schedule_type == "hard_then_exp":
        start = float(getattr(cfg, "start", 30.0))
        after = float(getattr(cfg, "after", 2.0))
        power = float(getattr(cfg, "power", 2.0))

        def schedule(step_idx, num_steps):
            if step_idx == 0:
                return start
            t = (step_idx - 1) / max(1, num_steps - 2)
            return after * ((1 - t) ** power)

        return schedule

    if schedule_type == "exponential":
        start = float(getattr(cfg, "start", 15.0))
        end = float(getattr(cfg, "end", 1.0))
        power = float(getattr(cfg, "power", 2.0))
        warmup = int(getattr(cfg, "warmup", 0))
        cooldown = int(getattr(cfg, "cooldown", 0))

        def schedule(step_idx, num_steps):
            if step_idx < warmup:
                return 0.0
            if step_idx >= num_steps - cooldown:
                return 0.0
            effective_steps = num_steps - warmup - cooldown
            t = (step_idx - warmup) / (effective_steps - 1)
            return end + (start - end) * ((1 - t) ** power)

        return schedule

    raise ValueError(f"Unknown lambda schedule type: {schedule_type}")
