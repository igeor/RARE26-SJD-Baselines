import math


def cosine_ema_momentum(step: int, total_steps: int, base_momentum: float = 0.996, final_momentum: float = 1.0) -> float:
    if total_steps <= 1:
        return final_momentum
    progress = step / (total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_momentum - (final_momentum - base_momentum) * cosine


def linear_warmup_value(step: int, warmup_steps: int, start: float, end: float) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return end
    alpha = step / max(1, warmup_steps - 1)
    return start + alpha * (end - start)

