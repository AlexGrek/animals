"""Helpers for loading opponent/counterpart models during training.

The three training environments each drive a *frozen* counterpart policy (the
snake env runs the prey/amphibia models; the prey/amphibia envs run the snake
model). After an observation-size change every saved checkpoint is
shape-incompatible, and on a cold start the counterpart may not exist at all.
Rather than hard-fail (which makes the pipeline un-bootstrappable), we fall back
to a static "do nothing" action so a first-generation agent can still be trained.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger("learner.model_utils")


def resolve_model_path(path: str) -> Optional[str]:
    """Return an existing path for `path` (trying a few common roots), or None."""
    candidates = [
        path,
        path if path.endswith(".zip") else path + ".zip",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_opponent(path: str, expected_obs_size: int) -> Optional[PPO]:
    """Load a frozen PPO opponent, or None (logging why) if it can't be used.

    Returns None when the file is missing or its observation space doesn't match
    `expected_obs_size` — callers treat None as "act statically" (see
    `predict_actions`), which lets a first-generation agent bootstrap before any
    counterpart exists.
    """
    resolved = resolve_model_path(path)
    if resolved is None:
        logger.warning(
            "Opponent model '%s' not found; falling back to a static action.", path
        )
        return None
    try:
        model = PPO.load(resolved)
    except Exception as e:  # noqa: BLE001 - want to degrade gracefully on any load error
        logger.warning("Failed to load opponent '%s' (%s); using a static action.", path, e)
        return None

    shape = model.observation_space.shape
    if shape != (expected_obs_size,):
        logger.warning(
            "Opponent '%s' obs shape %s != expected (%d,); using a static action.",
            path,
            shape,
            expected_obs_size,
        )
        return None
    return model


def predict_actions(
    model: Optional[PPO],
    obs: np.ndarray,
    num_actions: int,
    deterministic: bool = False,
) -> List[int]:
    """Batched action prediction with a "do nothing" fallback for a missing model.

    `obs` is `(N, obs_size)`; returns a list of `N` ints. When `model` is None,
    returns action `0` for every row — "Stand" for prey/amphibia (matching the
    static-prey convention already used in `play.py`) and "Straight" for
    snakes — so a first-generation counterpart can bootstrap against a
    stationary/non-adversarial opponent instead of undirected noise.
    """
    n = obs.shape[0]
    if n == 0:
        return []
    if model is None:
        return [0] * n
    actions, _ = model.predict(obs, deterministic=deterministic)
    return [int(a) for a in np.asarray(actions).reshape(-1)]
