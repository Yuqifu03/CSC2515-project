from typing import Dict, Any
import random

import wandb
import numpy as np
import torch


def setup_wandb(cfg: Dict[str, Any]) -> None:
    """Initialize the wandb process safely."""
    
    mode = cfg.get("wandb_mode", "disabled") 
    
    wandb.init(
        project="reward-shaping",
        name=cfg.get("wandb_name", "llm_reward_run"),
        config=cfg,
        mode=mode,
        id=None,
        resume="never", 
    )


def set_seeds(seed: int) -> None:
    """Set experiment seed across libraries.

    Args:
        seed (int): integer seed.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
