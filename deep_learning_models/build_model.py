import logging
from typing import Dict, Any
import torch.nn as nn

from .attention_model import AttentionPredictor

logger = logging.getLogger(__name__)


def build_attention_model(dl_cfg: Dict[str, Any]) -> nn.Module:
    """
    Build an attention-based model from a dict config.
    """
    input_dim = dl_cfg.get("input_dim", 62)
    d_model = dl_cfg.get("d_model", 64)
    num_heads = dl_cfg.get("num_heads", 4)
    num_layers = dl_cfg.get("num_layers", 2)
    ff_dim = dl_cfg.get("ff_dim", 128)
    dropout = dl_cfg.get("dropout", 0.1)
    n_tasks = dl_cfg.get("n_tasks", 1)

    logger.info(
        "[build_attention_model] Creating AttentionPredictor with "
        f"input_dim={input_dim}, d_model={d_model}, heads={num_heads}, layers={num_layers}, "
        f"ff_dim={ff_dim}, dropout={dropout}, n_tasks={n_tasks}"
    )

    model = AttentionPredictor(
        input_dim=input_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        n_tasks=n_tasks
    )
    return model
