"""DiT Layerwise Semantic Router — global VLM layer weighting for DiT action head.

Core idea:
  Instead of 1:1 binding (DiT block i ← VLM layer i), learn a global softmax weight
  over all 16 VLM layers, conditioned on (task, timestep, state). All DiT blocks
  share the same fused VLM features, allowing the model to dynamically emphasise
  shallow spatial features or deep semantic features per sample.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SemanticRouterConfig:
    """Configuration for DiTLayerwiseSemanticRouter."""

    use_semantic_router: bool = False
    router_type: str = "soft"  # "soft" | "topk"
    router_topk: int = 4       # only used when router_type == "topk"

    # Which signals feed into the router MLP
    use_task_in_router: bool = True
    use_timestep_in_router: bool = True
    use_state_in_router: bool = True

    # Identity initialisation: start close to uniform weighting
    router_identity_init: bool = True
    router_init_scale: float = 0.01  # std for random init (only if identity_init=False)


def build_router_config(action_config) -> SemanticRouterConfig:
    """Extract router config from framework action_model config section."""
    raw = getattr(action_config, "semantic_router", None)
    if raw is None:
        return SemanticRouterConfig()
    return SemanticRouterConfig(
        use_semantic_router=getattr(raw, "use_semantic_router", False),
        router_type=getattr(raw, "router_type", "soft"),
        router_topk=getattr(raw, "router_topk", 4),
        use_task_in_router=getattr(raw, "use_task_in_router", True),
        use_timestep_in_router=getattr(raw, "use_timestep_in_router", True),
        use_state_in_router=getattr(raw, "use_state_in_router", True),
        router_identity_init=getattr(raw, "router_identity_init", True),
        router_init_scale=getattr(raw, "router_init_scale", 0.01),
    )


class DiTLayerwiseSemanticRouter(nn.Module):
    """Lightweight router that outputs a single set of VLM-layer weights per sample.

    Inputs (concatenated into one vector):
      - task_pooled:   [B, D]  mean-pooled VLM hidden states (semantic summary)
      - timestep_emb:  [B, D]  DiT timestep embedding
      - state_emb:     [B, D]  projected robot proprioceptive state

    Output:
      - layer_weights: [B, num_vlm_layers]  softmax (or top-k + softmax)

    Parameter count: ~500 K (when D=960, mlp_hidden=256, L=16).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_vlm_layers: int,
        config: SemanticRouterConfig,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vlm_layers = num_vlm_layers
        self.cfg = config

        # --- determine input dimension ---
        input_parts = []
        if config.use_task_in_router:
            input_parts.append(hidden_dim)
        if config.use_timestep_in_router:
            input_parts.append(hidden_dim)
        if config.use_state_in_router:
            input_parts.append(hidden_dim)
        if not input_parts:
            raise ValueError("Router must have at least one input source enabled.")
        total_input_dim = sum(input_parts)

        # --- MLP body ---
        mlp_hidden = 256
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, num_vlm_layers),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise so that router starts close to uniform (all layers equal)."""
        if self.cfg.router_identity_init:
            # Last linear layer: small weights, bias = 0 → softmax ≈ uniform
            last_linear = self.mlp[-1]
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last_linear.bias)
            # First linear layer: normal init with small scale
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.mlp[0].bias)
        else:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=self.cfg.router_init_scale)
                    nn.init.zeros_(m.bias)

    def _build_input(self, task_pooled, timestep_emb, state_emb):
        """Concatenate enabled input sources into a single feature vector."""
        parts = []
        if self.cfg.use_task_in_router:
            assert task_pooled is not None, "task_pooled required but use_task_in_router=True"
            parts.append(task_pooled)
        if self.cfg.use_timestep_in_router:
            assert timestep_emb is not None, "timestep_emb required"
            parts.append(timestep_emb)
        if self.cfg.use_state_in_router:
            if state_emb is None: state_emb = torch.zeros(task_pooled.shape[0], self.hidden_dim, device=task_pooled.device, dtype=task_pooled.dtype)
            parts.append(state_emb)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        *,
        task_pooled: torch.Tensor,
        timestep_emb: torch.Tensor,
        state_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
          layer_weights: [B, num_vlm_layers], softmax-normalised.
        """
        x = self._build_input(task_pooled, timestep_emb, state_emb)  # [B, total_dim]
        logits = self.mlp(x)                                          # [B, L]

        if self.cfg.router_type == "topk":
            # Keep top-k logits, mask rest to -inf
            topk_logits, topk_idx = logits.topk(self.cfg.router_topk, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_logits)
            logits = mask

        weights = F.softmax(logits, dim=-1)  # [B, L]
        return weights

    def route_vlm_features(
        self,
        vl_embs_list,
        *,
        task_pooled,
        timestep_emb,
        state_emb,
    ):
        """Weighted fusion of VLM layer features according to router output.

        Args:
          vl_embs_list:  list[Tensor], len=L, each [B, seq, D]
          task_pooled:   [B, D]
          timestep_emb:  [B, D]
          state_emb:     [B, D]

        Returns:
          fused:      [B, seq, D]  weighted sum over VLM layers
          weights:    [B, L]       the softmax weights used
        """
        L = len(vl_embs_list)
        B, S, D = vl_embs_list[0].shape

        weights = self.forward(
            task_pooled=task_pooled,
            timestep_emb=timestep_emb,
            state_emb=state_emb,
        )  # [B, L]

        # Stack: [B, L, S, D] → einsum
        vl_stack = torch.stack(vl_embs_list, dim=1)  # [B, L, S, D]
        fused = torch.einsum("blsd,bl->bsd", vl_stack, weights)  # [B, S, D]
        return fused, weights
