# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].


import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.flow_matching_head.semantic_router import (
    DiTLayerwiseSemanticRouter,
    SemanticRouterConfig,
    build_router_config,
)

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # default for qwen2.5-vl


class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = getattr(action_config, "diffusion_model_cfg", None) or {}
        diffusion_model_cfg = dict(diffusion_model_cfg)

        action_config.add_pos_embed = getattr(action_config, "add_pos_embed", True)
        action_config.max_seq_len = getattr(action_config, "max_seq_len", 1024)
        action_config.num_target_vision_tokens = getattr(action_config, "num_target_vision_tokens", 32)
        action_config.noise_beta_alpha = getattr(action_config, "noise_beta_alpha", 1.5)
        action_config.noise_beta_beta = getattr(action_config, "noise_beta_beta", 1.0)
        action_config.noise_s = getattr(action_config, "noise_s", 0.999)
        action_config.num_timestep_buckets = getattr(action_config, "num_timestep_buckets", 1000)
        action_config.num_inference_timesteps = getattr(action_config, "num_inference_timesteps", 4) or 4
        action_config.state_dim = getattr(action_config, "state_dim", None)

        # 更新 DiTConfig 到 diffusion_model_cfg
        DiTConfig["num_layers"] = global_config.framework.qwenvl.num_vl_layers
        DiTConfig["input_embedding_dim"] = global_config.framework.qwenvl.vl_hidden_dim
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        diffusion_model_cfg.update(DiTConfig)
        # diffusion_model_cfg["interleave_self_attention"] = False
        diffusion_model_cfg["cross_attention_dim"] = DiTConfig[
            "input_embedding_dim"
        ]  # should match vl embedding dim, but for some case we might want to change it for cross + self attention
        self.input_embedding_dim = global_config.framework.qwenvl.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg)  # TODO better way is copy LLM from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        # ===== Semantic Router =====
        router_cfg = build_router_config(action_config)
        self.use_semantic_router = router_cfg.use_semantic_router
        if self.use_semantic_router:
            num_vlm_layers = global_config.framework.qwenvl.num_vl_layers
            self.semantic_router = DiTLayerwiseSemanticRouter(
                hidden_dim=self.input_embedding_dim,
                num_vlm_layers=num_vlm_layers,
                config=router_cfg,
            )
        else:
            self.semantic_router = None

        self.register_buffer("last_router_weights", None)
        self._task_names = None

        # ===== S2 Per-Block Static Routing (Semantic Routing style) =====
        s2_cfg = getattr(action_config, "s2_routing", None)
        self.use_s2_routing = getattr(s2_cfg, "use_s2_routing", False) if s2_cfg is not None else False
        self.use_residual_s2 = getattr(s2_cfg, "residual", False) if s2_cfg is not None else False
        self.entropy_weight = getattr(s2_cfg, "entropy_weight", 5e-4) if s2_cfg is not None else 5e-4
        if self.use_s2_routing:
            num_dit_layers = len(self.model.transformer_blocks)
            num_vlm_layers = global_config.framework.qwenvl.num_vl_layers
            s2_init = getattr(s2_cfg, "identity_init_scale", 1.0) if s2_cfg is not None else 1.0
            self.block_gate = nn.Parameter(torch.zeros(num_dit_layers, num_vlm_layers))
            # Identity init: block i prefers VLM layer i
            for i in range(num_dit_layers):
                self.block_gate.data[i, i] = s2_init
            if self.use_residual_s2:
                alpha_init = getattr(s2_cfg, "alpha_init", 0.01) if s2_cfg is not None else 0.01
                alpha_raw_init = math.log(alpha_init / (1 - alpha_init))
                self.block_alpha = nn.Parameter(torch.full((num_dit_layers,), alpha_raw_init))
                self.block_layernorm = nn.LayerNorm(self.input_embedding_dim)
            else:
                self.block_alpha = None
                self.block_layernorm = None
        else:
            self.block_gate = None
            self.block_alpha = None
            self.block_layernorm = None



    def set_task_metadata(self, task_names: list):
        self._task_names = task_names

    def get_last_router_weights(self):
        return self.last_router_weights

    def _compute_router_features(self, vl_embs_list, state_features, timestep_emb):
        last_vlm = vl_embs_list[-1]
        task_pooled = last_vlm.mean(dim=1)
        state_pooled = state_features.squeeze(1) if state_features is not None else None
        return task_pooled, state_pooled, timestep_emb

    def _s2_route(self, vl_embs_list):
        """Per-block static routing: learnable weight per (DiT block, VLM layer).

        Standard mode: fused_d = sum_j softmax(gate[d,j]) * vl_emb[j]
        Residual mode: fused_d = vl_emb[d] + sigmoid(alpha[d]) * sum_j softmax(gate[d,j]) * LN(vl_emb[j])
        """
        w = F.softmax(self.block_gate, dim=-1)  # [D, L]
        B, S, D = vl_embs_list[0].shape
        fused_list = []
        if self.use_residual_s2 and self.block_alpha is not None:
            for d in range(len(self.model.transformer_blocks)):
                h_d = vl_embs_list[d]  # 1:1 path, always preserved
                vl_norm = torch.stack(
                    [self.block_layernorm(vl_embs_list[j]) for j in range(len(vl_embs_list))], dim=1
                )  # [B, L, S, D]
                w_d = w[d].view(1, -1, 1, 1)  # [1, L, 1, 1]
                residual = (vl_norm * w_d).sum(dim=1)  # [B, S, D]
                alpha_d = torch.sigmoid(self.block_alpha[d])
                fused_d = h_d + alpha_d * residual
                fused_list.append(fused_d)
        else:
            vl_stack = torch.stack(vl_embs_list, dim=1)  # [B, L, S, D]
            for d in range(len(self.model.transformer_blocks)):
                w_d = w[d].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, L, 1, 1]
                fused_d = (vl_stack * w_d).sum(dim=1)  # [B, S, D]
                fused_list.append(fused_d)
        self.last_router_weights = w.detach()
        return fused_list

    def _route_or_passthrough(self, vl_embs_list, temb, state_features):
        # S2 routing: per-block static weights
        if self.use_s2_routing and self.block_gate is not None:
            return self._s2_route(vl_embs_list)

        # Semantic router: global dynamic weights
        if self.use_semantic_router and self.semantic_router is not None:
            task_pooled, state_emb, timestep_emb = self._compute_router_features(
                vl_embs_list, state_features, temb
            )
            fused, weights = self.semantic_router.route_vlm_features(
                vl_embs_list,
                task_pooled=task_pooled,
                timestep_emb=timestep_emb,
                state_emb=state_emb,
            )
            self.last_router_weights = weights.detach()
            num_dit_layers = len(self.model.transformer_blocks)
            return [fused] * num_dit_layers

        # Passthrough: 1:1 binding
        self.last_router_weights = None
        return vl_embs_list

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state).unsqueeze(1) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Encode timesteps
        temb = self.model.timestep_encoder(t_discretized)

        # Router: fuse VLM layers (or keep 1:1 binding when disabled)
        vl_for_dit = self._route_or_passthrough(vl_embs_list, temb, state_features)

        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_for_dit[layer_idx],
                temb=temb,
            )

        # TODO miss self att and _process_output, but work well
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        flow_loss = ((pred_actions - velocity) ** 2).mean()
        if self.use_residual_s2 and self.block_gate is not None:
            w = F.softmax(self.block_gate, dim=-1)
            log_w = F.log_softmax(self.block_gate, dim=-1)
            entropy = -(w * log_w).sum(dim=-1).mean()
            loss = flow_loss + self.entropy_weight * entropy
            self._last_entropy = entropy.detach()
        else:
            loss = flow_loss
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state).unsqueeze(1) if state is not None else None
        all_step_weights = []

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Encode timestep
            temb = self.model.timestep_encoder(timesteps_tensor)

            # Router: fuse VLM layers (or keep 1:1 binding when disabled)
            vl_for_dit = self._route_or_passthrough(vl_embs_list, temb, state_features)

            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_for_dit[layer_idx],
                    temb=temb,
                )
            # TODO miss self att and _process_output
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity

            if self.use_semantic_router and self.last_router_weights is not None:
                all_step_weights.append(self.last_router_weights)
        if all_step_weights:
            self.last_router_weights = torch.stack(all_step_weights, dim=0).mean(dim=0)

        return actions


    def save_router_analysis(self, save_dir: str):
        import os, numpy as np
        os.makedirs(save_dir, exist_ok=True)
        if self.last_router_weights is not None:
            np.save(os.path.join(save_dir, "router_weights.npy"),
                    self.last_router_weights.cpu().numpy())
        if self._task_names is not None:
            np.save(os.path.join(save_dir, "task_names.npy"),
                    np.array(self._task_names))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(global_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
