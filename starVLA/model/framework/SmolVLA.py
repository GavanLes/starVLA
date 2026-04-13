# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import contextlib



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm.smolvlm_interface import SmolVLMInterface
from starVLA.model.modules.action_model.smolvla_flow_matching import SmolVLAFlowMatching
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################

@FRAMEWORK_REGISTRY.register("SmolVLA")
class SmolVLA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        framework_cfg = getattr(self.config, "framework", None)
        qwenvl_cfg = getattr(framework_cfg, "qwenvl", None) if framework_cfg is not None else None
        self.num_vlm_layers_to_use = int(getattr(qwenvl_cfg, "num_vl_layers", 16) or 16)
        self.train_expert_only = bool(getattr(framework_cfg, "train_expert_only", True))
        self.freeze_vision_encoder = bool(getattr(framework_cfg, "freeze_vision_encoder", True))
        self.smolvlm_interface = SmolVLMInterface(config=self.config)

        # dynamic get SmolVLM config
        model_config = self.smolvlm_interface.model.config
        text_config = getattr(model_config, "text_config", None)
        num_vl_layers = self.num_vlm_layers_to_use

        llm_hidden_size = getattr(text_config, "hidden_size", None) if text_config is not None else None
        if llm_hidden_size is None:
            llm_hidden_size = getattr(model_config, "hidden_size", 2048)

        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model = SmolVLAFlowMatching(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size


    def _get_action_transformer_block_count(self):
        action_core = self.action_model
        while hasattr(action_core, "model") and not hasattr(action_core, "transformer_blocks"):
            action_core = action_core.model

        return len(action_core.transformer_blocks)

    def _get_action_device(self):
        return next(self.action_model.parameters()).device

    def _get_action_dtype(self):
        return next(self.action_model.parameters()).dtype

    def _select_vlm_hidden_states(self, all_hidden):
        available_hidden = list(all_hidden[1:])
        if not available_hidden:
            raise RuntimeError("SmolVLM did not return hidden states.")

        if len(available_hidden) >= self.num_vlm_layers_to_use:
            return available_hidden[: self.num_vlm_layers_to_use]

        if len(available_hidden) == 1:
            return available_hidden * self.num_vlm_layers_to_use

        layer_indices = np.linspace(0, len(available_hidden) - 1, self.num_vlm_layers_to_use)
        layer_indices = np.round(layer_indices).astype(int)
        return [available_hidden[index] for index in layer_indices]

    def _align_vlm_layers_to_action_layers(self, vlm_layers, action_layers):
        if len(vlm_layers) == action_layers:
            return vlm_layers

        if len(vlm_layers) == 1:
            return vlm_layers * action_layers

        layer_indices = np.linspace(0, len(vlm_layers) - 1, action_layers)
        layer_indices = np.round(layer_indices).astype(int)
        return [vlm_layers[index] for index in layer_indices]
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        # Step 1: SmolVLM input format
        qwen_inputs = self.smolvlm_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        vlm_context = torch.no_grad() if self.train_expert_only else contextlib.nullcontext()
        with vlm_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.smolvlm_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = self._select_vlm_hidden_states(all_hidden)
            action_device = self._get_action_device()
            action_dtype = self._get_action_dtype()
            vl_embs_list = [hidden.to(device=action_device, dtype=action_dtype) for hidden in vl_embs_list]
            vl_embs_list = self._align_vlm_layers_to_action_layers(vlm_layers=vl_embs_list, action_layers=self._get_action_transformer_block_count())
            base_hidden = vl_embs_list[-1]

        # Step 4: Action Expert Forward and Loss
        # 标签对齐：取最后 chunk_len 段
        actions = torch.tensor(
            np.array(actions), device=action_device, dtype=action_dtype
        )  # [B, T_full, action_dim]
        actions_target = actions[:, -(self.future_action_window_size + 1) :, :]  # (B, chunk_len, action_dim)

        repeated_diffusion_steps = int(getattr(self.config.trainer, "repeated_diffusion_steps", 1) or 1)
        if repeated_diffusion_steps > 1:
            actions_target = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if state is not None:
                state = torch.tensor(np.array(state), device=action_device, dtype=action_dtype)
                state = state.repeat(repeated_diffusion_steps, 1, 1)
        else:
            if state is not None:
                state = torch.tensor(np.array(state), device=action_device, dtype=action_dtype)

        action_loss = self.action_model(vl_embs_list, actions_target, state)  # (B, chunk_len, action_dim)



        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action( # TODO align  predict_action with forward, make api more flexible
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        from deployment.model_server.tools.image_tools import to_pil_preserve
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: SmolVLM input format
        qwen_inputs = self.smolvlm_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.smolvlm_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            all_hidden = qwenvl_outputs.hidden_states
            vl_embs_list = self._select_vlm_hidden_states(all_hidden)
            action_device = self._get_action_device()
            action_dtype = self._get_action_dtype()
            vl_embs_list = [hidden.to(device=action_device, dtype=action_dtype) for hidden in vl_embs_list]
            vl_embs_list = self._align_vlm_layers_to_action_layers(vlm_layers=vl_embs_list, action_layers=self._get_action_transformer_block_count())
            _ = vl_embs_list[-1]

        state = torch.from_numpy(np.array(state)).to(action_device, dtype=action_dtype) if state is not None else None
        pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().to(torch.float32).cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import importlib
    import os
    from PIL import Image

    try:
        debugpy = importlib.import_module("debugpy")
    except ImportError:
        debugpy = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml", help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "train"], help="smoke: no-grad forward, train: loss forward")
    args, clipargs = parser.parse_known_args()

    if debugpy is not None and os.environ.get("SMOLVLA_WAIT_FOR_DEBUGPY", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "SmolVLA"
    cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/SmolVLM2-500M-Video-Instruct"
    cfg.framework.qwenvl.num_vl_layers = 16
    cfg.framework.train_expert_only = True
    cfg.framework.freeze_vision_encoder = True
    cfg.trainer.freeze_modules = "smolvlm_interface"
    cfg.trainer.repeated_diffusion_steps = 1
    

    model = SmolVLA(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float16),
        "image": [image],
        "lang": "pick up the red block",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    if args.mode == "smoke":
        with torch.inference_mode():
            output = model.predict_action([sample])
        print("smoke forward OK")
        print(output["normalized_actions"].shape)
    else:
        with torch.no_grad():
            output = model([sample])
        print("train forward OK")
        print(output["action_loss"].item())

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
