# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
SmolVLA variant: proprioceptive state injected into VLM text input
instead of being fed directly to the action head.

============================================================================
  如何搭建自己的 VLA 模型 — 架构教学
============================================================================

一个 VLA (Vision-Language-Action) 模型由两个核心组件构成：

  ┌─────────────────────────────────────────────────────┐
  │ 1. VLM Backend (视觉语言模型)                        │
  │    - 输入: images (多视图) + instruction (文本指令)   │
  │    - 输出: hidden_states (多层特征, [B, seq_len, H])  │
  │    - 作用: 理解场景 + 指令，产出富含语义的特征          │
  │                                                     │
  │ 2. Action Head (动作头)                              │
  │    - 输入: VLM hidden_states + noise/actions          │
  │    - 输出: predicted actions [B, T, action_dim]       │
  │    - 作用: 从视觉-语言特征中解码出连续动作轨迹           │
  └─────────────────────────────────────────────────────┘

  数据流总结:
    images + instruction → [VLM] → hidden_states (16层, [B, seq_len, 2048])
                                       │
                                [Action Head / DiT] → actions (7-DoF trajectory)

  DiT block 结构 (每个 block):
    norm → Attention → residual → norm → FFN → residual

    Attention 层根据 encoder_hidden_states 是否为 None 切换行为:
      - 传了 encoder_hidden_states → cross-attention (Q=DiT序列, K/V=VLM特征)
      - 传了 None                  → self-attention  (Q/K/V=DiT序列)

    当前 SmolVLA 默认 interleave_self_attention=False，每一层 DiT block
    都传了 VLM 特征，所以全部走 cross-attention，没有 self-attention。
    (论文作者注: "TODO miss self att and _process_output, but work well")

  VLM ↔ DiT 层对应关系:
    DiTConfig["num_layers"] = num_vl_layers (SmolVLA 默认 16)
    所以 VLM 16 层 ←→ DiT 16 block，一一对应，无需对齐操作。
    _align_vlm_layers_to_action_layers 仅在用户手动改了两边层数不一致时
    才做线性插值，属于防御性代码。

  关于 state (本体感知信息):
    state 是机器人当前关节角度、末端位姿等 proprioceptive 信息。
    它有两种注入位置:

    方案 A: 注入 Action Head (原版 SmolVLA)
      state → MLP编码 → 拼接到 DiT 输入序列前面
      优点: 简单，不动 VLM
      缺点: VLM 不知道机器人当前状态，语义理解缺少本体感知上下文

    方案 B: 注入 VLM (本文件)
      state → 格式化为文本 → 拼接到 instruction 后面
      优点: VLM 能结合 state 理解场景，语义更丰富
      缺点: 文本 token 有精度损失（通常可忽略）

============================================================================
  搭建自己的 VLA — step by step
============================================================================

  Step 1: 继承 baseframework
    基类帮你处理了:
    - from_pretrained()  加载/保存 checkpoint
    - compute_loss()     训练时 loss 分发
    - 与 Trainer 的对接
    你只需要实现 forward() 和 predict_action()

  Step 2: 选 VLM
    StarVLA 的 VLM 接口统一在 starVLA.model.modules.vlm 下
    支持 Qwen2.5-VL, Qwen3-VL, SmolVLM 等
    全部暴露相同接口: build_qwenvl_inputs() + forward()

  Step 3: 选 Action Head
    starVLA.model.modules.action_model 下有多种:
    - flow matching (DiT-based, GR00T/AML style)
    - MLP (简单回归)
    - FAST (离散 token 预测)
    选一个匹配你的任务

  Step 4: 注册到 FRAMEWORK_REGISTRY
    @FRAMEWORK_REGISTRY.register("YourName")
    然后 config 中 framework.name = "YourName" 即可使用

  Step 5: 决定 state 注入位置
    这就是本文件要展示的——你可以自由选择 state 放进哪
============================================================================
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import contextlib

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace 中用来标记"忽略"位置的 label id
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm.smolvlm_interface import SmolVLMInterface
from starVLA.model.modules.action_model.smolvla_flow_matching import SmolVLAFlowMatching
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


# ============================================================================
# 核心实现
# ============================================================================

@FRAMEWORK_REGISTRY.register("SmolVLAStateInVLM")
class SmolVLAStateInVLM(baseframework):
    """
    ============================================================================
    SmolVLA 变体：state 信息注入 VLM 而非 Action Head
    ============================================================================

    组件:
      1. SmolVLM Interface  — 轻量 VLM (500M 参数)，负责视觉+语言+state 融合
      2. Flow Matching Head — Layer-wise DiT，从 VLM 多层 hidden states 解码动作

    与原始 SmolVLA 的区别:
      - state 向量格式化为文本 "Robot state: [...]" 追加到 instruction
      - Action Head 不再接收单独的 state tensor (state 信息已在 VLM 特征中)

    ============================================================================
    设计笔记
    ============================================================================

    Q: 为什么要 train_expert_only=True (默认冻结 VLM)?
    A: VLM 参数量大，全量微调容易过拟合且慢。冻结 VLM 只用其 pretrained
       语义表征能力，让 Action Head 专注学习"特征→动作"的映射。
       如果你想端到端训练，设 train_expert_only=False。

    Q: 为什么用多层 hidden states 而不是只用最后一层？
    A: 不同层的特征包含不同粒度的信息:
       - 浅层: 更多低级视觉特征 (边缘、纹理)
       - 深层: 更多高级语义特征 (物体类别、空间关系)
       Layer-wise cross-attention 让 Action Head 能同时利用多层信息。

    Q: state 作为文本有什么问题？
    A: 浮点数 → 字符串 → tokenize，有精度损失。对 7-DoF 动作来说通常可忽略。
       如果 state_dim 很大（>100），建议用专门的 state encoder 投影到 VLM
       embedding 空间（即方案 A 或混合方案）。
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        ====================================================================
        __init__ 做了什么
        ====================================================================

        一个 VLA 模型的 __init__ 需要完成:
          1. 读取 config 中的超参数
          2. 初始化 VLM backend
          3. 获取 VLM 的 hidden_size / num_layers (Action Head 需要对齐)
          4. 初始化 Action Head
          5. 设置 action window 等训练参数

        下面是逐步拆解:
        """

        # ---- Step 0: 调用基类构造函数 ----
        # baseframework 继承自 HuggingFace PreTrainedModel，帮你初始化
        # 一些 HF 基础设施（config 存储等）
        super().__init__()
        self.config = config

        # ---- Step 1: 从 config 中提取参数 ----
        # config 是一个 OmegaConf/dict 对象，层级结构为:
        #   config.framework.qwenvl.*       — VLM 相关
        #   config.framework.action_model.* — Action Head 相关
        #   config.framework.train_expert_only — 是否只训练 expert
        #   config.trainer.*                — 训练相关
        framework_cfg = getattr(self.config, "framework", None)
        qwenvl_cfg = getattr(framework_cfg, "qwenvl", None) if framework_cfg is not None else None

        # 用多少层 VLM hidden states (取最后 N 层)
        self.num_vlm_layers_to_use = int(getattr(qwenvl_cfg, "num_vl_layers", 16) or 16)

        # train_expert_only=True: 冻结 VLM，只训练 Action Head
        self.train_expert_only = bool(getattr(framework_cfg, "train_expert_only", True))

        # freeze_vision_encoder: 是否冻结视觉编码器
        self.freeze_vision_encoder = bool(getattr(framework_cfg, "freeze_vision_encoder", True))

        # state 格式化精度（小数点后几位）
        self.state_format_precision = int(getattr(qwenvl_cfg, "state_format_precision", 4) or 4)

        # ---- Step 2: 初始化 VLM Backend ----
        # SmolVLMInterface 封装了:
        #   - AutoModelForImageTextToText (SmolVLM2-500M)
        #   - AutoProcessor (图像预处理 + tokenizer)
        #   - 冻结逻辑 (train_expert_only / freeze_vision_encoder)
        #   - 图像尺寸对齐 (保证能被 patch_size 整除)
        self.smolvlm_interface = SmolVLMInterface(config=self.config)

        # ---- Step 3: 获取 VLM 的架构参数 ----
        # Action Head 的 hidden_size 必须和 VLM 对齐，因为要做 cross-attention
        model_config = self.smolvlm_interface.model.config
        text_config = getattr(model_config, "text_config", None)

        # hidden_size: VLM 每个 token 的特征维度 (SmolVLM2-500M 是 2048)
        llm_hidden_size = (
            getattr(text_config, "hidden_size", None)
            if text_config is not None
            else None
        )
        if llm_hidden_size is None:
            llm_hidden_size = getattr(model_config, "hidden_size", 2048)

        # 把 VLM 架构参数写回 config，供 Action Head 读取
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = self.num_vlm_layers_to_use

        # ---- Step 4: 初始化 Action Head ----
        # SmolVLAFlowMatching 内部是一个 LayerwiseFlowmatchingActionHead:
        #   - DiT transformer (cross-attention to VLM features)
        #   - ActionEncoder (sinusoidal time encoding + MLP)
        #   - ActionDecoder (MLP → action_dim)
        #   - StateEncoder (MLP — 但我们不会用，state 走 VLM)
        self.action_model = SmolVLAFlowMatching(config=self.config)

        # ---- Step 5: Action 时间窗口 ----
        # future_action_window_size: 预测未来多少步
        # past_action_window_size: 用过去多少步作为上下文 (通常为 0)
        # chunk_len = past + 当前步 + future
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = getattr(
            config.framework.action_model, "past_action_window_size", 0
        )
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # 读取 interleave_self_attention 配置，与 LayerwiseFM_ActionHeader 保持一致
        diffusion_cfg = getattr(config.framework.action_model, "diffusion_model_cfg", None) or {}
        self.interleave_self_attention = diffusion_cfg.get("interleave_self_attention", False)

    # ========================================================================
    # 辅助方法
    # ========================================================================

    # 以下方法从 SmolVLA 原样搬过来。它们处理的是 VLM 特征和
    # Action Head 之间的"胶水"逻辑，与 state 注入位置无关。

    def _get_action_transformer_block_count(self):
        """获取 Action Head 中 DiT 的 transformer block 数量。

        递归查找 .transformer_blocks 属性，因为 action_model
        可能有包装层 (SmolVLAFlowMatching → LayerwiseFlowmatchingActionHead → DiT)。
        """
        action_core = self.action_model
        while hasattr(action_core, "model") and not hasattr(action_core, "transformer_blocks"):
            action_core = action_core.model
        return len(action_core.transformer_blocks)

    def _get_action_device(self):
        """Action Head 所在的设备 (cuda:0 等)。"""
        return next(self.action_model.parameters()).device

    def _get_action_dtype(self):
        """Action Head 的参数 dtype (bfloat16/float32)。"""
        return next(self.action_model.parameters()).dtype

    def _select_vlm_hidden_states(self, all_hidden):
        """从 VLM 所有层的 hidden states 中选取最后 N 层。

        Args:
            all_hidden: tuple of [B, seq_len, H], 每层一个, 第 0 层是 embedding

        Returns:
            list of [B, seq_len, H], 长度 = num_vlm_layers_to_use

        为什么跳过第 0 层? 第 0 层是 input embedding (未经 transformer 处理)，
        信息太浅，跳过它从第 1 层开始取。
        """
        available_hidden = list(all_hidden[1:])  # 跳过 embedding 层
        if not available_hidden:
            raise RuntimeError("SmolVLM did not return hidden states.")

        if len(available_hidden) >= self.num_vlm_layers_to_use:
            return available_hidden[: self.num_vlm_layers_to_use]

        if len(available_hidden) == 1:
            return available_hidden * self.num_vlm_layers_to_use

        # VLM 层数不足时，线性插值采样
        layer_indices = np.linspace(
            0, len(available_hidden) - 1, self.num_vlm_layers_to_use
        )
        layer_indices = np.round(layer_indices).astype(int)
        return [available_hidden[index] for index in layer_indices]

    def _align_vlm_layers_to_action_layers(self, vlm_layers, action_layers):
        """将 VLM 层数与 Action Head 的 cross-attention block 数对齐。

        interleave 模式下 DiT 有 num_vl_layers*2 层，半数做 cross-attention，
        所以对齐目标为 action_layers // 2，默认 16→16 无需操作。
        """
        if self.interleave_self_attention:
            action_layers = action_layers // 2

        if len(vlm_layers) == action_layers:
            return vlm_layers

        if len(vlm_layers) == 1:
            return vlm_layers * action_layers

        layer_indices = np.linspace(0, len(vlm_layers) - 1, action_layers)
        layer_indices = np.round(layer_indices).astype(int)
        return [vlm_layers[index] for index in layer_indices]

    # ========================================================================
    # State → VLM text injection (本文件的核心新增逻辑)
    # ========================================================================

    @staticmethod
    def _format_state_text(state_array: np.ndarray, precision: int = 4) -> str:
        """将 state 向量格式化为 VLM 可读的文本。

        Input:  np.array([0.123, -0.456, 1.789])  shape [state_dim]
        Output: "Robot state: [0.1230, -0.4560, 1.7890]"

        设计选择:
          - 用 "Robot state:" 前缀明确语义角色
          - 用方括号包裹让 VLM 理解这是一个向量
          - precision 控制精度/长度权衡

        你也可以自定义格式，比如:
          - 命名每个维度: "joint_0=0.12, joint_1=-0.45, ..."
          - 分多行: "Robot state:\n  position: [x, y, z]\n  rotation: [...]"
        """
        flat = np.asarray(state_array).flatten()
        parts = [f"{v:.{precision}f}" for v in flat]
        return "Robot state: [" + ", ".join(parts) + "]"

    def _inject_state_into_instructions(self, examples: List[dict]) -> None:
        """原地修改 examples: 将 state 文本追加到 lang，然后删除 state key。

        这是唯一的数据预处理步骤。修改后:
          - example["lang"] 从 "pick up the block" 变成
            "pick up the block\nRobot state: [0.12, -0.45, 1.79, ...]"
          - example["state"] 被删除（消失）

        后续父类代码读到 "state" not in example 时，自然地不对 Action Head
        传 state，因此不需要修改 Action Head 的任何代码。
        """
        for ex in examples:
            if "state" in ex and ex["state"] is not None:
                state_text = self._format_state_text(
                    ex["state"], self.state_format_precision
                )
                ex["lang"] = f"{ex['lang']}\n{state_text}"
                del ex["state"]

    # ========================================================================
    # 核心方法: forward (训练) 和 predict_action (推理)
    # ========================================================================

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        ====================================================================
        训练前向传播
        ====================================================================

        输入 examples 格式 (List[dict]):
          [
            {
              "image":  [PIL.Image, ...],   # 多视图图像列表
              "lang":   "pick up the block", # 自然语言指令
              "action": np.ndarray [T, 7],   # 动作序列 (7-DoF)
              "state":  np.ndarray [7,],      # 本体感知 (关节角/末端位姿)
            },
            ...  # batch 中的其他样本
          ]

        数据流:
          examples
            │
            ├─[1]─→ _inject_state_into_instructions()
            │        state 文本追加到 lang，state key 删除
            │
            ├─[2]─→ SmolVLM Interface
            │        images + modified instructions → hidden_states
            │        (VLM 现在"看到"了 state 信息)
            │
            ├─[3]─→ _select_vlm_hidden_states()
            │        选取最后 N 层 hidden states
            │
            ├─[4]─→ _align_vlm_layers_to_action_layers()
            │        对齐 VLM 层数与 DiT block 数
            │
            └─[5]─→ Action Head (state=None)
                      hidden_states + noisy actions → loss
                      (Action Head 用 cross-attention 从 VLM 特征中解码动作)

        返回:
          {"action_loss": torch.Tensor (scalar)}
        """

        # ---- Step 1: 提取数据 + state 注入 ----
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # ---- Step 2: 将 state 注入 instruction ----
        # 这一步是本变体的核心。修改后:
        #   - instruction 包含了 state 文本
        #   - state key 被删除
        self._inject_state_into_instructions(examples)

        # ---- Step 3: 构建 VLM 输入并前向传播 ----
        # build_qwenvl_inputs 负责:
        #   1. 将 images + instruction 组装成 chat messages 格式
        #   2. apply_chat_template (模板化)
        #   3. tokenize + pad
        #   4. 返回 {input_ids, attention_mask, pixel_values, ...}
        qwen_inputs = self.smolvlm_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        # train_expert_only=True 时，VLM 走 no_grad (冻结 VLM 参数)
        vlm_context = torch.no_grad() if self.train_expert_only else contextlib.nullcontext()
        with vlm_context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.smolvlm_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,  # 必须! 需要多层 hidden states
                    return_dict=True,
                )

            # ---- Step 4: 处理 VLM 输出 ----
            all_hidden = qwenvl_outputs.hidden_states
            # 选取 N 层 hidden states
            vl_embs_list = self._select_vlm_hidden_states(all_hidden)
            # 转移到 Action Head 的设备/dtype
            action_device = self._get_action_device()
            action_dtype = self._get_action_dtype()
            vl_embs_list = [
                hidden.to(device=action_device, dtype=action_dtype)
                for hidden in vl_embs_list
            ]
            # 对齐层数
            vl_embs_list = self._align_vlm_layers_to_action_layers(
                vlm_layers=vl_embs_list,
                action_layers=self._get_action_transformer_block_count(),
            )

        # ---- Step 5: 准备 Action 标签 ----
        actions = torch.tensor(
            np.array(actions), device=action_device, dtype=action_dtype
        )  # [B, T_full, action_dim]

        # 只取最后 chunk_len 步作为 target
        actions_target = actions[
            :, -(self.future_action_window_size + 1):, :
        ]  # [B, chunk_len, action_dim]

        # repeated_diffusion_steps: 每个样本重复多次扩散步数(数据增强)
        repeated_diffusion_steps = int(
            getattr(self.config.trainer, "repeated_diffusion_steps", 1) or 1
        )
        if repeated_diffusion_steps > 1:
            actions_target = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list = [
                h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list
            ]

        # ---- Step 6: Action Head 前向传播 ----
        # state=None: state 信息已经在 VLM hidden states 中了
        action_loss = self.action_model(vl_embs_list, actions_target, state=None)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        ====================================================================
        推理: 给定观测，预测未来动作序列
        ====================================================================

        与 forward() 的流程一致，区别在于:
          - 不需要 actions 标签 (用随机噪声初始化)
          - Action Head 内部做 flow matching 逆扩散采样
          - 返回的是 normalized actions (需要 unnormalize 才能给机器人执行)

        返回:
          {"normalized_actions": np.ndarray [B, T, action_dim]}
        """

        if type(examples) is not list:
            examples = [examples]

        from deployment.model_server.tools.image_tools import to_pil_preserve

        # ---- Step 1: 提取数据 ----
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        # ---- Step 2: state 注入 VLM ----
        self._inject_state_into_instructions(examples)

        # ---- Step 3: 图像尺寸对齐 (可选) ----
        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "image_size", None
        )
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # ---- Step 4: VLM 编码 ----
        qwen_inputs = self.smolvlm_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
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
            vl_embs_list = [
                hidden.to(device=action_device, dtype=action_dtype)
                for hidden in vl_embs_list
            ]
            vl_embs_list = self._align_vlm_layers_to_action_layers(
                vlm_layers=vl_embs_list,
                action_layers=self._get_action_transformer_block_count(),
            )

        # ---- Step 5: 推理动作 (state=None) ----
        pred_actions = self.action_model.predict_action(vl_embs_list, state=None)

        normalized_actions = pred_actions.detach().to(torch.float32).cpu().numpy()
        return {"normalized_actions": normalized_actions}


# ============================================================================
# 自己搭建 VLA 的 checklist
# ============================================================================
#
# 1. 选 VLM:
#    from starVLA.model.modules.vlm.xxx_interface import XxxInterface
#    或者直接用 SmolVLMInterface / QWen3 等现成的
#
# 2. 选 Action Head:
#    from starVLA.model.modules.action_model.xxx import XxxHead
#    或者用现成的 SmolVLAFlowMatching / FlowmatchingActionHead / MLP 等
#
# 3. 写你的类:
#    @FRAMEWORK_REGISTRY.register("MyVLA")
#    class MyVLA(baseframework):
#        def __init__(self, config):
#            super().__init__()
#            self.vlm = XxxInterface(config)
#            self.action_head = XxxHead(config)
#
#        def forward(self, examples):
#            # a. 从 examples 提取 images, instructions, actions, (state)
#            # b. VLM 编码 → hidden_states
#            # c. (可选) 处理 state
#            # d. Action Head 前向 → loss
#            return {"action_loss": loss}
#
#        def predict_action(self, examples):
#            # 同上，但 Action Head 做推理模式
#            return {"normalized_actions": actions}
#
# 4. Config:
#    framework:
#      name: "MyVLA"
#      qwenvl:
#        base_vlm: "your/vlm/path"
#        num_vl_layers: 16
#      action_model:
#        future_action_window_size: 10
#        action_dim: 7
#        state_dim: 7
#
# 5. 使用:
#    from starVLA.model.framework.base_framework import build_framework
#    model = build_framework(cfg)
#    loss = model(examples)
#    actions = model.predict_action(examples)
# ============================================================================


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
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke",
        choices=["smoke", "train"],
        help="smoke: no-grad forward, train: loss forward",
    )
    args, clipargs = parser.parse_known_args()

    if debugpy is not None and os.environ.get("SMOLVLA_WAIT_FOR_DEBUGPY", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "SmolVLAStateInVLM"
    cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/SmolVLM2-500M-Video-Instruct"
    cfg.framework.qwenvl.num_vl_layers = 16
    cfg.framework.train_expert_only = True
    cfg.framework.freeze_vision_encoder = True
    cfg.trainer.freeze_modules = "smolvlm_interface"
    cfg.trainer.repeated_diffusion_steps = 1

    model = SmolVLAStateInVLM(cfg)
    print(model)
    print("\n[SmolVLAStateInVLM action model]\n")
    print(model.action_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float16),
        "image": [image],
        "lang": "pick up the red block",
        "state": np.random.uniform(-1, 1, size=(7,)).astype(np.float16),
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
