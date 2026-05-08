#!/usr/bin/env bash
set -euo pipefail

# Optional NCCL settings for multi-node training
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000
# export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===

#Framework_name=SmolVLA
#freeze_module_list=''
#base_vlm=playground/Pretrained_models/SmolVLM2-500M-Video-Instruct
config_yaml=./examples/LIBERO/train_files/smolvla_libero_stage2.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA/libero
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=smolvla_libero_stage2_sa
# VLM training switches

# Memory controls for 24GB GPUs
vlm_num_vl_layers=16
vla_per_device_batch_size=1
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  # --framework.name ${Framework_name} \
  # --framework.qwenvl.base_vlm ${base_vlm} \
  # --framework.qwenvl.num_vl_layers ${vlm_num_vl_layers} \
  # --framework.train_expert_only ${train_expert_only} \
  # --framework.freeze_vision_encoder ${freeze_vision_encoder} \

  # --datasets.vla_data.per_device_batch_size ${vla_per_device_batch_size} \
  # --trainer.freeze_modules ${freeze_module_list} \
  # --trainer.repeated_diffusion_steps 8 \
  # --trainer.max_train_steps 80000 \
  # --trainer.save_interval 20000 \
  # --trainer.logging_frequency 10 \
  # --trainer.eval_interval 100 \
  # --run_root_dir ${run_root_dir} \
  # --run_id ${run_id} \
  # --wandb_project starvla \
  # --wandb_entity 761402180-ustb
