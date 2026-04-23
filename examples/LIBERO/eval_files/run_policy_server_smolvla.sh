#!/bin/bash
set -euo pipefail

cd /home/robot/yjw/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

# ================== SmolVLA Policy Server ==================
star_vla_python=/home/robot/anaconda3/envs/starVLA/bin/python
your_ckpt=/home/robot/yjw/starVLA/results/Checkpoints/smolvla_libero_stage2/checkpoints/steps_50000_pytorch_model.pt
gpu_id=0
port=5694

# export DEBUG=true
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
  --ckpt_path ${your_ckpt} \
  --port ${port} \
  --use_bf16
