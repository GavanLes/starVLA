

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# # used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
cd /home/jye624/Projcets/starVLA

Framework_name=QwenOFT
freeze_module_list='qwen_vl_interface,dino'
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA/libero
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=1229_libero4in1_qwen3oft
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-8}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
accelerate_config_file=${ACCELERATE_CONFIG_FILE:-starVLA/config/deepseeds/deepspeed_zero2.yaml}


# Fix: ensure vonneumann1 group is active for NFS file access on compute nodes
# Worker processes spawned by accelerate/deepspeed may lose supplementary group context
if id -nG 2>/dev/null | grep -qw vonneumann1; then
  export _STARVLA_GROUP_FIX=vonneumann1
  echo "[INFO] Group vonneumann1 detected, using newgrp for NFS access"
fi

# Resolve conda activation command for sub-shells (sg spawns a new shell)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")
CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV:-starVLA}"

sg vonneumann1 -c "
${CONDA_INIT} && \
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --framework.qwenvl.attn_implementation ${attn_implementation} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starvla \
  --wandb_entity 761402180-ustb \ 
