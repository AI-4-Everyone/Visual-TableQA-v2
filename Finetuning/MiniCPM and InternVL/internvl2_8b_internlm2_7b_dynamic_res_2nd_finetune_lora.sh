#!/bin/bash
set -x
# Hardware configuration - optimized for single A6000 48GB
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}  # Total effective batch size
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}  # Increased for A6000 48GB
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
# Output directory
OUTPUT_DIR='/notebooks/InternVL-sft-lora'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
echo "Training Configuration:"
echo "  GPUs: ${GPUS}"
echo "  Per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient accumulation steps: ${GRADIENT_ACC}"
echo "  Total effective batch size: ${BATCH_SIZE}"
# Start training
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/notebooks/InternVL/pretrained/InternVL2-8B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/notebooks/InternVL/internvl_chat/shell/data/tableqa.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --tf32 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 2 \
  --load_best_model_at_end False \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --optim "adamw_torch_fused" \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-8 \
  --logging_steps 10 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  --remove_unused_columns False \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
