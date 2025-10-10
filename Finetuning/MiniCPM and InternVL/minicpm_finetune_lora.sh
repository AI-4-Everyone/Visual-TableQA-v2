#!/bin/bash
GPUS_PER_NODE=1  # Changed from 8 to 1 for single GPU setup
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
 
MODEL="openbmb/MiniCPM-Llama3-V-2_5"
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="../../minicpm_ds/minicpm_train.json"
EVAL_DATA="../../minicpm_ds/minicpm_val.json"
OUTPUT="../../MiniCPM-sft-lora"
LLM_TYPE="llama3"   
MODEL_MAX_Length=4096 

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --tf32 true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "all-linear" \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 1 \
    --eval_steps 100 \
    --output_dir $OUTPUT \
    --logging_dir $OUTPUT \
    --logging_strategy "steps" \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 0.5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --optim "adamw_torch_fused" \
    --logging_steps 20 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to "tensorboard"