#!/bin/bash
set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

nproc_per_node=1
save_path="./multiturn-sft-qwen-3-0.6b-sp2"

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/ubuntu/projects/dataset/agentloop/rollout/sft_filtered.parquet \
    data.val_files=/home/ubuntu/projects/dataset/agentloop/rollout/sft_filtered.parquet \
    data.max_length=8192 \
    data.train_batch_size=4 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    +data.multiturn.enable_thinking=true \
    data.truncation=left \
    data.micro_batch_size=2 \
    model.partial_pretrain=Qwen/Qwen3-0.6B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-06b-sp2 \
    trainer.logger=console \
    trainer.total_training_steps=10 $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true