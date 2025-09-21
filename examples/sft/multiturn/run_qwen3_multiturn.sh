#!/bin/bash
set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

nproc_per_node=8
model_path=Qwen/Qwen3-14B
max_len=32768
save_path="./multiturn-sft-qwen-3-14b-$max_len"
train_data=/workspace/verl/mulititurn-sft-cleaned.parquet

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_data \
    data.val_files=$train_data \
    data.max_length=$max_len \
    data.train_batch_size=16 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    +data.multiturn.enable_thinking=true \
    data.truncation=left \
    data.micro_batch_size_per_gpu=1 \
    +model.gradient_checkpointing=true \
    model.partial_pretrain=$model_path \
    model.strategy=fsdp2 \
    model.fsdp_config.wrap_policy.min_num_params=0 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=$expname \
    trainer.logger=console \
    trainer.save_freq=50 \
    trainer.logger='["wandb", "console"]' \
    trainer.total_epochs=3 $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true


