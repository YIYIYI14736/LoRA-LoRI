#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=/path/to/your/cache/dir
export PROJECT_CACHE=/path/to/your/project/cache
export WANDB_MODE=offline
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRI-D training and mask extraction
dataset_name=commonsense
model=mistral7b
n_epochs=1
batch_size=32
grad_norm=1
save_every=epoch_$n_epochs
sparsity_ratio=0.0
lr=1e-5
lora_rank=32
lora_alpha=64

exp_name="${dataset_name}_${model}/LoRI-D_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
results_path="${PROJECT_CACHE}/${dataset_name}_${model}"

python -u src/train_lori.py \
        model=$model \
        datasets=[$dataset_name] \
        exp_name=$exp_name \
        lr=$lr \
        save_every=$save_every \
        n_epochs=$n_epochs \
        batch_size=$batch_size \
        model.fsdp_policy_mp=bfloat16 \
        fsdp_port=$MASTER_PORT \
        optimizer=AdamW \
        grad_norm_strategy=even \
        max_grad_norm=$grad_norm \
        lora_rank=$lora_rank \
        lora_alpha=$lora_alpha

python src/eval_model.py --model_name $model --adapter_path $adapter_path --datasets $dataset_name --results_path $results_path --sparsity_ratio $sparsity_ratio

python src/extract_mask.py --model_name $model --adapter_path $adapter_path --sparsity_ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99

# LoRI-S training
sparsity_ratio=0.9
lr=1e-4
lora_rank=32
lora_alpha=64

mask_path="${adapter_path}/masks/0.9_mask.pt"
exp_name="${dataset_name}_${model}/LoRI-S_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
results_path="${PROJECT_CACHE}/${dataset_name}_${model}"

python -u src/train_lori.py \
        model=$model \
        datasets=[$dataset_name] \
        exp_name=$exp_name \
        lr=$lr \
        save_every=$save_every \
        n_epochs=$n_epochs \
        batch_size=$batch_size \
        model.fsdp_policy_mp=bfloat16 \
        fsdp_port=$MASTER_PORT \
        optimizer=AdamW \
        grad_norm_strategy=even \
        max_grad_norm=$grad_norm \
        lora_rank=$lora_rank \
        lora_alpha=$lora_alpha \
        mask_path=$mask_path

python src/eval_model.py --model_name $model --adapter_path $adapter_path --datasets $dataset_name --results_path $results_path --sparsity_ratio $sparsity_ratio