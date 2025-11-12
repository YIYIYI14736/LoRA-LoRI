#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=/path/to/your/cache/dir
export PROJECT_CACHE=/path/to/your/project/cache
export WANDB_MODE=offline
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRI-D continual learning
dataset_name=gsm8k
model=llama3
n_epochs=3
batch_size=32
grad_norm=1
save_every=epoch_$n_epochs
sparsity_ratio=0.0
lr=5e-5
lora_rank=32
lora_alpha=64
model_archive=/path/to/your/lori-d/safety/adapter

exp_name="${dataset_name}_${model}_continual/LoRI-D_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
results_path="${PROJECT_CACHE}/${dataset_name}_${model}_continual"

python -u src/train_lori.py \
        model=$model \
        model.archive=$model_archive \
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

python src/eval_model.py --model_name $model --adapter_path $adapter_path --datasets hexphi,$dataset_name --results_path $results_path --sparsity_ratio $sparsity_ratio


# LoRI-S continual learning
dataset_name=gsm8k
model=llama3
n_epochs=3
batch_size=32
grad_norm=1
save_every=epoch_$n_epochs
sparsity_ratio=0.9
lr=5e-4
lora_rank=32
lora_alpha=64
model_archive=/path/to/your/lori-s/safety/adapter

exp_name="${dataset_name}_${model}_continual/LoRI-S_rank_${lora_rank}_alpha_${lora_alpha}_lr_${lr}_bs_${batch_size}"
adapter_path="${PROJECT_CACHE}/${exp_name}/epoch-${n_epochs}"
results_path="${PROJECT_CACHE}/${dataset_name}_${model}_continual"
mask_path=/path/to/your/lori-d/math/adapter/masks/0.9_mask.pt

python -u src/train_lori.py \
        model=$model \
        model.archive=$model_archive \
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

python src/eval_model.py --model_name $model --adapter_path $adapter_path --datasets hexphi,$dataset_name --results_path $results_path --sparsity_ratio $sparsity_ratio