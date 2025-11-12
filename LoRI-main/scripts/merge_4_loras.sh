#!/bin/bash
# Set cache directories (please update the paths to your own)
export HF_HOME=/path/to/your/cache/dir
export PROJECT_CACHE=/path/to/your/project/cache
export WANDB_MODE=offline
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=1

# LoRI-D merging
model_name=llama3
adapter_path_1=/path/to/your/lori-d/nlu/adapter
adapter_path_2=/path/to/your/lori-d/math/adapter
adapter_path_3=/path/to/your/lori-d/code/adapter
adapter_path_4=/path/to/your/lori-d/safety/adapter
results_path=/path/to/save/results

# Concat merging
python src/merge_4_loras.py \
        --dataset commonsense,gsm8k,hexphi \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --results_path $results_path \
        --combination_type cat --weights 0.4 0.4 0.4 0.4

accelerate launch bigcode/main_merge_4.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type cat --weights 0.4 0.4 0.4 0.4

# Linear merging
python src/merge_4_loras.py \
        --dataset commonsense,gsm8k,hexphi \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --results_path $results_path \
        --combination_type linear --weights 0.4 0.4 0.4 0.4

accelerate launch bigcode/main_merge_4.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type linear --weights 0.4 0.4 0.4 0.4


# LoRI-S merging
model_name=llama3
adapter_path_1=/path/to/your/lori-s/nlu/adapter
adapter_path_2=/path/to/your/lori-s/math/adapter
adapter_path_3=/path/to/your/lori-s/code/adapter
adapter_path_4=/path/to/your/lori-s/safety/adapter
results_path=/path/to/save/results

# Concat merging
python src/merge_4_loras.py \
        --dataset commonsense,gsm8k,hexphi \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --results_path $results_path \
        --combination_type cat --weights 0.3 0.3 0.3 0.3

accelerate launch bigcode/main_merge_4.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type cat --weights 0.3 0.3 0.3 0.3

# Linear merging
python src/merge_4_loras.py \
        --dataset commonsense,gsm8k,hexphi \
        --model_name $model_name \
        --adapter_path $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --results_path $results_path \
        --combination_type linear --weights 0.3 0.3 0.3 0.3

accelerate launch bigcode/main_merge_4.py \
        --model $model_name \
        --peft_model $adapter_path_1 $adapter_path_2 $adapter_path_3 $adapter_path_4 \
        --metric_output_path $results_path \
        --tasks humaneval \
        --temperature 0.2 \
        --n_samples 20 \
        --batch_size 10 \
        --allow_code_execution \
        --combination_type linear --weights 0.3 0.3 0.3 0.3
