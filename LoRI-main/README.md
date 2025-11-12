# LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation

[Juzheng Zhang](https://juzhengz.github.io/), [Jiacheng You](https://github.com/YouJiacheng), [Ashwinee Panda](https://kiddyboots216.github.io/), [Tom Goldstein](https://www.cs.umd.edu/~tomg/)

📄 [Paper](https://arxiv.org/abs/2504.07448) | 💻 [Code](https://github.com/juzhengz/LoRI/) | 🤗 [HuggingFace](https://huggingface.co/collections/tomg-group-umd/lori-adapters-67f795549d792613e1290011) | COLM 2025

LoRI (LoRA with Reduced Interference) is a simple yet effective variant of LoRA for fine-tuning LLMs. It freezes the projection matrices `A` as random projections and sparsifies `B` using task-specific masks. LoRI significantly reduces the number of trainable parameters, preserves single-task performance, and minimizes cross-task interference during adapter merging and continual learning.

<div align="center">
    <img src="./LoRI.png" alt="LoRI" width="80%">
</div>

## Installation

Create and activate a Conda environment:
```
conda create -n lori python=3.10 -y
conda activate lori
```

Clone the repository and install dependencies:
```
git clone https://github.com/juzhengz/LoRI.git
cd LoRI
pip install -r requirements.txt
```

## Training from Scratch

LoRI is implemented using [Fully Sharded Data Parallel](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and can be executed in a multi-GPU environment. We provide training scripts covering Natural language understanding (NLU), Code generation, Mathematical reasoning, and Safety alignment. These scripts support LLaMA-3-8B and Mistral-7B base models with adapter ranks of 32 and 64. Each script performs `LoRI-D` training, extracts sparse masks, continues with `LoRI-S` training at 90% sparsity, and evaluates on downstream tasks.

### Example script

Training code generation adapters `LoRI-D` and `LoRI-S` on the CodeAlpaca dataset using LLaMA-3-8B with rank 32:

```
bash scripts/codealpaca_llama3_r_32.sh
```

### Available scripts

- `scripts/codealpaca_*.sh` — Code generation tasks
- `scripts/gsm8k_*.sh` — Mathematical reasoning tasks
- `scripts/nlu_*.sh` — Natural language understanding tasks
- `scripts/saferpaca_*.sh` — Safety alignment tasks

## Inference with Pretrained Adapters

Pretrained LoRI adapters are available via our [HuggingFace collection](https://huggingface.co/collections/tomg-group-umd/lori-adapters-67f795549d792613e1290011) and can be loaded as follows:

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
adapter = PeftModel.from_pretrained(base_model, "tomg-group-umd/LoRI-S_code_llama3_rank_32")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

`LoRI-D` and `LoRI-S` adapters are provided for code, math, NLU, and safety tasks, using LLaMA-3-8B and Mistral-7B models at ranks 32 and 64. 

## Adapter Merging

Use the following scripts for merging adapters:

- [`scripts/merge_3_loras.sh`](https://github.com/juzhengz/LoRI/blob/main/scripts/merge_3_loras.sh)
- [`scripts/merge_4_loras.sh`](https://github.com/juzhengz/LoRI/blob/main/scripts/merge_4_loras.sh)

Update the adapter paths in the scripts to point to either your own trained adapters or those available on [HuggingFace](https://huggingface.co/collections/tomg-group-umd/lori-adapters-67f795549d792613e1290011).

## Continual Learning

Use these scripts to perform continual learning with LoRI:

- [`scripts/continual_safety_code.sh`](https://github.com/juzhengz/LoRI/blob/main/scripts/continual_safety_code.sh)
- [`scripts/continual_safety_math.sh`](https://github.com/juzhengz/LoRI/blob/main/scripts/continual_safety_math.sh)
- [`scripts/continual_safety_nlu.sh`](https://github.com/juzhengz/LoRI/blob/main/scripts/continual_safety_nlu.sh)

### For LoRI-D:

Before running the scripts, set `model_archive` to the path of your trained `LoRI-D` safety adapter, or use the [safety adapter](https://huggingface.co/tomg-group-umd/LoRI-D_safety_llama3_rank_32) from our HuggingFace collection.

```
model_archive=/path/to/your/lori-d/safety/adapter
```

### For LoRI-S:

Before running the scripts, set `model_archive` to the path of your `LoRI-S` safety adapter and set `mask_path` to the path of the sparse mask for the downstream task. Alternatively, you can use the [safety adapter](https://huggingface.co/tomg-group-umd/LoRI-S_safety_llama3_rank_32) and the corresponding [sparse mask](https://huggingface.co/tomg-group-umd/LoRI-D_code_llama3_rank_32/tree/main/masks) from our HuggingFace collection.

```
model_archive=/path/to/your/lori-s/safety/adapter
mask_path=/path/to/your/lori-d/code/adapter/masks/0.9_mask.pt
```

## Customizing Base Models and Losses

LoRI supports a variety of base models and loss functions, which can be found in the [`config/model`](https://github.com/juzhengz/LoRI/tree/main/config/model) and [`config/loss`](https://github.com/juzhengz/LoRI/tree/main/config/loss) directories of the repository. To add a new model or loss function, you can simply create a new `.yaml` file in the respective directory. 

## Acknowledgements

This project builds on the codebase of [dpo-rlaif](https://github.com/architsharma97/dpo-rlaif) and incorporates code from [lottery-ticket-adaptation](https://github.com/kiddyboots216/lottery-ticket-adaptation). We evaluate code generation performance on HumanEval using the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness).

## Citation

If you use LoRI in your work, please cite:

```
@article{zhang2025lori,
  title={LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation},
  author={Zhang, Juzheng and You, Jiacheng and Panda, Ashwinee and Goldstein, Tom},
  journal={arXiv preprint arXiv:2504.07448},
  year={2025}
}
```

Feel free to reach out if you have any questions!

