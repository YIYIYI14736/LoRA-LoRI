import re
import torch
import os
from get_datasets import get_batch_iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import pad_to_length, all_gather_if_needed
from rouge_score import rouge_scorer
from peft import PeftModel
import numpy as np
import argparse
import yaml
from eval_model import evaluate
cache_dir = os.getenv("PROJECT_CACHE", "~/.cache")
commonsense_tasks = [
    'boolq',
    'piqa',
    'social_i_qa',
    'arc-challenge',
    'arc-easy',
    'openbookqa',
    'hellaswag',
    'winogrande',
]
torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
parser.add_argument('--adapter_path', type=str, default=['mistralai/Mistral-7B-v0.1'], nargs='+')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--sample', action='store_true')
parser.add_argument('--datasets', type=str, default='commonsense,gsm8k')
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--results_path', type=str, default='results')
parser.add_argument('--combination_type', type=str, default='cat')
parser.add_argument('--density', type=float, default=0.7)
parser.add_argument('--weights', type=float, default=[1.0, 1.0, 1.0, 1.0], nargs='+')
args = parser.parse_args()
args.datasets = args.datasets.replace("commonsense", ','.join(commonsense_tasks))
args.datasets = args.datasets.split(',')

model_yaml_path = os.path.join('config', 'model', f"{args.model_name}.yaml")
with open(model_yaml_path, 'r') as file:
    model_config = yaml.safe_load(file)
    load_path = model_config.get('name_or_path')

base_model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained(load_path)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.resize_token_embeddings(len(tokenizer))
            
model = PeftModel.from_pretrained(base_model, args.adapter_path[0], adapter_name='math')
model.load_adapter(args.adapter_path[1], adapter_name="commonsense")
model.load_adapter(args.adapter_path[2], adapter_name="code")
model.load_adapter(args.adapter_path[3], adapter_name="safety")

adapters = ["math", "commonsense", "code", "safety"]
adapter_name = "merge"
model.add_weighted_adapter(adapters, args.weights, adapter_name, combination_type=args.combination_type, density=args.density)
model.set_adapter("merge")
model.merge_and_unload()
for _ in range(args.num_runs):
    for dataset in args.datasets:
        acc = evaluate(dataset, model, tokenizer, args)
        os.makedirs(os.path.join(args.results_path, "results"), exist_ok=True)
        with open(f"{args.results_path}/results/merge_4_loras.txt", "a") as f:
            f.write(f"Model: {args.adapter_path}\nCombination_type: {args.combination_type}, density: {args.density}, weights: {args.weights}\nDataset: {dataset}, Accuracy: {acc * 100}\n")