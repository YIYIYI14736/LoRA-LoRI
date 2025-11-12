import torch
import os
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from argparse import ArgumentParser
import yaml
import re
from collections import defaultdict
torch.manual_seed(0)

def load_model_tokenizer(args):
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
                
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    return model, tokenizer


def check_sparsity(model):
    layer_sparsity = {}
    total_zero = 0
    total_elements = 0

    for name, param in model.named_parameters():
        if "lora_B.default.weight" in name:
            layer_id = name.split(".")[4]
            zero_count = (param == 0).sum().item()
            total_count = param.numel()

            if layer_id not in layer_sparsity:
                layer_sparsity[layer_id] = {"zero": 0, "total": 0}

            layer_sparsity[layer_id]["zero"] += zero_count
            layer_sparsity[layer_id]["total"] += total_count

            total_zero += zero_count
            total_elements += total_count

    for key in layer_sparsity:
        layer_sparsity[key] = layer_sparsity[key]["zero"] / layer_sparsity[key]["total"]
        print(f"Layer {key} sparsity: {layer_sparsity[key]:.4f}")

    total_sparsity = total_zero / total_elements
    print(f"Total sparsity: {total_sparsity:.4f}")

    return layer_sparsity, total_sparsity


def get_model_mask(model, sparsity_ratios, save_path):
    mask_dict = {name: param for name, param in model.named_parameters() if "lora_B.mask" in name}
    all_params_abs = []
    for name, p in model.named_parameters():
        if 'lora_B.default.weight' in name:
            param_values = p.data.view(-1)
            if mask_dict:
                mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                constraint = mask_dict[mask_name].view(-1)
                param_values[constraint] = 0
            all_params_abs.append(torch.abs(param_values).cpu())

    all_params_abs = torch.cat(all_params_abs)
    total_num = all_params_abs.numel()
    
    for sparsity_ratio in sparsity_ratios:
        k = int((1 - sparsity_ratio) * total_num)
        threshold = torch.topk(all_params_abs, k).values[-1]
        print(f"Threshold for {sparsity_ratio:.2f} sparsity: {threshold:.6f}")
        
        retained_num = 0
        constrained_mask = {}
        for name, param in model.named_parameters():
            if 'lora_B.default.weight' in name:
                constrained_mask[name] = (torch.abs(param.data) >= threshold).to('cpu')
                if mask_dict:
                    mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                    constrained_mask[name] = constrained_mask[name] & ~mask_dict[mask_name]
                retained_num += constrained_mask[name].sum().item()

        print(f"{(100 * retained_num / total_num):.2f}% of parameters will be retained.")
        os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)
        torch.save(constrained_mask, f"{save_path}/masks/{sparsity_ratio}_mask.pt")
        print(f"Model mask saved to {save_path}/masks/{sparsity_ratio}_mask.pt")
    
    return


def get_modulewise_mask(model, sparsity_ratios, save_path):
    mask_dict = {name: param for name, param in model.named_parameters() if "lora_B.mask" in name}
    attention_keys = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_keys = ["gate_proj", "up_proj", "down_proj"]

    module_groups = {
        "attention": {"params": [], "names": []},
        "mlp": {"params": [], "names": []}
    }

    for name, param in model.named_parameters():
        if "lora_B.default.weight" in name:
            if any(f".{key}." in name for key in attention_keys):
                group = "attention"
            elif any(f".{key}." in name for key in mlp_keys):
                group = "mlp"
            else:
                continue  # skip unrelated modules

            param_values = param.data.view(-1).clone()
            if mask_dict:
                mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                constraint = mask_dict[mask_name].view(-1)
                param_values[constraint] = 0

            module_groups[group]["params"].append(torch.abs(param_values).cpu())
            module_groups[group]["names"].append(name)

    for sparsity_ratio in sparsity_ratios:
        constrained_mask = {}
        total_retained = 0
        total_elements = 0

        for group_name, group_data in module_groups.items():
            all_params = torch.cat(group_data["params"])
            total_num = all_params.numel()
            k = int((1 - sparsity_ratio) * total_num)
            threshold = torch.topk(all_params, k).values[-1]
            retained_num = 0

            for i, name in enumerate(group_data["names"]):
                param = dict(model.named_parameters())[name]
                param_mask = (torch.abs(param.data) >= threshold).to('cpu')
                if mask_dict:
                    mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                    param_mask = param_mask & ~mask_dict[mask_name]
                constrained_mask[name] = param_mask
                retained_num += param_mask.sum().item()

            total_retained += retained_num
            total_elements += total_num

        total_sparsity = 1 - (total_retained / total_elements)
        print(f"Total sparsity for {sparsity_ratio:.2f}: {total_sparsity:.4f}")

        os.makedirs(os.path.join(save_path, "modulewise_masks"), exist_ok=True)
        torch.save(constrained_mask, f"{save_path}/modulewise_masks/{sparsity_ratio}_mask.pt")
        print(f"Module mask saved to {save_path}/modulewise_masks/{sparsity_ratio}_mask.pt")

    return


def get_projectionwise_mask(model, sparsity_ratios, save_path):
    mask_dict = {name: param for name, param in model.named_parameters() if "lora_B.mask" in name}
    module_types = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    module_params = {mod: [] for mod in module_types}
    module_names = {mod: [] for mod in module_types}

    for name, param in model.named_parameters():
        if "lora_B.default.weight" in name:
            for mod in module_types:
                if f".{mod}." in name:
                    param_values = param.data.view(-1).clone()
                    if mask_dict:
                        mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                        constraint = mask_dict[mask_name].view(-1)
                        param_values[constraint] = 0
                    module_params[mod].append(torch.abs(param_values).cpu())
                    module_names[mod].append(name)
                    break

    for sparsity_ratio in sparsity_ratios:
        constrained_mask = {}
        total_retained = 0
        total_elements = 0

        for mod in module_types:
            all_params = torch.cat(module_params[mod])
            total_num = all_params.numel()
            k = int((1 - sparsity_ratio) * total_num)
            threshold = torch.topk(all_params, k).values[-1]
            retained_num = 0

            for i, name in enumerate(module_names[mod]):
                param = dict(model.named_parameters())[name]
                param_mask = (torch.abs(param.data) >= threshold).to('cpu')
                if mask_dict:
                    mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                    param_mask = param_mask & ~mask_dict[mask_name]
                constrained_mask[name] = param_mask
                retained_num += param_mask.sum().item()

            total_retained += retained_num
            total_elements += total_num

        total_sparsity = 1 - (total_retained / total_elements)
        print(f"Total sparsity for {sparsity_ratio:.2f}: {total_sparsity:.4f}")

        os.makedirs(os.path.join(save_path, "projectionwise_masks"), exist_ok=True)
        torch.save(constrained_mask, f"{save_path}/projectionwise_masks/{sparsity_ratio}_mask.pt")
        print(f"Projection mask saved to {save_path}/projectionwise_masks/{sparsity_ratio}_mask.pt")

    return


def get_layerwise_mask(model, sparsity_ratios, save_path):
    mask_dict = {name: param for name, param in model.named_parameters() if "lora_B.mask" in name}
    layer_params = {}

    for name, param in model.named_parameters():
        if "lora_B.default.weight" in name:
            layer_id = name.split(".")[4]
            if layer_id not in layer_params:
                layer_params[layer_id] = []
            param_values = param.data.view(-1)
            if mask_dict:
                mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                constraint = mask_dict[mask_name].view(-1)
                param_values[constraint] = 0
            layer_params[layer_id].append(torch.abs(param_values).cpu())

    for sparsity_ratio in sparsity_ratios:
        constrained_mask = {}
        total_retained = 0
        total_elements = 0

        for layer_id, params in layer_params.items():
            layer_params_abs = torch.cat(params)
            total_num = layer_params_abs.numel()
            retained_num = 0
            k = int((1 - sparsity_ratio) * total_num)
            threshold = torch.topk(layer_params_abs, k).values[-1]

            for name, param in model.named_parameters():
                if f'layers.{layer_id}.' in name and 'lora_B.default.weight' in name:
                    param_mask = (torch.abs(param.data) >= threshold).to('cpu')
                    if mask_dict:
                        mask_name = name.replace("lora_B.default.weight", "lora_B.mask")
                        param_mask = param_mask & ~mask_dict[mask_name]
                    constrained_mask[name] = param_mask
                    retained_num += constrained_mask[name].sum().item()

            total_retained += retained_num
            total_elements += total_num

        total_sparsity = 1 - (total_retained / total_elements)
        print(f"Total sparsity for {sparsity_ratio:.2f}: {total_sparsity:.4f}")

        os.makedirs(os.path.join(save_path, "layerwise_masks"), exist_ok=True)
        torch.save(constrained_mask, f"{save_path}/layerwise_masks/{sparsity_ratio}_mask.pt")
        print(f"Layer mask saved to {save_path}/layerwise_masks/{sparsity_ratio}_mask.pt")

    return


def get_matrixwise_mask(model, sparsity_ratios, save_path):
    for sparsity_ratio in sparsity_ratios:
        matrix_masks = {}
        total_retained = 0
        total_elements = 0

        for name, param in model.named_parameters():
            if "lora_B.default.weight" in name:
                param_values = param.data.view(-1)
                total_num = param_values.numel()
                k = int((1 - sparsity_ratio) * total_num)
                threshold = torch.topk(torch.abs(param_values), k).values[-1]
                
                param_mask = (torch.abs(param.data) >= threshold).to('cpu')
                matrix_masks[name] = param_mask

                total_retained += param_mask.sum().item()
                total_elements += total_num

        total_sparsity = 1 - (total_retained / total_elements)
        print(f"Total sparsity for {sparsity_ratio:.2f}: {total_sparsity:.4f}")

        os.makedirs(os.path.join(save_path, "matrixwise_masks"), exist_ok=True)
        torch.save(matrix_masks, f"{save_path}/matrixwise_masks/{sparsity_ratio}_mask.pt")
        print(f"Matrix mask saved to {save_path}/matrixwise_masks/{sparsity_ratio}_mask.pt")


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--adapter_path', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument("--sparsity_ratios", type=float, default=[0.9], nargs='+')
    args = parser.parse_args()
    model, tokenizer = load_model_tokenizer(args)
    get_model_mask(model, args.sparsity_ratios, args.adapter_path)
