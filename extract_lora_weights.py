"""
从PyTorch模型文件中提取LoRA相关权重并保存到txt文件
"""
import torch
import os

def extract_lora_weights(model_path, output_txt_path):
    """
    从模型文件中提取LoRA相关权重并保存到txt文件
    
    Args:
        model_path: 模型文件路径 (.pth)
        output_txt_path: 输出txt文件路径
    """
    print(f"正在加载模型文件: {model_path}")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 如果checkpoint是字典且包含'state_dict'键，则提取state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 查找所有LoRA相关的权重
    # LoRA权重通常包含: linear_a, linear_b, w_a, w_b 等关键字
    lora_keys = []
    for key in state_dict.keys():
        # 检查是否包含LoRA相关的关键字
        if any(keyword in key.lower() for keyword in ['linear_a', 'linear_b', 'lora', 'w_a', 'w_b']):
            lora_keys.append(key)
    
    if len(lora_keys) == 0:
        print("警告: 未找到LoRA相关的权重!")
        print("模型中的所有键名:")
        for key in list(state_dict.keys())[:20]:  # 只显示前20个
            print(f"  - {key}")
        if len(state_dict.keys()) > 20:
            print(f"  ... 共 {len(state_dict.keys())} 个键")
    else:
        print(f"找到 {len(lora_keys)} 个LoRA相关的权重:")
        for key in lora_keys:
            print(f"  - {key}")
    
    # 将LoRA权重保存到txt文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"LoRA权重提取结果\n")
        f.write(f"模型文件: {model_path}\n")
        f.write(f"提取时间: {torch.__version__}\n")
        f.write("=" * 80 + "\n\n")
        
        if len(lora_keys) == 0:
            f.write("未找到LoRA相关的权重!\n\n")
            f.write("模型中的所有键名:\n")
            for key in state_dict.keys():
                f.write(f"  {key}\n")
        else:
            f.write(f"共找到 {len(lora_keys)} 个LoRA相关的权重:\n\n")
            
            for key in lora_keys:
                weight = state_dict[key]
                f.write("-" * 80 + "\n")
                f.write(f"权重名称: {key}\n")
                f.write(f"形状: {list(weight.shape)}\n")
                f.write(f"数据类型: {weight.dtype}\n")
                f.write(f"参数数量: {weight.numel()}\n")
                f.write(f"最小值: {weight.min().item():.6f}\n")
                f.write(f"最大值: {weight.max().item():.6f}\n")
                f.write(f"均值: {weight.mean().item():.6f}\n")
                f.write(f"标准差: {weight.std().item():.6f}\n")
                f.write("\n权重值:\n")
                
                # 将权重转换为numpy数组并保存
                weight_np = weight.detach().cpu().numpy()
                if weight_np.ndim == 1:
                    # 一维数组，每行一个值
                    for i, val in enumerate(weight_np):
                        f.write(f"{val:.8e}\n")
                elif weight_np.ndim == 2:
                    # 二维数组，按行保存
                    for i in range(weight_np.shape[0]):
                        row_str = " ".join([f"{val:.8e}" for val in weight_np[i]])
                        f.write(f"{row_str}\n")
                else:
                    # 多维数组，展平后保存
                    weight_flat = weight_np.flatten()
                    for i, val in enumerate(weight_flat):
                        f.write(f"{val:.8e}\n")
                        if (i + 1) % 1000 == 0:  # 每1000个值换行分隔
                            f.write("\n")
                
                f.write("\n")
    
    print(f"\nLoRA权重已保存到: {output_txt_path}")
    print(f"文件大小: {os.path.getsize(output_txt_path) / 1024 / 1024:.2f} MB")


def _load_state_dict_generic(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        return checkpoint
    else:
        return checkpoint


def compare_lora_weights(model_path_stage1, model_path_stage2, output_txt_path):
    print(f"正在加载模型文件(Stage1): {model_path_stage1}")
    print(f"正在加载模型文件(Stage2): {model_path_stage2}")

    sd1 = _load_state_dict_generic(model_path_stage1)
    sd2 = _load_state_dict_generic(model_path_stage2)

    def _collect_lora_keys(state_dict):
        keys = []
        for k in state_dict.keys():
            if any(kw in k.lower() for kw in ['linear_a', 'linear_b', 'lora', 'w_a', 'w_b']):
                keys.append(k)
        return keys

    lora_keys_1 = set(_collect_lora_keys(sd1))
    lora_keys_2 = set(_collect_lora_keys(sd2))
    common_keys = sorted(list(lora_keys_1.intersection(lora_keys_2)))

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LoRA权重差异对比结果\n")
        f.write(f"Stage1: {model_path_stage1}\n")
        f.write(f"Stage2: {model_path_stage2}\n")
        f.write("=" * 80 + "\n\n")

        if len(common_keys) == 0:
            f.write("未找到同时存在于两阶段的LoRA相关权重键\n")
        else:
            f.write(f"共找到 {len(common_keys)} 个共同的LoRA权重键:\n\n")
            for key in common_keys:
                w1 = sd1[key].to(torch.float32)
                w2 = sd2[key].to(torch.float32)
                diff = (w2 - w1).cpu()

                f.write("-" * 80 + "\n")
                f.write(f"权重名称: {key}\n")
                f.write(f"形状: {list(diff.shape)}\n")
                f.write(f"数据类型: {diff.dtype}\n")
                f.write(f"参数数量: {diff.numel()}\n")
                f.write(f"差值最小值: {diff.min().item():.6f}\n")
                f.write(f"差值最大值: {diff.max().item():.6f}\n")
                f.write(f"差值均值: {diff.mean().item():.6f}\n")
                f.write(f"差值标准差: {diff.std().item():.6f}\n")
                f.write("\n差值矩阵:\n")

                diff_np = diff.numpy()
                if diff_np.ndim == 1:
                    for val in diff_np:
                        f.write(f"{val:.8e}\n")
                elif diff_np.ndim == 2:
                    for i in range(diff_np.shape[0]):
                        row_str = " ".join([f"{val:.8e}" for val in diff_np[i]])
                        f.write(f"{row_str}\n")
                else:
                    flat = diff_np.flatten()
                    for i, val in enumerate(flat):
                        f.write(f"{val:.8e}\n")
                        if (i + 1) % 1000 == 0:
                            f.write("\n")

    print(f"\nLoRA差值已保存到: {output_txt_path}")
    print(f"文件大小: {os.path.getsize(output_txt_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    # 示例：比较同一任务目录下的 Stage1 与 Stage2 的最佳模型
    num = 4
    rank = 4
    batch = 1
    base_dir = rf"F:\SAM-New\LoRA-LoRI\work_dir\b{num}_{rank}_{batch}折_LoRI"
    stage1_path = os.path.join(base_dir, "medsam_model_stage1_best.pth")
    stage2_path = os.path.join(base_dir, "medsam_model_stage2_best.pth")

    # 单独导出 Stage1 与 Stage2 的 LoRA 权重到 1.txt / 2.txt
    output_stage1_txt = rf"F:\SAM-New\LoRA-LoRI\1.txt"
    output_stage2_txt = rf"F:\SAM-New\LoRA-LoRI\2.txt"
    extract_lora_weights(stage1_path, output_stage1_txt)
    extract_lora_weights(stage2_path, output_stage2_txt)

    # 生成差值输出到 3.txt
    output_diff_path = rf"F:\SAM-New\LoRA-LoRI\3.txt"
    compare_lora_weights(stage1_path, stage2_path, output_diff_path)

