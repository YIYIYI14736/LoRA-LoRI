import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b  # , SamPredictor
from src.lora import LoRA_sam   # 仅为对比/兼容，可不使用
from src.lori import LoRI_sam   # 你刚实现的 LoRI_sam

import os
join = os.path.join
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

# ----------------- 基本配置 -----------------
num = 4
batch = 1
rank = 4

train_dataset_path = rf"F:\医学数据集\vitb-11个标签\b-{num}\CT_Abd-Gallbladder\{batch}"
checkpoint = r"E:\SAM\sam_vit_b_01ec64.pth"
work_dir = r"F:\SAM-New\LoRA-LoRI\work_dir"
task_name = f"b{num}_{rank}_{batch}折_LoRI"
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)

# 备份当前训练脚本
shutil.copyfile(__file__, join(model_save_path, os.path.basename(__file__)))

# ----------------- 构建 SAM + LoRI 结构 -----------------
sam = build_sam_vit_b(checkpoint=checkpoint)

# LoRI：在 SAM 的 image_encoder 上插 LoRA(A,B)，并把 A 冻结、B 训练
sam_lori = LoRI_sam(sam, rank=rank)
model = sam_lori.sam

processor = Samprocessor(model)

train_ds = DatasetSegmentation(train_dataset_path, processor)
train_dataloader = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=True,
                              collate_fn=collate_fn)

# 注意：LoRI_sam 里已经自己处理了冻结原始 encoder、设置 A/B 的 requires_grad，
# 不再需要你之前那种 “freeze_non_lora_params + 只解冻 linear” 的逻辑。

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

print("=== LoRI Stage1 & Stage2 训练 ===")
print("设备:", device)

# ----------------- 损失函数 -----------------
seg_loss = monai.losses.DiceCELoss(sigmoid=True,
                                   squared_pred=True,
                                   reduction='mean')

# ----------------- 训练配置 -----------------
num_epochs_stage1 = 10   # 第一阶段：dense LoRI（B 全部可训练）
num_epochs_stage2 = 10   # 第二阶段：sparse LoRI（按 mask 只训练 Top-k）
accumulation_steps = 8
sparsity_ratio = 0.9     # 稀疏率：90% 位置变 0，只保留 10% 的 B 参数
lr_stage1 = 1e-3
lr_stage2 = 1e-3         # 你可以酌情调小一点
weight_decay_B = 0.0     # 关键：对 B 不做 weight decay，保证 mask=0 的位置不动

# ----------------- Stage1：dense LoRI 训练（只训练所有 B） -----------------
# 使用 LoRI_sam 提供的接口，只优化 B 权重
optimizer = Adam(sam_lori.get_B_parameters(), lr=lr_stage1, weight_decay=weight_decay_B)

print("=== Stage1：dense LoRI 训练（所有 B 参数参与训练） ===")
print("可训练参数数量（B）：", sum(p.numel() for p in sam_lori.get_B_parameters()))
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

losses = []
best_loss = 1e10

for epoch in range(num_epochs_stage1):
    epoch_loss = 0.0

    for i, batch in enumerate(tqdm(train_dataloader)):
        # SAM 接口本身会把 image 等搬到 device，
        # 如果你自己的 Dataset 没做，可以在这里手动 .to(device)

        outputs = model(batched_input=batch,
                        multimask_output=False)

        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)  # [H, W] -> [B, C, H, W]

        loss = seg_loss(stk_out, stk_gt.float().to(device))
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        torch.cuda.empty_cache()

    losses.append(epoch_loss)
    print(f'[Stage1] EPOCH: {epoch}  loss: {epoch_loss}')

    # 保存 latest
    # torch.save(model.state_dict(), join(model_save_path, 'medsam_model_stage1_latest.pth'))
    # 保存 best（以 Stage1 的 epoch_loss 为准）
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), join(model_save_path, 'medsam_model_stage1_best.pth'))

    # 每 10 个 epoch 画一次 loss
    if (epoch + 1) % 10 == 0:
        plt.plot(losses)
        plt.title('Stage1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(join(model_save_path, 'train_stage1_loss_curve.png'))
        plt.close()

# 可选：保存 LoRI 的 A/B 权重和 Stage1 的 dense B
sam_lori.save_lori_parameters(join(model_save_path, 'sam_lori_stage1_dense.safetensors'))

# ----------------- 构建稀疏 mask（基于 Stage1 的 B） -----------------
print("=== 构建全局 Top-k 稀疏 mask（基于 Stage1 的 B） ===")
masks, threshold = sam_lori.compute_global_top_masks(
    sparsity_ratio=sparsity_ratio,
    device="cpu",  # mask 存在 CPU 即可
)
print(f"Global Top-k 阈值（基于 |B|）：{threshold:.6f}")

# 把 mask==0 的 B 位置直接置 0
sam_lori.apply_mask_to_B(zero_out_unselected=True)

# 在 B.weight 上注册 grad hook：grad = grad * mask
sam_lori.register_gradient_masks()

# ----------------- Stage2：sparse LoRI 训练（只训练 mask==1 的 B） -----------------
print("=== Stage2：sparse LoRI 训练（仅 mask==1 的 B 继续更新） ===")
# 重新构建 optimizer，仍然只优化 B，且关闭 weight_decay
optimizer = Adam(sam_lori.get_B_parameters(), lr=lr_stage2, weight_decay=weight_decay_B)

losses_stage2 = []
best_loss_stage2 = 1e10

for epoch in range(num_epochs_stage2):
    epoch_loss = 0.0

    for i, batch in enumerate(tqdm(train_dataloader)):

        outputs = model(batched_input=batch,
                        multimask_output=False)

        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)

        loss = seg_loss(stk_out, stk_gt.float().to(device))
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        torch.cuda.empty_cache()

    losses_stage2.append(epoch_loss)
    print(f'[Stage2] EPOCH: {epoch}  loss: {epoch_loss}')

    # 保存 latest（Stage2）
    torch.save(model.state_dict(), join(model_save_path, 'medsam_model_stage2_latest.pth'))
    # 保存 Stage2 best
    if epoch_loss < best_loss_stage2:
        best_loss_stage2 = epoch_loss
        torch.save(model.state_dict(), join(model_save_path, 'medsam_model_stage2_best.pth'))

    if (epoch + 1) % 10 == 0:
        plt.plot(losses_stage2)
        plt.title('Stage2 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(join(model_save_path, 'train_stage2_loss_curve.png'))
        plt.close()

# 最终再画一张总的曲线
plt.plot(list(range(len(losses))), losses, label='Stage1')
offset = len(losses)
plt.plot(list(range(offset, offset + len(losses_stage2))), losses_stage2, label='Stage2')
plt.title('LoRI Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(join(model_save_path, 'train_lori_loss.png'))
plt.close()
