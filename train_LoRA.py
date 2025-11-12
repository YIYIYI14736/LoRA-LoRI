import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import os
join = os.path.join
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度相关模块

# Take dataset path
train_dataset_path = "dataset/train"
# Load SAM model
checkpoint = r"E:\SAM\Sam_LoRA-main - 多卡\sam_vit_b_01ec64.pth"

work_dir = 'work_dir/btcv'
task_name = 'b1_4_1折'
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(work_dir, task_name + "-" + run_id)
os.makedirs(model_save_path, exist_ok=True)

shutil.copyfile(
    __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
)

sam = build_sam_vit_b(checkpoint=checkpoint)

sam_lora = LoRA_sam(sam, 4)
model = sam_lora.sam

processor = Samprocessor(model)

train_ds = DatasetSegmentation(train_dataset_path, processor)

train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

freeze_non_lora_params(model)

# for param in model.mask_decoder.parameters():
#     param.requires_grad = True

for name, param in model.named_parameters():
    if "linear" in name:  # 只解冻包含 "linear" 的参数
        param.requires_grad = True

optimizer = Adam(model.image_encoder.parameters(), lr=1e-3, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.train()
model.to(device)

print("可训练参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

losses = []
best_loss = 1e10

for epoch in range(num_epochs):

    # log_gpu_memory_usage("gpu_memory_log.txt")  # 每个 epoch 记录一次显存占用
    epoch_loss = 0
    accumulation_steps = 8
    for i, batch in enumerate(tqdm(train_dataloader)):
        # for bat in batch:
        #     bat['image'] = bat['image'].to(device)

        outputs = model(batched_input=batch,
                        multimask_output=False)

        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)  # We need to get the [B, C, H, W] starting from [H, W]

        loss = seg_loss(stk_out, stk_gt.float().to(device))

        # print(f"Model parameters memory: {sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2:.2f} MB")
        # print(f"Total allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        # print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度

        epoch_loss += loss.item()

        torch.cuda.empty_cache() # 主动释放未使用的显存以减少预留显存的浪费


#
# scaler = GradScaler()
#
# for epoch in range(num_epochs):
#     epoch_loss = 0
#     accumulation_steps = 4  # 梯度累积步数
#
#     for i, batch in enumerate(tqdm(train_dataloader)):
#         # 前向传播和计算损失放在 autocast 中
#         with autocast():  # 启用混合精度计算
#             outputs = model(batched_input=batch, multimask_output=False)
#
#             stk_gt, stk_out = utils.stacking_batch(batch, outputs)
#             stk_out = stk_out.squeeze(1)
#             stk_gt = stk_gt.unsqueeze(1)  # 调整为 [B, C, H, W]
#
#             # 计算损失
#             loss = seg_loss(stk_out, stk_gt.float().to(device))
#
#         print(f"Model parameters memory: {sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2:.2f} MB")
#         print(f"Total allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
#         print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
#
#         epoch_loss += loss.item()
#         # 混合精度的反向传播
#         scaler.scale(loss).backward()
#
#         # 梯度累积步数控制更新
#         if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
#             # 使用 Scaler 进行优化器步数更新
#             scaler.step(optimizer)
#             scaler.update()  # 更新 Scaler
#             optimizer.zero_grad()  # 清空梯度

    print(f'EPOCH: {epoch}')
    print(f'loss training: {epoch_loss}')

    # save the latest model checkpoint
    torch.save(model.state_dict(), join(model_save_path, 'medsam_model_latest.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), join(model_save_path, 'medsam_model_best.pth'))
        # save loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        # plot loss and save figure
        plt.plot(losses)
        plt.title('Cross Entropy Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(join(model_save_path, 'train_updata_loss.png'))
        plt.close()

# plot loss
plt.plot(losses)
plt.title('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show() # comment this line if you are running on a server
plt.savefig(join(model_save_path, 'train_loss.png'))
plt.close()


# # Save the parameters of the model in safetensors format
# rank = 4
# sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
# torch.save(model.state_dict(), 'model_weights.pth')