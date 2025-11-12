import torch
import monai
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
from src.lori import LoRI_sam
import os
join = os.path.join
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度相关模块

num = 1
batch = 1
rank = 4

train_dataset_path = rf"F:\医学数据集\vitb-11个标签\b-{num}\CT_Abd-Gallbladder\{batch}"
checkpoint = r"E:\SAM\sam_vit_b_01ec64.pth"
work_dir = r"F:\SAM-New\LoRA-LoRI\work_dir"
task_name = f"b{num}_{rank}_{batch}折"
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)

shutil.copyfile(
    __file__, join(model_save_path, os.path.basename(__file__))
)

sam = build_sam_vit_b(checkpoint=checkpoint)

sam_lori = LoRI_sam(sam, rank=rank)
model = sam_lori.sam

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
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度

        epoch_loss += loss.item()

        torch.cuda.empty_cache() # 主动释放未使用的显存以减少预留显存的浪费

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
