import os
import numpy as np
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from PIL import Image
import torchvision.transforms as transforms

from src.lori import LoRI_sam          # ★ 用 LoRI_sam，而不是 LoRA_sam
from src.segment_anything.utils.transforms import ResizeLongestSide
from src.segment_anything import sam_model_registry
from utils.SurfaceDice import compute_dice_coefficient
from matplotlib.patches import Rectangle

torch.manual_seed(2023)
np.random.seed(2023)

# ================== 基本配置 ==================
num = 1
batch = 1
rank = 4
device = "cuda:0" 

npz_ts_path = rf"F:\医学数据集\vitb-11个标签\b-{num}\CT_Abd-Gallbladder\{3 - batch}"
test_npzs = sorted(os.listdir(npz_ts_path))
sam_ckpt = r"E:\SAM\sam_vit_b_01ec64.pth"
lori_ckpt = f"F:\SAM-New\LoRA-LoRI\work_dir\b{num}_{rank}_{batch}折_LoRI\medsam_model_stage2_best.pth"

output_dir = rf"F:\SAM-New\LoRA-LoRI\work_dir\b{num}_{rank}_{batch}折"
os.makedirs(output_dir, exist_ok=True)

# ================== 构建 SAM + LoRI ==================
# 1) 构建原始 SAM 结构
ori_sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)

# 按 LoRA/LoRI 做法：先把原始参数冻结（LoRI_sam 内部对 image_encoder 也会再处理）
for param in ori_sam.parameters():
    param.requires_grad = False

rank = 4   # 和训练时保持一致
sam_lori = LoRI_sam(ori_sam, rank=rank)
lori_sam = sam_lori.sam    # 挂完 LoRI 之后的 SAM 模型

# 2) 加载 LoRI 训练好的完整模型权重（包括 base + A/B）
state_dict = torch.load(lori_ckpt, map_location="cpu")
lori_sam.load_state_dict(state_dict, strict=True)

lori_sam = lori_sam.to(device)
lori_sam.eval()

# ================== 一些工具函数（跟你原来基本一样） ==================
def get_random_points_from_mask(mask, num_points=5):
    y_indices, x_indices = np.where(mask == 1)
    if len(x_indices) == 0:
        # 没点就随便给几个 (0,0)，反正这种情况一般不会走到这里
        return [[0, 0]] * num_points

    points = np.column_stack((x_indices, y_indices))
    selected_indices = np.random.choice(len(points), size=num_points, replace=True)
    selected_points = points[selected_indices]
    return selected_points.tolist()

def get_boxs(mask):
    # 获取框
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        # 没前景就给整幅图
        H, W = mask.shape
        return np.array([0, 0, W - 1, H - 1])

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 5))
    x_max = min(W - 1, x_max + np.random.randint(0, 5))
    y_min = max(0, y_min - np.random.randint(0, 5))
    y_max = min(H - 1, y_max + np.random.randint(0, 5))
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

all_Dice = []
all_SurfaceDice = []

# ================== 循环每个 npz 做 3D (slice-wise) 推理 ==================
for npz_idx in range(len(test_npzs)):
    npz_name = test_npzs[npz_idx]
    npz = np.load(join(npz_ts_path, npz_name), allow_pickle=True)
    imgs = npz['imgs']   # [N, H, W, 3]
    gts  = npz['gts']    # [N, H, W]

    medsam_segs = []

    # 遍历每一张 slice
    for img, gt in tqdm(list(zip(imgs, gts)), desc=f"{npz_name}"):
        non_zero_elements_exist = np.any(gt > 0)

        if non_zero_elements_exist and gt.sum() > 1:
            sam_trans = ResizeLongestSide(lori_sam.image_encoder.img_size)

            resize_img = sam_trans.apply_image(img)  # (1024, 1024, 3)
            resize_img_tensor = torch.as_tensor(
                resize_img.transpose(2, 0, 1)
            ).to(device)  # (3, H, W)
            input_image = lori_sam.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)

            with torch.no_grad():
                # 1) image encoder
                image_embedding = lori_sam.image_encoder(input_image.to(device))  # (1, 256, 64, 64)

                original_image_size = img.shape[:2]
                transform = ResizeLongestSide(lori_sam.image_encoder.img_size)

                # 2) 点提示
                points = np.array([get_random_points_from_mask(gt)])  # (1, num_points, 2)
                points = points[0]  # (num_points, 2)

                point_coords = transform.apply_coords(points, original_image_size)
                coords_torch = torch.as_tensor(
                    point_coords, dtype=torch.float, device=device
                )
                point_coords = coords_torch[None, :, :]  # (1, num_points, 2)
                num_points = point_coords.shape[1]

                labels_torch = torch.full(
                    (1, num_points), 1, dtype=torch.float, device=device
                )  # (1, num_points)

                point_torch = (point_coords, labels_torch)

                # 3) 框提示
                boxes = np.array([get_boxs(gt)])  # (1, 4)
                sam_trans = ResizeLongestSide(lori_sam.image_encoder.img_size)
                box = sam_trans.apply_boxes(boxes, (gt.shape[-2], gt.shape[-1]))  # (1, 4)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

                # 4) prompt encoder
                sparse_embeddings, dense_embeddings = lori_sam.prompt_encoder(
                    points=point_torch,
                    boxes=box_torch,
                    masks=None,
                )

                # 5) mask decoder
                medsam_seg_prob, _ = lori_sam.mask_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=lori_sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
                medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                medsam_segs.append(medsam_seg)

                # --------- 保存可视化图像 ----------
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].imshow(img)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                axs[1].imshow(gt, cmap='gray')
                axs[1].set_title('Ground Truth')
                axs[1].axis('off')

                axs[2].imshow(medsam_seg, cmap='gray')
                axs[2].set_title('LoRI-SAM Segmentation')
                axs[2].axis('off')

                # 画 bbox 和 points
                box_vis = get_boxs(gt)
                rect = Rectangle(
                    (box_vis[0], box_vis[1]),
                    box_vis[2] - box_vis[0],
                    box_vis[3] - box_vis[1],
                    linewidth=2,
                    edgecolor='green',
                    facecolor='none'
                )
                axs[2].add_patch(rect)

                selected_points = get_random_points_from_mask(gt)
                for p in selected_points:
                    axs[2].scatter(p[0], p[1], color='red', s=50)

                plt.suptitle(f'{npz_name.split(".npz")[0]} slice')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_name = f'{npz_name.split(".npz")[0]}_{len(medsam_segs):03d}.png'
                plt.savefig(os.path.join(output_dir, save_name))
                plt.close(fig)

        else:
            # 没前景：直接全 0 mask
            medsam_seg_zeros = np.zeros_like(gt, dtype=np.uint8)
            medsam_segs.append(medsam_seg_zeros)

    # ================== 3D 上计算 Dice / Surface Dice ==================
    from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance

    gts_binary = np.array(gts)
    medsam_binary = np.array(medsam_segs)

    medsam_dice = compute_dice_coefficient(gts_binary, medsam_binary)

    madsam_surface_dist = compute_surface_distances(
        gts_binary, medsam_binary, spacing_mm=[1, 1, 1]
    )
    madsam_surface_dice = compute_surface_dice_at_tolerance(
        madsam_surface_dist, tolerance_mm=2.0
    )

    all_Dice.append(medsam_dice)
    all_SurfaceDice.append(madsam_surface_dice)

    print(npz_name.split('.npz')[0], "测试结果：")
    print(f"LoRI-SAM Dice: {medsam_dice:.4f}")
    print()

# ================== 汇总结果 ==================
avg_Dice = sum(all_Dice) / len(all_Dice)
avg_SurfaceDice = sum(all_SurfaceDice) / len(all_SurfaceDice)

print()
print('Average LoRI-SAM Dice:', avg_Dice)
# print('Average LoRI-SAM SurfaceDice:', avg_SurfaceDice)
print()
