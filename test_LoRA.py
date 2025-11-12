import os
import numpy as np
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from src.lora import LoRA_sam
from PIL import Image
import torchvision.transforms as transforms
from src.segment_anything.utils.transforms import ResizeLongestSide
from src.segment_anything import sam_model_registry
from utils.SurfaceDice import compute_dice_coefficient
from matplotlib.patches import Rectangle
torch.manual_seed(2023)
np.random.seed(2023)

num = 1

# npz_ts_path = f"/dev/shm/btcv/vitb-11个标签/b-{num}/CT_Abd-Gallbladder/2"
npz_ts_path = '/dev/shm/3Dircadb/vith/h-4肝脏肿瘤/2'
test_npzs = sorted(os.listdir(npz_ts_path))
device = "cuda:1"
# 重新创建模型架构
checkpoint = "/dev/shm/SAM/LoRA-sam_vit_b/work_dir/sam模型/sam_vit_b_01ec64.pth"
ori_sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

for param in ori_sam.parameters():
    param.requires_grad = False
target_modules = ["image_encoder.blocks.{}.attn.qkv".format(i) for i in range(12)]

sam_lora = LoRA_sam(ori_sam, 4)
lora_sam = sam_lora.sam  # 获取经过LoRA微调的SAM模型

# 加载训练好的权重
# lora_sam.load_state_dict(
#     torch.load(f"/dev/shm/SAM/LoRA平均图/work_dir/k1_k2_0.75_0.75/b{num}_1折/medsam_model_best.pth"))
lora_sam.load_state_dict(
    torch.load("/dev/shm/SAM/LoRA平均图/work_dir/k1_k2_0.25_0.25/肝脏肿瘤_1折/medsam_model_best.pth"))

lora_sam = lora_sam.to(device)
lora_sam.eval()

# result_path = "/usr/local/SAM/SAM-ltk/work_dir/3dircadb/vith-100轮/label2-肝脏/pred"
# if not os.path.exists(result_path):
#     os.makedirs(result_path)
output_dir = f'/dev/shm/SAM/LoRA平均图/work_dir/k1_k2_0.25_0.25/out改/肝脏肿瘤_1折'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_Dice = []
all_SurfaceDice = []

for i in range(len(test_npzs)):

    npz_idx = i
    npz = np.load(join(npz_ts_path, test_npzs[npz_idx]), allow_pickle=True)
    imgs = npz['imgs']
    gts = npz['gts']


    ####################################### 随机打点#################################################
    def get_random_points_from_mask(mask):
        y_indices, x_indices = np.where(mask == 1)
        points = np.column_stack((x_indices, y_indices))

        # Randomly choose five indices from the list of indices, allowing repetition
        selected_indices = np.random.choice(len(points), size=5, replace=True)
        selected_points = points[selected_indices]
        # Convert the selected points to a list of lists
        selected_points_list = selected_points.tolist()

        return selected_points_list


    def get_boxs(mask):

        # 获取框
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = mask.shape
        x_min = max(0, x_min - np.random.randint(0, 5))
        x_max = min(W, x_max + np.random.randint(0, 5))
        y_min = max(0, y_min - np.random.randint(0, 5))
        y_max = min(H, y_max + np.random.randint(0, 5))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return bboxes


    medsam_segs = []
    
    # for j in tqdm(range(len(imgs))):
    #     img, gt = imgs[j], gts[j]
    #     non_zero_elements_exist = np.any(gt > 0)

    #     if j == 0 or j == len(imgs) - 1: 
    #         img_processed = img
    #     else:
    #         img_processed = np.mean([imgs[j - 1], imgs[j], imgs[j + 1]], axis=0).astype(np.uint8)

    #     if non_zero_elements_exist and gt.sum() > 1:  # 掩码存在并且像素大于1个 否则直接进入else
    #         sam_trans = ResizeLongestSide(lora_sam.image_encoder.img_size)
    #         resize_img = sam_trans.apply_image(img_processed)
    #         resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    #         input_image = lora_sam.preprocess(resize_img_tensor[None, :, :, :])

    for img, gt in tqdm(zip(imgs, gts)):
        non_zero_elements_exist = np.any(gt > 0)
        if non_zero_elements_exist and gt.sum() > 1:  # 掩码存在并且像素大于1个 否则直接进入else

            sam_trans = ResizeLongestSide(lora_sam.image_encoder.img_size)
            # print(111,img.shape)   #(256, 256, 3)
            resize_img = sam_trans.apply_image(img)
            # print(resize_img.shape)   (1024, 1024, 3)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = lora_sam.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)

            with torch.no_grad():
                image_embedding = lora_sam.image_encoder(input_image.to(device))  # (1, 256, 64, 64)
                original_image_size = img.shape[:2]
                transform = ResizeLongestSide(lora_sam.image_encoder.img_size)

                # 设置点
                point = np.array([get_random_points_from_mask(gt)])  # 获取两个点

                point = point[0]  # 降维

                point_coords = transform.apply_coords(point, original_image_size)
                coords_torch = torch.as_tensor(
                    point_coords, dtype=torch.float, device=device
                )
                point_coords = coords_torch[None, :, :]
                num_points = point_coords.shape[0]
                # 设置标签
                labels_torch = torch.full((num_points, 5), 1, dtype=torch.float, device=device)

                # 点和标签合在一起作为输入
                point_torch = (point_coords, labels_torch)

                # 设置框提示
                boxes = np.array([get_boxs(gt)])
                # box_np = boxes.numpy()
                sam_trans = ResizeLongestSide(lora_sam.image_encoder.img_size)
                box = sam_trans.apply_boxes(boxes, (gt.shape[-2], gt.shape[-1]))
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

                sparse_embeddings, dense_embeddings = lora_sam.prompt_encoder(
                    points=point_torch,
                    boxes=box_torch,
                    masks=None,
                )
                medsam_seg_prob, _ = lora_sam.mask_decoder(
                    image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
                    image_pe=lora_sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
                # convert soft mask to hard mask
                medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                medsam_segs.append(medsam_seg)

                # 绘制和保存图像
                # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img)
                axs[0].set_title('Original Image')
                axs[0].axis('off')

                # axs[1].imshow(img_processed)
                # axs[1].set_title('Processed Image')
                # axs[1].axis('off')

                axs[1].imshow(gt, cmap='gray')
                axs[1].set_title('Ground Truth')
                axs[1].axis('off')

                axs[2].imshow(medsam_seg, cmap='gray')
                axs[2].set_title('MedSAM Segmentation')
                axs[2].axis('off')

                medsam_seg_with_box = np.stack([medsam_seg] * 3, axis=-1)

                # 获取矩形框
                box = get_boxs(gt)
                color = 'green'
                rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=color,
                                 facecolor='none')
                axs[2].add_patch(rect)

                selected_points = get_random_points_from_mask(gt)
                for point in selected_points:
                    axs[2].scatter(point[0], point[1], color='red', s=50)

                plt.suptitle(f'{test_npzs[npz_idx].split(".npz")[0]} Image {i + 1}')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(output_dir, f'{test_npzs[npz_idx].split(".npz")[0]}_{i + 1}.png'))
                plt.close(fig)
                i += 1


        else:
            # 如果 gt 中的所有元素都零或者只有一个1，则添加全零的 medsam_seg
            medsam_seg_zeros = np.zeros_like(gt, dtype=np.uint8)
            medsam_segs.append(medsam_seg_zeros)

    from sklearn.metrics import classification_report, accuracy_score
    from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance
    import numpy as np

    gts_binary = np.array(gts)
    medsam_binary = np.array(medsam_segs)

    # compute metrics
    medsam_dice = compute_dice_coefficient(gts_binary, medsam_binary)

    madsam_surface_dice = compute_surface_distances(gts_binary, medsam_binary, spacing_mm=[1, 1, 1])
    madsam_surface_dice = compute_surface_dice_at_tolerance(madsam_surface_dice, tolerance_mm=2.0)

    all_Dice.append(medsam_dice)
    all_SurfaceDice.append(madsam_surface_dice)

    print(test_npzs[npz_idx].split('.npz')[0], "测试结果：")
    print(f"MedSAM Dice: {medsam_dice:.4f}")
    print()
    # print(f"MedSAM Surface Dice: {madsam_surface_dice:.4f}")

# 计算各自指标的平均值
avg_Dice = sum(all_Dice) / len(all_Dice)
avg_SurfaceDice = sum(all_SurfaceDice) / len(all_SurfaceDice)

# 打印输出平均值
print()
print('Average MedSAM Dice:', avg_Dice)
# print('Average MedSAM SurfaceDice:', avg_SurfaceDice)
print()
