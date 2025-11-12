import torch  # 导入 PyTorch 库，用于张量计算
from PIL import Image  # 从 PIL 导入 Image，用于图像处理
import numpy as np  # 导入 NumPy，用于数值计算
from torch.utils.data import Dataset, DataLoader  # 导入 Dataset 和 DataLoader，用于处理数据集
from src.processor import Samprocessor  # 从 src 模块导入 Samprocessor，用于图像预处理
import src.utils as utils  # 从 src 模块导入工具函数
import os
join = os.path.join
import random  # 导入随机数生成库

class DatasetSegmentation(Dataset):

    def __init__(self, data_root, processor: Samprocessor):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]

        self.imgs = np.vstack([d['imgs'] for d in self.npz_data])
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])

        self.processor = processor

    def __len__(self):
        # 返回数据集中图像的总数量
        return self.ori_gts.shape[0]

    def __getitem__(self, index: int) -> list:
        # 获取指定索引对应的图像和掩膜
        img = self.imgs[index]  # 获取图像
        mask = self.ori_gts[index]  # 获取对应掩码
        mask = (mask > 0).astype(np.uint8)  # 将掩膜转换为二进制格式（0/1）

        gt2D = np.array(mask)  # 将掩码转换为 NumPy 数组
        original_size = tuple(img.shape)[:2]  # 获取图像的原始大小（高度，宽度）

        box = utils.get_bounding_box(gt2D)

        point = utils.get_point(gt2D)

        inputs = self.processor(img, original_size, box, point)
        inputs["ground_truth_mask"] = torch.from_numpy(gt2D)
        return inputs

def collate_fn(batch: torch.utils.data) -> list:
    """
    自定义 collate 函数，用于处理来自 DataLoader 的一批数据。

    参数：
        batch: 批处理数据集（字典列表）

    返回：
        (list): 包含批处理数据的字典列表。
    """
    return list(batch)  # 将批处理数据转换为列表并返回


