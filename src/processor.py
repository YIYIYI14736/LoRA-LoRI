from src.segment_anything.utils.transforms import ResizeLongestSide  # 导入 ResizeLongestSide，用于调整图像大小
from src.lora import LoRA_sam  # 导入 LoRA_sam 模型
import numpy as np  # 导入 NumPy，用于数值计算
import torch  # 导入 PyTorch，用于深度学习计算
import PIL  # 导入 PIL，用于图像处理
from typing import Optional, Tuple  # 导入可选和元组类型提示

class Samprocessor:
    def __init__(self, sam_model: LoRA_sam):

        super().__init__()  # 初始化父类
        self.model = sam_model  # 保存传入的 SAM 模型实例
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)  # 初始化图像变换器
        self.reset_image()  # 重置图像状态

    def __call__(self, image: np, original_size: tuple, box: list, points: list) -> dict:
            # 处理图像和边界框提示
        image_torch = self.process_image(image, original_size)  # 处理图像

            # 转换输入提示
        box_torch = self.process_prompt(box, original_size)  # 处理边界框提示

            # 处理点提示（如果存在）
        point_torch = self.process_points(points, original_size) if points is not None else None

            # 构建输入字典
        inputs = {
                "image": image_torch,  # 添加处理后的图像
                "original_size": original_size,  # 添加原始图像大小
                "boxes": box_torch,  # 添加处理后的边界框
                "points": point_torch  # 添加点提示
        }

        return inputs  # 返回输入字典

    def process_image(self, image: np, original_size: tuple) -> torch.tensor:

        nd_image = np.array(image)  # 将图像转换为 NumPy 数组
        input_image = self.transform.apply_image(nd_image)  # 应用图像变换
        input_image_torch = torch.as_tensor(input_image, device=self.device)  # 转换为张量并移动到相应设备
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  # 调整维度并增加批次维度
        return input_image_torch  # 返回处理后的图像张量

    def process_prompt(self, box: list, original_size: tuple) -> torch.tensor:

        box_torch = None  # 初始化边界框张量
        # 设置框提示
        box_np = torch.tensor(box).float().numpy()
        sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, original_size=original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)

        return box_torch  # 返回处理后的边界框张量

    def process_points(self, points: list, original_size: tuple) -> torch.tensor:

        points_torch = None  # 初始化点提示张量

        points = torch.tensor(points).to(self.device)
        original_image_size = (256, 256)  # 手动设置每个切片的size(不太智能)
        transform = ResizeLongestSide(self.model.image_encoder.img_size)

        point_coords = transform.apply_coords_torch(points, original_image_size)
        num_points = point_coords.shape[0]
        label = torch.full((num_points, 5), 1, dtype=torch.float, device=self.device)
        point_torch = (point_coords, label)

        return point_torch  # 返回处理后的点提示张量

    @property
    def device(self) -> torch.device:
            # 返回模型所在的设备（CPU 或 GPU）
        return self.model.device

    def reset_image(self) -> None:
        """重置当前设置的图像。"""
        self.is_image_set = False  # 标记图像未设置
        self.features = None  # 重置特征
        self.orig_h = None  # 重置原始高度
        self.orig_w = None  # 重置原始宽度
        self.input_h = None  # 重置输入高度
        self.input_w = None  # 重置输入宽度


