import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch
from torch.nn.functional import pad


def show_mask(mask: np.array, ax, random_color=False):
    """
    Plot the mask

    Arguments:
        mask: Array of the binary mask (or float)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """
    Plot the image and the mask superposed

    Arguments:
        image: PIL original image
        mask: PIL original binary mask
    """
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()
    

def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """
    Take an image from the dataset and plot it

    Arguments:
        dataset: Dataset class loaded with our images
        idx: Index of the data we want
    """
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def get_bounding_box(gt2D: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # 获取框
  y_indices, x_indices = np.where(gt2D > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = gt2D.shape
  x_min = max(0, x_min - np.random.randint(0, 5))
  x_max = min(W, x_max + np.random.randint(0, 5))
  y_min = max(0, y_min - np.random.randint(0, 5))
  y_max = min(H, y_max + np.random.randint(0, 5))
  bboxes = np.array([x_min, y_min, x_max, y_max])

  return bboxes

def get_point(gt2D: np.array) -> list:
    if gt2D.sum() == 0:

        n = 5  # 你想要[0, 0]的个数
        selected_points = [[0, 0] for _ in range(n)]
        print(selected_points)

    else:
        ########### 随机打点 #################
        y_indices, x_indices = np.where(gt2D == 1)
        points = np.column_stack((x_indices, y_indices))

        selected_indices = np.random.choice(len(points), size=5, replace=True)
        selected_points = points[selected_indices]
        selected_points = selected_points.tolist()

    return selected_points


def stacking_batch(batch, outputs):
    """
    Given the batch and outputs of SAM, stacks the tensors to compute the loss. We stack by adding another dimension.

    Arguments:
        batch(list(dict)): List of dict with the keys given in the dataset file
        outputs: list(dict): List of dict that are the outputs of SAM
    
    Return: 
        stk_gt: Stacked tensor of the ground truth masks in the batch. Shape: [batch_size, H, W] -> We will need to add the channels dimension (dim=1)
        stk_out: Stacked tensor of logits mask outputed by SAM. Shape: [batch_size, 1, 1, H, W] -> We will need to remove the extra dimension (dim=1) needed by SAM 
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)
        
    return stk_gt, stk_out