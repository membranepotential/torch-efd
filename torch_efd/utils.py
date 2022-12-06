import torch
from torch import Tensor
from skimage.draw import polygon, polygon_perimeter

from .reconstruct import reconstruct_contours


def draw_mask(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    contour = reconstruct_contours(efds, n_points)

    # scale and offset contour to pixel coordinates
    contour = contour.add_(1.0).mul_(torch.tensor(shape).sub_(1).div(2))

    # draw mask
    mask = torch.zeros(shape, dtype=torch.uint8)
    mask[polygon(contour[:, 0], contour[:, 1], shape)] = 1
    return mask


def draw_perimeter(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    contour = reconstruct_contours(efds, n_points)

    # scale and offset contour to pixel coordinates
    shape_t = torch.tensor(shape)
    contour = contour * shape_t / 2 + shape_t / 2

    # draw perimeter
    perimeter = polygon_perimeter(contour[:, 0], contour[:, 1], shape=shape, clip=True)
    img = torch.zeros(shape, dtype=torch.float64)
    img[perimeter] = 1

    return img
