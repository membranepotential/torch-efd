import torch
from torch import Tensor
from skimage.draw import polygon2mask, polygon_perimeter

from .reconstruct import reconstruct_contours


def draw_mask(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    contour = reconstruct_contours(efds, n_points)

    # scale and offset contour to pixel coordinates
    shape_t = torch.tensor(shape)
    contour = contour * shape_t / 2 + shape_t / 2

    # delete out of bounds points
    out_of_bounds = ((contour < 0) | (contour > shape_t)).any(dim=-1)
    contour = contour[~out_of_bounds]

    # draw mask
    mask = polygon2mask(shape, contour)
    return torch.tensor(mask, dtype=torch.uint8)


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
