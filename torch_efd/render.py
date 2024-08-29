import torch
from torch import Tensor

from .reconstruct import reconstruct_efds


def draw_mask(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    try:
        from skimage.draw import polygon
    except ImportError:
        raise ImportError("scikit-image is required for drawing mask")

    contour = reconstruct_efds(efds, n_points)

    # scale and offset contour to pixel coordinates
    scale = torch.tensor(shape, dtype=torch.float).sub_(1).div_(2)
    contour = contour.add_(1).mul_(scale)

    # draw mask
    mask = torch.zeros(shape, dtype=torch.uint8)
    mask[polygon(contour[:, 0], contour[:, 1], shape)] = 1
    return mask


def draw_perimeter(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    try:
        from skimage.draw import polygon_perimeter
    except ImportError:
        raise ImportError("scikit-image is required for drawing perimeter")

    contour = reconstruct_efds(efds, n_points)

    # scale and offset contour to pixel coordinates
    shape_t = torch.tensor(shape)
    contour = contour * shape_t / 2 + shape_t / 2

    # draw perimeter
    perimeter = polygon_perimeter(contour[:, 0], contour[:, 1], shape=shape, clip=True)
    img = torch.zeros(shape, dtype=torch.float64)
    img[perimeter] = 1

    return img


def compute_sdf(efds: Tensor, shape: tuple[int, int], n_points: int) -> Tensor:
    try:
        from scipy.ndimage import distance_transform_edt  # pyright: ignore[reportMissingTypeStubs]
    except ImportError:
        raise ImportError("scipy is required for computing SDF")

    mask = draw_mask(efds, shape, n_points).float().mul_(-2).add_(1)
    sampling = [2 / x for x in shape]
    sdf = distance_transform_edt(mask.numpy(), sampling)
    return torch.as_tensor(sdf, dtype=torch.float32)


def plot_efds(efds: Tensor, ts: int | Tensor = 100, ax=None, *args, **kwargs):  # pyright: ignore [reportUnknownParameterType, reportMissingParameterType]
    """
    Plot contours of the EFDs.
    :param efds: EFDs tensor of shape (n, m, 2)
    :param ts: time steps
    :param ax: Optional matplotlib axis
    :param args: additional arguments for matplotlib.pyplot.plot
    :param kwargs: additional keyword arguments for matplotlib.pyplot.plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        ax = plt.gca()

    coords = reconstruct_efds(efds, ts).cpu()
    return ax.plot(coords[..., 0], coords[..., 1], *args, **kwargs)
