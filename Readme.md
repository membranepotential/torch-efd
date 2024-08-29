# Elliptic Fourier Descriptors in PyTorch

Elliptic Fourier descriptors (EFDs) [[1]](#1) represent a closed contour and are useful for shape analysis, matching, recognition and augmentation.

Adapted from [PyEFD](https://github.com/hbldh/pyefd).

## Install

```sh
pip install torch-efd
```

## <a id="example">Example</a>

```python
import torch
from torch_efd import (
    compute_efds,
    integrate_total_length,
    compute_curvature,
    rotate_efds,
    reconstruct_efds
)

# 2d contour of a square
contour = torch.tensor(
    [[1, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.float32,
)

# compute EFD coefficents
# higher order increases precision
# set normalize=True for scale/phase/rotation-invariant descriptors
efds = compute_efds(contour, order=20)

# compute length and curvature of shape
length = integrate_total_length(efds)
curvature = compute_curvature(efds, ts=100)

# transform shape
efds = rotate_efds(efds, angle=torch.pi / 4)

# back to coordinates
reconstructed = reconstruct_efds(efds, ts=100)
```

## API

### Basics

<table>
<tr>
<td style="width:50%;"> <b>Function</b> </td>
<td> <b>Description</b> </td>
</tr>

<tr>
<td>

```python
compute_efds(
    contour: Tensor,
    order: int,
    normalize: bool = False,
) -> Tensor
```

</td>
<td> Compute EFDs, see <a href="#example">example</a> </td>
</tr>

<tr>
<td>

```python
reconstruct_efds(
    efds: Tensor,
    ts: Tensor | int
) -> Tensor
```

</td>
<td> Reconstruct contour, see <a href="#example">example</a> </td>
</tr>

<tr>
<td>

```python
normalize_phase(efds: Tensor) -> Tensor
```

</td>
<td> Normalize the EFDs to have zero phase shift from the first major axis. </td>
</tr>

<tr>
<td>

```python
normalize_rotation(efds: Tensor) -> Tensor
```

</td>
<td> Normalize the EFDs to be rotation invariant by aligning the semi-major axis with the x-axis. </td>
</tr>

<tr>
<td>

```python
normalize_scale(efds: Tensor) -> Tensor
```

</td>
<td> Normalize the scale of the EFDs. </td>
</tr>

<tr>
<td>

```python
normalize_efds(efds: Tensor) -> Tensor
```

</td>
<td> Normalize phase, rotation and scale of EFDs. </td>
</tr>

</table>

### Geometry

<table>
<tr>
<td style="width:50%;"> <b>Function</b> </td>
<td> <b>Description</b> </td>
</tr>

<tr>
<td>

```python
derive_tangent_efds(efds: Tensor) -> Tensor
derive_normal_efds(efds: Tensor) -> Tensor
```

</td>
<td> Compute EFDs for tangent or normal function </td>
</tr>

<tr>
<td>

```python
reconstruct_tangents(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor

reconstruct_normals(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor

reconstruct_tangrads(
    efds: Tensor,
    ts: Tensor | int,
    normed: bool = False,
) -> Tensor
```

</td>
<td> Compute tangent/normal/tangrad vectors </td>
</tr>

<tr>
<td>

```python
compute_curvature(
    efds: Tensor,
    ts: Tensor | int,
    signed: bool = True,
) -> Tensor
```

</td>
<td> Compute (signed) curvature of a contour. </td>
</tr>

<tr>
<td>

```python
compute_speed(
    efds: Tensor,
    ts: Tensor | int
) -> Tensor

compute_arclens(
    efds: Tensor,
    ts: Tensor | int
) -> Tensor
```

</td>
<td> Compute speed (magnitude of tangents) or arc lengths </td>
</tr>

<tr>
<td>

```python
integrate_total_length(
    efds: Tensor,
    ts: Tensor | int = 100
) -> Tensor
```

</td>
<td> Compute total length of contour </td>
</tr>

<tr>
<td>

```python
compute_area(
    efds: Tensor,
    ts: Tensor | int = 100
) -> Tensor
```

</td>
<td> Compute area of polygon bounded by contour using the shoelace method </td>
</tr>

</table>

### Transform

<table>
<tr>
<td style="width:50%;"> <b>Function</b> </td>
<td> <b>Description</b> </td>
</tr>
<tr>
<td>

```python
scale_efds(
    efds: Tensor,
    scale: float | Tensor
) -> Tensor

rotate_efds(
    efds: Tensor,
    angle: float | Tensor
) -> Tensor
```

</td>
<td> Scale or rotate contour by transforming EFDs </td>
</tr>

<tr>
<td>

```python
smooth_efds(
    efds: Tensor,
    sigma: float,
    normalize: bool = False
) -> Tensor
```

</td>
<td> Apply gaussian smoothing with standard deviation <code>sigma</code> in fourier space </td>
</tr>
</table>

### Render

<table>
<tr>
<td style="width:50%;"> <b>Function</b> </td>
<td> <b>Description</b> </td>
</tr>
<tr>
<td>

```python
draw_mask(
    efds: Tensor,
    shape: tuple[int, int],
    n_points: int
) -> Tensor

draw_perimeter(
    efds: Tensor,
    shape: tuple[int, int],
    n_points: int
) -> Tensor
```

</td>
<td>
Draw image of mask/perimeter of the polygon bounded by the contour.
Requires <code>scikit-image</code>. </td>
</tr>

<tr>
<td>

```python
compute_sdf(
    efds: Tensor,
    shape: tuple[int, int],
    n_points: int
) -> Tensor
```

</td>
<td>
Compute signed distance function for a contour.
Requires <code>scipy</code>. </td>
</tr>

<tr>
<td>

```python
plot_efds(
    efds: Tensor,
    ts: int | Tensor = 100,
    ax: Axes | None = None,
    *args,
    **kwargs,
)
```

</td>
<td>
Plot a contour. Extra args/kwargs will be passed to <code>Axes.plot</code>.
Requires <code>matplotlib</code>. </td>
</tr>
</table>

## References

<a id="1">[1]</a>
F. Kuhl and C. P. Giardina.
Elliptic Fourier Features of a Closed Contour.
Computer Graphics and Image Processing 18, 1982, 236-258.
https://www.sci.utah.edu/~gerig/CS7960-S2010/handouts/Kuhl-Giardina-CGIP1982.pdf
