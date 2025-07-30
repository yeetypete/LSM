import torch
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from torch import Tensor


def rescale(
    image: Shaped[Tensor, "dim h_in w_in"],
    shape: tuple[int, int],
) -> Shaped[Tensor, "dim h_out w_out"]:
    h_out, w_out = shape
    image = image.unsqueeze(0)

    mode = "bilinear" if torch.is_floating_point(image) else "nearest"

    if image.dtype == torch.bool:
        image_uint = image.to(torch.uint8)
        image_out: Tensor = F.interpolate(
            image_uint,
            size=(h_out, w_out),
            mode=mode,
            align_corners=True if mode == "bilinear" else None,
        )
        return image_out.to(torch.bool).squeeze(0)

    image_out: Tensor = F.interpolate(
        image,
        size=(h_out, w_out),
        mode=mode,
        align_corners=True if mode == "bilinear" else None,
    )
    return image_out.squeeze(0)


def center_crop(
    images: Shaped[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> (
    tuple[
        Shaped[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Shaped[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> (
    tuple[
        Shaped[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)


    return center_crop(images, intrinsics, shape)

