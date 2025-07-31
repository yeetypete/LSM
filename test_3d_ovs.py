import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int, UInt, UInt8
from PIL import Image
from torch import Tensor
from torchvision import transforms as tf
from tqdm import tqdm

from geometry_3d_ovs import rescale_and_crop
from large_spatial_model.utils.path_manager import init_all_submodules
from metrics_3d_ovs import compute_lpips, compute_psnr, compute_ssim

init_all_submodules()

from large_spatial_model.model import LSM_Dust3R  # noqa: E402
from large_spatial_model.utils.visualization_utils import render_pose  # noqa: E402

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]
    prompts: list[str]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]
    masks: list[UInt8[Tensor, "..."]]
    features: list[Float[Tensor, "..."]]
    gt_masks: list[UInt8[Tensor, "..."]]
    gt_bboxes: list[Float[Tensor, "..."]]


def load_chunk(chunk_path: Path, data_index: Dict[str, str]) -> List[Example]:
    """Load a chunk file and filter by data index."""
    chunk = torch.load(chunk_path, weights_only=True)
    chunk = [example for example in chunk if example["key"] in data_index]
    return chunk


def load_index(index_path: Path) -> Dict[str, Any]:
    """Load the index.json file."""
    with open(index_path, "r") as f:
        index = json.load(f)
    return index


def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(
    images: list[UInt8[Tensor, "..."]],
) -> Float[Tensor, "batch 3 height width"]:
    torch_images: list[Float[Tensor, "3 height width"]] = []
    to_tensor = tf.ToTensor()
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(to_tensor(image))
    return torch.stack(torch_images)


def convert_masks(
    masks: list[UInt[Tensor, "..."]],
) -> UInt[Tensor, "batch 1 height width"]:
    return torch.stack(masks).unsqueeze(1)


def convert_gt_masks(
    masks: list[Bool[Tensor, "..."]],
) -> Bool[Tensor, "batch n_classes height width"]:
    return torch.stack(masks)


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def eval_model_3d_ovs(
    model: Any, dataset_path: Path, eval_index_path: Path, output_path: Path, resolution: int
) -> None:
    # Load the dataset index
    data_index = load_index(dataset_path / "index.json")

    # Load evaluation index to filter contex / target image pairs to evaluate
    eval_index = load_index(eval_index_path)

    # Get unique chunk names from the data index
    chunk_names = set(data_index.values())
    chunk_paths: list[Path] = sorted([dataset_path / f for f in chunk_names])

    # Process each chunk
    with tqdm(desc="Evaluating examples") as pbar:
        for chunk_path in sorted(chunk_paths):
            chunk: list[Example] = load_chunk(chunk_path, data_index)
            for example in chunk:
                if example["key"] not in eval_index:
                    continue
                context_indices = eval_index[example["key"]]["context"]
                assert len(context_indices) == 2, (
                    "LSM requires exactly two context images."
                )
                target_indices = eval_index[example["key"]]["target"]

                prompts = example["prompts"]
                images = convert_images(example["images"])
                gt_masks = convert_gt_masks(example["gt_masks"])
                extrinsics, intrinsics = convert_poses(example["cameras"])

                # Rescale and crop images and masks
                images, intrinsics = rescale_and_crop(
                    images,
                    intrinsics,
                    shape=(resolution, resolution),
                )

                gt_masks, _ = rescale_and_crop(
                    gt_masks,
                    intrinsics,
                    shape=(resolution, resolution),
                )

                for target_index in target_indices:
                    context_images = [
                        images[context_indices][0],
                        images[context_indices][1],
                    ]
                    target_image = images[target_index].cuda()
                    target_extrinsics = extrinsics[target_index]
                    context_extrinsics = extrinsics[context_indices]

                    # Run inference
                    pred_rgb, pred_segmentation = render_pose(
                        context_images,
                        context_extrinsics,
                        target_extrinsics,
                        model,
                        labelset=prompts,
                    )

                    save_image(
                        pred_rgb,
                        output_path / f"{example['key']}_pred_rgb_{target_index}.png",
                    )
                    save_image(
                        pred_segmentation,
                        output_path
                        / f"{example['key']}_pred_segmentation_{target_index}.png",
                    )

                    ssim = compute_ssim(
                        target_image.unsqueeze(0),
                        pred_rgb.unsqueeze(0),
                    ).item()

                    psnr = compute_psnr(
                        target_image.unsqueeze(0),
                        pred_rgb.unsqueeze(0),
                    ).item()

                    lpips = compute_lpips(
                        target_image.unsqueeze(0),
                        pred_rgb.unsqueeze(0),
                    ).item()

                    print(
                        f"Evaluating {example['key']} - "
                        f"SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, LPIPS: {lpips:.4f}"
                    )

                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--eval_index",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of the output images.",
    )

    args = parser.parse_args()
    data_path: Path = args.data_path
    eval_index: Path = args.eval_index
    output_path: Path = args.output_path
    resolution: int = args.resolution

    # 1. load model
    model = LSM_Dust3R.from_pretrained(args.model_path)
    model.eval()

    # 2. evaluate model on 3D OVS dataset
    eval_model_3d_ovs(
        model,
        dataset_path=data_path,
        eval_index_path=eval_index,
        output_path=output_path,
        resolution=resolution,
    )
