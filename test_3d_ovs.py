import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int, UInt, UInt8
from PIL import Image
from torch import Tensor
from torchvision import transforms as tf
from tqdm import tqdm

from large_spatial_model.utils.path_manager import init_all_submodules

init_all_submodules()

from large_spatial_model.model import LSM_Dust3R  # noqa: E402


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


def load_index(index_path: Path) -> Dict[str, str]:
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


def eval_model_3d_ovs(
    model: Any, dataset_path: Path, eval_index_path: Path, output_path: Path
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
            # Filter chunk by eval index

            for example in chunk:
                images = convert_images(example["images"])
                masks = convert_masks(example["masks"])
                gt_masks = convert_gt_masks(example["gt_masks"])
                extrinsics, intrinsics = convert_poses(example["cameras"])
                pbar.update(1)

        #     # Run inference
        #         try:
        #             image_rgb, image_seg = render_pose(
        #                 context_images, target_intrinsics, target_extrinsics, model
        #             )

        #         # Save results or compute metrics here
        #         # TODO: Implement result saving/metric computation

        #     except Exception as e:
        #         print(f"Error processing example {example['key']}: {e}")


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

    args = parser.parse_args()
    data_path: Path = args.data_path
    eval_index: Path = args.eval_index
    output_path: Path = args.output_path

    # 1. load model
    model = LSM_Dust3R.from_pretrained(args.model_path)
    model.eval()

    # 2. evaluate model on 3D OVS dataset
    eval_model_3d_ovs(
        model,
        dataset_path=data_path,
        eval_index_path=eval_index,
        output_path=output_path,
    )
