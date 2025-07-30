import argparse
from typing import Any
from large_spatial_model.utils.path_manager import init_all_submodules
from pathlib import Path
import torch

init_all_submodules()

from large_spatial_model.model import LSM_Dust3R  # noqa: E402
from large_spatial_model.utils.visualization_utils import render_pose  # noqa: E402


def eval_model_3d_ovs(
    model: Any, dataset_path: Path, eval_index_path: Path, output_path: Path
) -> None:
    context_images: list[torch.Tensor] = []
    # dummy target pose
    context_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))
    context_images.append(torch.zeros((3, 224, 224), dtype=torch.float32))

    target_extrinsics = torch.eye(4, dtype=torch.float32)
    target_intrinsics = torch.eye(3, dtype=torch.float32)
    image_rgb, image_seg = render_pose(
        context_images, target_intrinsics, target_extrinsics, model
    )


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
