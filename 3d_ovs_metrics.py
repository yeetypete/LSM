import warnings
from functools import cache

import torch
from einops import rearrange, reduce
from jaxtyping import Bool, Float, UInt
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.nn import functional as F


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    assert isinstance(value, Tensor)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def compute_cosine_similarity(
    gt_mask: UInt[Tensor, "batch 1 height width"],
    gt_features: Float[Tensor, "batch n_feat d_feat"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "..."]:
    gt_mask = rearrange(gt_mask, "b 1 h w -> b h w")
    predicted = rearrange(predicted, "b c h w -> b h w c")
    b, h, w = gt_mask.nonzero(as_tuple=True)
    feature_idx = (gt_mask[b, h, w] - 1).long()

    gt_valid = gt_features[b, feature_idx, :]
    predicted_valid = predicted[b, h, w]

    similarity = F.cosine_similarity(gt_valid, predicted_valid)

    return similarity


@torch.no_grad()
def compute_per_prompt_iou(
    prompts: list[str],
    gt_mask: Bool[Tensor, "batch n_masks height width"],
    pred_masks: dict[str, Bool[Tensor, "batch 1 height width"]],
) -> dict[str, Float[Tensor, "..."]]:
    gt_masks = _unpack_gt_mask(gt_mask, prompts)
    ious = {}
    for prompt, pred_mask in pred_masks.items():
        if gt_masks[prompt].sum() == 0:
            warnings.warn(f"Prompt {prompt} has no ground truth mask", UserWarning)
        ious[prompt] = (
            compute_iou(gt_masks[prompt], pred_mask) * 100
        )  # average per-view IOUs

    return ious


@torch.no_grad()
def compute_iou(
    gt_mask: Bool[Tensor, "batch 1 height width"],
    predicted_mask: Bool[Tensor, "batch 1 height width"],
    eps: float = 1e-6,
) -> Float[Tensor, "..."]:
    intersection = reduce(gt_mask & predicted_mask, "b c h w -> b", "sum")
    union = reduce(gt_mask | predicted_mask, "b c h w -> b", "sum")
    iou = intersection.float() / (union.float() + eps)
    return iou


@torch.no_grad()
def _unpack_gt_mask(
    gt_mask: Bool[Tensor, "batch n_masks height width"],
    prompts: list[str],
) -> dict[str, Bool[Tensor, "batch 1 height width"]]:
    return {p: gt_mask[:, i].unsqueeze(1) for i, p in enumerate(prompts)}
