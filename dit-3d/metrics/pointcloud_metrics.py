"""Point-cloud generation metrics.

This module implements the PointFlow/PVD-style set metrics:

* COV and MMD use the sample-vs-reference pairwise distance matrix.
* 1-NNA is a leave-one-out 1-nearest-neighbor classifier over
  reference + generated point-cloud sets.

Inputs are point clouds shaped ``(num_clouds, num_points, 3)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch


MetricName = Literal["cd", "emd"]

_CHAMFER = None
_CHAMFER_UNAVAILABLE = False


def _normalize_metric_names(metrics: Iterable[str] | str) -> tuple[MetricName, ...]:
    if isinstance(metrics, str):
        if metrics == "both":
            return ("cd", "emd")
        metrics = (metrics,)
    names: list[MetricName] = []
    for metric in metrics:
        name = metric.lower()
        if name not in {"cd", "emd"}:
            raise ValueError(f"unsupported metric {metric!r}; expected cd, emd, or both")
        names.append(name)  # type: ignore[arg-type]
    return tuple(dict.fromkeys(names))


def _validate_pcs(name: str, pcs: torch.Tensor) -> torch.Tensor:
    pcs = torch.as_tensor(pcs, dtype=torch.float32)
    if pcs.ndim != 3 or pcs.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (num_clouds, num_points, 3), got {tuple(pcs.shape)}")
    if pcs.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    return pcs.contiguous()


def _get_chamfer():
    global _CHAMFER, _CHAMFER_UNAVAILABLE
    if _CHAMFER_UNAVAILABLE:
        return None
    if _CHAMFER is None:
        try:
            from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        except Exception as exc:
            _CHAMFER_UNAVAILABLE = True
            print(
                f"[pointcloud-metrics] CUDA Chamfer unavailable ({exc!r}); falling back to torch.cdist",
                flush=True,
            )
            return None

        try:
            _CHAMFER = chamfer_3DDist()
        except Exception as exc:
            _CHAMFER_UNAVAILABLE = True
            print(
                f"[pointcloud-metrics] CUDA Chamfer init failed ({exc!r}); falling back to torch.cdist",
                flush=True,
            )
            return None
    return _CHAMFER


def _cd_pairs_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Match the CUDA Chamfer implementation convention: squared L2 distances,
    # averaged in both directions and summed.
    d = torch.cdist(x, y, p=2).pow_(2)
    return d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1)


def _cd_pairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and y.is_cuda:
        chamfer = _get_chamfer()
        if chamfer is not None:
            dl, dr, _, _ = chamfer(x, y)
            return dl.mean(dim=1) + dr.mean(dim=1)
    return _cd_pairs_torch(x, y)


def _emd_pairs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (x.is_cuda and y.is_cuda):
        raise RuntimeError("EMD evaluation requires CUDA tensors")
    from metrics.PyTorchEMD.emd import earth_mover_distance

    return earth_mover_distance(x, y, transpose=False)


@torch.no_grad()
def pairwise_distances(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    *,
    metric: MetricName,
    ref_batch_size: int = 64,
    sample_batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute a ``num_sample x num_ref`` pairwise distance matrix."""

    sample_pcs = _validate_pcs("sample_pcs", sample_pcs).cpu()
    ref_pcs = _validate_pcs("ref_pcs", ref_pcs).cpu()
    if sample_pcs.shape[1:] != ref_pcs.shape[1:]:
        raise ValueError(
            "sample_pcs/ref_pcs point shapes differ: "
            f"{tuple(sample_pcs.shape[1:])} vs {tuple(ref_pcs.shape[1:])}"
        )
    if ref_batch_size <= 0 or sample_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    metric = metric.lower()  # type: ignore[assignment]
    if metric not in {"cd", "emd"}:
        raise ValueError(f"unsupported metric {metric!r}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if metric == "emd" and device.type != "cuda":
        raise RuntimeError("EMD evaluation requires a CUDA device")

    n_sample, n_ref = sample_pcs.shape[0], ref_pcs.shape[0]
    out = torch.empty(n_sample, n_ref, dtype=torch.float32)
    progress_every = max(1, n_sample // 10)

    for i0 in range(0, n_sample, sample_batch_size):
        i1 = min(n_sample, i0 + sample_batch_size)
        sample_block = sample_pcs[i0:i1].to(device, non_blocking=True)
        s_count = i1 - i0
        for j0 in range(0, n_ref, ref_batch_size):
            j1 = min(n_ref, j0 + ref_batch_size)
            ref_block = ref_pcs[j0:j1].to(device, non_blocking=True)
            r_count = j1 - j0

            sample_pairs = (
                sample_block[:, None]
                .expand(s_count, r_count, -1, -1)
                .reshape(s_count * r_count, sample_block.shape[1], 3)
                .contiguous()
            )
            ref_pairs = (
                ref_block[None]
                .expand(s_count, r_count, -1, -1)
                .reshape(s_count * r_count, ref_block.shape[1], 3)
                .contiguous()
            )

            if metric == "cd":
                dist = _cd_pairs(sample_pairs, ref_pairs)
            else:
                dist = _emd_pairs(sample_pairs, ref_pairs)
            out[i0:i1, j0:j1] = dist.reshape(s_count, r_count).detach().cpu()

            del ref_block, sample_pairs, ref_pairs, dist
        del sample_block
        if i1 == n_sample or i0 % progress_every == 0:
            print(
                f"[pointcloud-metrics] {metric.upper()} pairwise rows {i1}/{n_sample}",
                flush=True,
            )

    return out


def lgan_mmd_cov(dist_sample_ref: torch.Tensor) -> dict[str, float]:
    """LGAN MMD/COV from a ``num_sample x num_ref`` distance matrix."""

    if dist_sample_ref.ndim != 2:
        raise ValueError(f"distance matrix must be 2D, got {tuple(dist_sample_ref.shape)}")
    n_ref = dist_sample_ref.shape[1]
    sample_to_ref_dist, sample_to_ref_idx = dist_sample_ref.min(dim=1)
    ref_to_sample_dist, _ = dist_sample_ref.min(dim=0)
    return {
        "mmd": float(ref_to_sample_dist.mean().item()),
        "mmd_smp": float(sample_to_ref_dist.mean().item()),
        "cov": float(sample_to_ref_idx.unique().numel()) / float(n_ref),
    }


def one_nn_accuracy(
    dist_ref_ref: torch.Tensor,
    dist_sample_ref: torch.Tensor,
    dist_sample_sample: torch.Tensor,
) -> dict[str, float]:
    """Leave-one-out 1-NN accuracy over reference and generated sets."""

    n_ref = dist_ref_ref.shape[0]
    n_sample = dist_sample_sample.shape[0]
    if dist_ref_ref.shape != (n_ref, n_ref):
        raise ValueError("dist_ref_ref must be square")
    if dist_sample_sample.shape != (n_sample, n_sample):
        raise ValueError("dist_sample_sample must be square")
    if dist_sample_ref.shape != (n_sample, n_ref):
        raise ValueError(
            "dist_sample_ref must have shape "
            f"({n_sample}, {n_ref}), got {tuple(dist_sample_ref.shape)}"
        )

    matrix = torch.cat(
        (
            torch.cat((dist_ref_ref, dist_sample_ref.t()), dim=1),
            torch.cat((dist_sample_ref, dist_sample_sample), dim=1),
        ),
        dim=0,
    ).clone()
    matrix.fill_diagonal_(float("inf"))

    labels = torch.cat(
        (
            torch.ones(n_ref, dtype=torch.bool),
            torch.zeros(n_sample, dtype=torch.bool),
        )
    )
    pred = labels[matrix.argmin(dim=1)]
    ref_mask = labels
    sample_mask = ~labels
    return {
        "acc": float((pred == labels).float().mean().item()),
        "acc_ref": float((pred[ref_mask] == labels[ref_mask]).float().mean().item()),
        "acc_sample": float((pred[sample_mask] == labels[sample_mask]).float().mean().item()),
    }


@torch.no_grad()
def compute_all_metrics(
    sample_pcs: torch.Tensor,
    ref_pcs: torch.Tensor,
    *,
    metrics: Iterable[str] | str = "both",
    ref_batch_size: int = 64,
    sample_batch_size: int = 1,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Compute 1-NNA, COV, and MMD metrics for complete point-cloud sets."""

    sample_pcs = _validate_pcs("sample_pcs", sample_pcs)
    ref_pcs = _validate_pcs("ref_pcs", ref_pcs)
    metric_names = _normalize_metric_names(metrics)

    results: dict[str, float] = {
        "num_samples": float(sample_pcs.shape[0]),
        "num_refs": float(ref_pcs.shape[0]),
        "num_points": float(sample_pcs.shape[1]),
    }
    for metric in metric_names:
        suffix = metric.upper()
        print(f"[pointcloud-metrics] computing {suffix} sample-reference distances", flush=True)
        dist_sample_ref = pairwise_distances(
            sample_pcs,
            ref_pcs,
            metric=metric,
            ref_batch_size=ref_batch_size,
            sample_batch_size=sample_batch_size,
            device=device,
        )
        lgan = lgan_mmd_cov(dist_sample_ref)
        results[f"MMD-{suffix}"] = lgan["mmd"]
        results[f"MMD-SMP-{suffix}"] = lgan["mmd_smp"]
        results[f"COV-{suffix}"] = lgan["cov"]

        print(f"[pointcloud-metrics] computing {suffix} reference-reference distances", flush=True)
        dist_ref_ref = pairwise_distances(
            ref_pcs,
            ref_pcs,
            metric=metric,
            ref_batch_size=ref_batch_size,
            sample_batch_size=sample_batch_size,
            device=device,
        )
        print(f"[pointcloud-metrics] computing {suffix} sample-sample distances", flush=True)
        dist_sample_sample = pairwise_distances(
            sample_pcs,
            sample_pcs,
            metric=metric,
            ref_batch_size=ref_batch_size,
            sample_batch_size=sample_batch_size,
            device=device,
        )
        one_nn = one_nn_accuracy(dist_ref_ref, dist_sample_ref, dist_sample_sample)
        results[f"1-NNA-{suffix}"] = one_nn["acc"]
        results[f"1-NNA-{suffix}-ref"] = one_nn["acc_ref"]
        results[f"1-NNA-{suffix}-sample"] = one_nn["acc_sample"]

    return results
