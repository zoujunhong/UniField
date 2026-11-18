from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.cache.high_speed_train import (
    HIGH_SPEED_TRAIN_ROOT,
    HIGH_SPEED_TRAIN_CACHE_ROOT,
    complete_high_speed_raw_case,
    high_speed_case_sort_key,
    normalized_train_type,
    auxiliary_cache_path,
    register_auxiliary_cache,
    shared_cache_dir,
)
from dataset.cache.tripnet_high_speed import (
    _bounds_tuple_to_array,
    _load_centered_surface,
    _sample_occupancy,
)
from dataset.utils.high_speed_train_dataset import _split_high_speed_names_by_geometry
from dataset.utils.load import high_speed_type_dir, list_high_speed_cases
from dataset.utils.sampling import filter_names
from model.utils.gaussian3d import atomic_save_npz, gaussian_rbf_aggregate, normalize_quaternion


DEFAULT_GAUSSIAN3D_COUNT = 4096
DEFAULT_GAUSSIAN3D_CHANNELS = 32
DEFAULT_GAUSSIAN3D_K = 32


def gaussian3d_cache_dir(
    cache_root: str | Path,
    train_type: str,
    num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
    channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
    knn: int = DEFAULT_GAUSSIAN3D_K,
) -> str:
    name = f"gaussian3d_M{int(num_gaussians)}_C{int(channels)}_K{int(knn)}_fp32"
    return shared_cache_dir(cache_root, train_type, name)


def gaussian3d_cache_path(
    cache_root: str | Path,
    train_type: str,
    case_name: str,
    num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
    channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
    knn: int = DEFAULT_GAUSSIAN3D_K,
) -> str:
    profile = f"gaussian3d_M{int(num_gaussians)}_C{int(channels)}_K{int(knn)}_fp32"
    return auxiliary_cache_path(cache_root, case_name, profile, ".npz")


def shared_decoder_path(
    cache_root: str | Path,
    train_type: str,
    num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
    channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
    knn: int = DEFAULT_GAUSSIAN3D_K,
) -> str:
    return os.path.join(gaussian3d_cache_dir(cache_root, train_type, num_gaussians, channels, knn), "_shared_occupancy_decoder.pt")


def valid_gaussian3d_cache(
    path: str | Path,
    num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
    channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        data = np.load(path)
        checks = [
            ("means", (int(num_gaussians), 3)),
            ("log_scales", (int(num_gaussians), 3)),
            ("rotations", (int(num_gaussians), 4)),
            ("opacity_logits", (int(num_gaussians), 1)),
            ("features", (int(num_gaussians), int(channels))),
            ("bounds", (2, 3)),
        ]
        for key, shape in checks:
            value = data[key]
            if value.shape != shape or value.dtype != np.float32:
                return False
            if not np.all(np.isfinite(value)):
                return False
        if np.asarray(data["x_center"]).size != 1:
            return False
        return True
    except Exception:
        return False


class GaussianOccupancyDecoder(nn.Module):
    def __init__(self, channels: int = DEFAULT_GAUSSIAN3D_CHANNELS, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(channels)
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.GELU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _init_gaussian_parameters(
    bounds: np.ndarray,
    num_gaussians: int,
    channels: int,
    seed: int,
    device: torch.device,
) -> dict[str, nn.Parameter]:
    bounds_t = torch.from_numpy(bounds).to(device=device, dtype=torch.float32)
    lower = bounds_t[0]
    upper = bounds_t[1]
    span = (upper - lower).clamp_min(1e-6)
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=int(seed))
    samples = sobol.draw(int(num_gaussians)).to(device=device, dtype=torch.float32)
    means = lower + samples * span

    grid = max(round(float(num_gaussians) ** (1.0 / 3.0)), 1)
    init_scale = (span / float(grid * 2)).clamp_min(1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 17)
    log_scales = init_scale.log().view(1, 3).expand(int(num_gaussians), -1).clone()
    log_scales = log_scales + 0.05 * torch.randn(log_scales.shape, generator=generator, device=device)

    rotations = torch.zeros(int(num_gaussians), 4, device=device)
    rotations[:, 0] = 1.0
    opacity_logits = torch.zeros(int(num_gaussians), 1, device=device)
    features = 0.02 * torch.randn(int(num_gaussians), int(channels), generator=generator, device=device)
    return {
        "means": nn.Parameter(means),
        "log_scales": nn.Parameter(log_scales),
        "rotations": nn.Parameter(rotations),
        "opacity_logits": nn.Parameter(opacity_logits),
        "features": nn.Parameter(features),
    }


def _batched_params(params: dict[str, nn.Parameter]) -> tuple[torch.Tensor, ...]:
    rotations = normalize_quaternion(params["rotations"])
    return (
        params["means"].unsqueeze(0),
        params["log_scales"].unsqueeze(0),
        rotations.unsqueeze(0),
        params["opacity_logits"].unsqueeze(0),
        params["features"].unsqueeze(0),
    )


def _gaussian_regularization(
    params: dict[str, nn.Parameter],
    bounds_t: torch.Tensor,
    *,
    mean_bound_weight: float,
    scale_bound_weight: float,
    norm_weight: float,
) -> torch.Tensor:
    means = params["means"]
    lower = bounds_t[0]
    upper = bounds_t[1]
    outside = F.relu(lower - means).square().mean() + F.relu(means - upper).square().mean()
    log_scales = params["log_scales"]
    scale_bounds = F.relu(-8.0 - log_scales).square().mean() + F.relu(log_scales - 4.0).square().mean()
    norm = (
        params["features"].square().mean()
        + 0.01 * params["opacity_logits"].square().mean()
        + 0.01 * params["rotations"][..., 1:].square().mean()
    )
    return (
        float(mean_bound_weight) * outside
        + float(scale_bound_weight) * scale_bounds
        + float(norm_weight) * norm
    )


def _select_occupancy_batch(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    batch_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(batch_points) <= 0 or int(batch_points) >= points.shape[0]:
        return points, labels
    idx = rng.choice(points.shape[0], size=int(batch_points), replace=False)
    return points[idx], labels[idx]


def _evaluate_occupancy_loss(
    decoder: GaussianOccupancyDecoder,
    params: dict[str, nn.Parameter],
    query: torch.Tensor,
    target: torch.Tensor,
    *,
    knn: int,
    occupancy_query_chunk: int,
) -> torch.Tensor:
    means, log_scales, rotations, opacity_logits, features = _batched_params(params)
    aggregated = gaussian_rbf_aggregate(
        query,
        means,
        log_scales,
        rotations,
        opacity_logits,
        features,
        k=int(knn),
        max_query_chunk=int(occupancy_query_chunk),
    )
    logits = decoder(aggregated)
    return F.binary_cross_entropy_with_logits(logits, target)


def fit_case_gaussian3d(
    case_dir: str | Path,
    decoder: GaussianOccupancyDecoder,
    *,
    num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
    channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
    knn: int = DEFAULT_GAUSSIAN3D_K,
    epochs: int = 70,
    lr: float = 1e-3,
    uniform_points: int = 16_384,
    near_surface_points: int = 16_384,
    occupancy_batch_points: int = 8_192,
    occupancy_query_chunk: int = 8_192,
    near_surface_sigma: float = 0.01,
    bounds_padding: float = 0.05,
    mean_bound_weight: float = 1e-2,
    scale_bound_weight: float = 1e-4,
    norm_weight: float = 1e-6,
    seed: int = 0,
    device: str | torch.device | None = None,
    resample_each_epoch: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    surface, x_center = _load_centered_surface(case_dir)
    bounds = _bounds_tuple_to_array(surface.bounds, padding=bounds_padding)
    rng = np.random.default_rng(int(seed))
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoder = decoder.to(device_t).float().eval()
    for param in decoder.parameters():
        param.requires_grad_(False)

    params = _init_gaussian_parameters(bounds, num_gaussians, channels, seed, device_t)
    bounds_t = torch.from_numpy(bounds).to(device_t)
    optimizer = torch.optim.Adam(list(params.values()), lr=float(lr))

    fixed_points = None
    fixed_labels = None
    if not resample_each_epoch:
        fixed_points, fixed_labels = _sample_occupancy(
            surface,
            bounds,
            rng,
            uniform_points=uniform_points,
            near_surface_points=near_surface_points,
            near_surface_sigma=near_surface_sigma,
        )

    last_loss = 0.0
    for _ in range(int(epochs)):
        if resample_each_epoch:
            points, labels = _sample_occupancy(
                surface,
                bounds,
                rng,
                uniform_points=uniform_points,
                near_surface_points=near_surface_points,
                near_surface_sigma=near_surface_sigma,
            )
        else:
            points, labels = fixed_points, fixed_labels
        points, labels = _select_occupancy_batch(points, labels, rng, occupancy_batch_points)
        query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
        target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
        loss = _evaluate_occupancy_loss(
            decoder,
            params,
            query,
            target,
            knn=knn,
            occupancy_query_chunk=occupancy_query_chunk,
        )
        loss = loss + _gaussian_regularization(
            params,
            bounds_t,
            mean_bound_weight=mean_bound_weight,
            scale_bound_weight=scale_bound_weight,
            norm_weight=norm_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    arrays = {
        "means": params["means"].detach().cpu().numpy().astype(np.float32),
        "log_scales": params["log_scales"].detach().cpu().numpy().astype(np.float32),
        "rotations": normalize_quaternion(params["rotations"]).detach().cpu().numpy().astype(np.float32),
        "opacity_logits": params["opacity_logits"].detach().cpu().numpy().astype(np.float32),
        "features": params["features"].detach().cpu().numpy().astype(np.float32),
    }
    return arrays, bounds.astype(np.float32), float(x_center), last_loss


def identity_collate(batch):
    return batch


class HighSpeedTrainGaussian3DCacheBuilder(Dataset):
    """Build per-case 3D Gaussian caches for high_speed_train cases."""

    def __init__(
        self,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
        train_type: str = "CRH450",
        cache_root: str | None = None,
        num_gaussians: int = DEFAULT_GAUSSIAN3D_COUNT,
        channels: int = DEFAULT_GAUSSIAN3D_CHANNELS,
        knn: int = DEFAULT_GAUSSIAN3D_K,
        ids_file: str | None = None,
        split: str = "all",
        val_ratio: float = 0.1,
        geometry_count: int | None = None,
        overwrite: bool = False,
        skip_incomplete: bool = True,
        stage1_epochs: int = 500,
        stage1_case_limit: int | None = 32,
        stage2_epochs: int = 70,
        lr: float = 1e-3,
        uniform_points: int = 16_384,
        near_surface_points: int = 16_384,
        occupancy_batch_points: int = 8_192,
        occupancy_query_chunk: int = 8_192,
        near_surface_sigma: float = 0.01,
        bounds_padding: float = 0.05,
        decoder_hidden_dim: int = 128,
        decoder_depth: int = 3,
        cache_stage1_surfaces: bool = True,
        resample_stage1_each_epoch: bool = False,
        resample_stage2_each_epoch: bool = False,
        mean_bound_weight: float = 1e-2,
        scale_bound_weight: float = 1e-4,
        norm_weight: float = 1e-6,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or HIGH_SPEED_TRAIN_CACHE_ROOT
        self.num_gaussians = int(num_gaussians)
        self.channels = int(channels)
        self.knn = int(knn)
        self.overwrite = bool(overwrite)
        self.stage1_epochs = int(stage1_epochs)
        self.stage1_case_limit = None if stage1_case_limit is None else int(stage1_case_limit)
        self.stage2_epochs = int(stage2_epochs)
        self.lr = float(lr)
        self.uniform_points = int(uniform_points)
        self.near_surface_points = int(near_surface_points)
        self.occupancy_batch_points = int(occupancy_batch_points)
        self.occupancy_query_chunk = int(occupancy_query_chunk)
        self.near_surface_sigma = float(near_surface_sigma)
        self.bounds_padding = float(bounds_padding)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_depth = int(decoder_depth)
        self.cache_stage1_surfaces = bool(cache_stage1_surfaces)
        self.resample_stage1_each_epoch = bool(resample_stage1_each_epoch)
        self.resample_stage2_each_epoch = bool(resample_stage2_each_epoch)
        self.mean_bound_weight = float(mean_bound_weight)
        self.scale_bound_weight = float(scale_bound_weight)
        self.norm_weight = float(norm_weight)
        self.seed = int(seed)
        self.device = device

        case_dirs = sorted(list_high_speed_cases(self.root_dir, self.train_type), key=high_speed_case_sort_key)
        names = filter_names([path.name for path in case_dirs], ids_file)
        names = _split_high_speed_names_by_geometry(
            names,
            self.train_type,
            split,
            float(val_ratio),
            geometry_count,
        )
        type_dir = high_speed_type_dir(self.root_dir, self.train_type)
        self.case_dirs = [type_dir / name for name in names]
        if skip_incomplete:
            before = len(self.case_dirs)
            self.case_dirs = [
                path
                for path in self.case_dirs
                if complete_high_speed_raw_case(path, build_surface=True, build_volume=False)
            ]
            skipped = before - len(self.case_dirs)
            if skipped > 0:
                print(f"Gaussian3D cache builder: skipped {skipped} incomplete cases.", flush=True)
        os.makedirs(gaussian3d_cache_dir(self.cache_root, self.train_type, self.num_gaussians, self.channels, self.knn), exist_ok=True)
        print(
            "Gaussian3D cache builder: "
            f"{self.train_type}, {len(self.case_dirs)} cases, "
            f"M={self.num_gaussians}, C={self.channels}, K={self.knn}.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.case_dirs)

    def _decoder(self) -> GaussianOccupancyDecoder:
        return GaussianOccupancyDecoder(
            channels=self.channels,
            hidden_dim=self.decoder_hidden_dim,
            depth=self.decoder_depth,
        )

    @property
    def decoder_path(self) -> str:
        return shared_decoder_path(self.cache_root, self.train_type, self.num_gaussians, self.channels, self.knn)

    def fit_shared_decoder(self, indices: Sequence[int] | None = None) -> str:
        if indices is None:
            indices = list(range(len(self.case_dirs)))
            if self.stage1_case_limit is not None:
                indices = indices[: self.stage1_case_limit]
        if not indices:
            raise RuntimeError("No cases available for shared occupancy decoder fitting.")

        rng = np.random.default_rng(self.seed)
        device_t = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        decoder = self._decoder().to(device_t).float()
        case_states = []
        for local_idx, idx in enumerate(indices):
            case_dir = self.case_dirs[int(idx)]
            surface, _ = _load_centered_surface(case_dir)
            bounds = _bounds_tuple_to_array(surface.bounds, padding=self.bounds_padding)
            params = _init_gaussian_parameters(bounds, self.num_gaussians, self.channels, self.seed + local_idx, device_t)
            if self.resample_stage1_each_epoch:
                cached_surface = surface if self.cache_stage1_surfaces else None
                cached_bounds = bounds if self.cache_stage1_surfaces else None
                points = None
                labels = None
            else:
                points, labels = _sample_occupancy(
                    surface,
                    bounds,
                    rng,
                    uniform_points=self.uniform_points,
                    near_surface_points=self.near_surface_points,
                    near_surface_sigma=self.near_surface_sigma,
                )
                cached_surface = None
                cached_bounds = None
            case_states.append((case_dir, cached_surface, cached_bounds, bounds, points, labels, params))

        params_list = list(decoder.parameters())
        for *_, params in case_states:
            params_list.extend(params.values())
        optimizer = torch.optim.Adam(params_list, lr=self.lr)
        for epoch in range(self.stage1_epochs):
            total = 0.0
            for case_dir, cached_surface, cached_bounds, fixed_bounds, fixed_points, fixed_labels, params in case_states:
                if self.resample_stage1_each_epoch:
                    if cached_surface is None or cached_bounds is None:
                        surface, _ = _load_centered_surface(case_dir)
                        bounds = _bounds_tuple_to_array(surface.bounds, padding=self.bounds_padding)
                    else:
                        surface = cached_surface
                        bounds = cached_bounds
                    points, labels = _sample_occupancy(
                        surface,
                        bounds,
                        rng,
                        uniform_points=self.uniform_points,
                        near_surface_points=self.near_surface_points,
                        near_surface_sigma=self.near_surface_sigma,
                    )
                else:
                    bounds = fixed_bounds
                    points = fixed_points
                    labels = fixed_labels
                points, labels = _select_occupancy_batch(points, labels, rng, self.occupancy_batch_points)
                query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
                target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
                bounds_t = torch.from_numpy(bounds).to(device_t)
                loss = _evaluate_occupancy_loss(
                    decoder,
                    params,
                    query,
                    target,
                    knn=self.knn,
                    occupancy_query_chunk=self.occupancy_query_chunk,
                )
                loss = loss + _gaussian_regularization(
                    params,
                    bounds_t,
                    mean_bound_weight=self.mean_bound_weight,
                    scale_bound_weight=self.scale_bound_weight,
                    norm_weight=self.norm_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
            if epoch == 0 or epoch + 1 == self.stage1_epochs or (epoch + 1) % 10 == 0:
                print(
                    f"Gaussian3D shared decoder [{epoch + 1}/{self.stage1_epochs}] "
                    f"loss={total / max(len(case_states), 1):.6f}",
                    flush=True,
                )

        torch.save(
            {
                "state_dict": decoder.cpu().state_dict(),
                "channels": self.channels,
                "hidden_dim": self.decoder_hidden_dim,
                "depth": self.decoder_depth,
            },
            self.decoder_path,
        )
        return self.decoder_path

    def load_shared_decoder(self) -> GaussianOccupancyDecoder:
        decoder = self._decoder()
        if not os.path.exists(self.decoder_path):
            raise FileNotFoundError(
                f"Missing shared occupancy decoder: {self.decoder_path}. "
                "Run fit_shared_decoder() or build_cache(..., ensure_shared_decoder=True) first."
            )
        state = torch.load(self.decoder_path, map_location="cpu")
        decoder.load_state_dict(state.get("state_dict", state), strict=True)
        return decoder

    def __getitem__(self, idx: int) -> dict[str, str | float]:
        case_dir = self.case_dirs[int(idx)]
        path = gaussian3d_cache_path(
            self.cache_root,
            self.train_type,
            case_dir.name,
            self.num_gaussians,
            self.channels,
            self.knn,
        )
        if not self.overwrite and valid_gaussian3d_cache(path, self.num_gaussians, self.channels):
            return {"gaussian3d": path, "loss": 0.0}

        decoder = self.load_shared_decoder()
        arrays, bounds, x_center, loss = fit_case_gaussian3d(
            case_dir,
            decoder,
            num_gaussians=self.num_gaussians,
            channels=self.channels,
            knn=self.knn,
            epochs=self.stage2_epochs,
            lr=self.lr,
            uniform_points=self.uniform_points,
            near_surface_points=self.near_surface_points,
            occupancy_batch_points=self.occupancy_batch_points,
            occupancy_query_chunk=self.occupancy_query_chunk,
            near_surface_sigma=self.near_surface_sigma,
            bounds_padding=self.bounds_padding,
            mean_bound_weight=self.mean_bound_weight,
            scale_bound_weight=self.scale_bound_weight,
            norm_weight=self.norm_weight,
            seed=self.seed + int(idx),
            device=self.device,
            resample_each_epoch=self.resample_stage2_each_epoch,
        )
        atomic_save_npz(
            path,
            **arrays,
            bounds=bounds,
            x_center=np.array(x_center, dtype=np.float32),
            num_gaussians=np.array(self.num_gaussians, dtype=np.int32),
            channels=np.array(self.channels, dtype=np.int32),
            knn=np.array(self.knn, dtype=np.int32),
            loss=np.array(loss, dtype=np.float32),
        )
        register_auxiliary_cache(path, arrays=arrays, shared_decoder=self.decoder_path)
        return {"gaussian3d": path, "loss": loss}

    def build_cache(
        self,
        start_index: int = 0,
        limit: int | None = None,
        log_every: int = 1,
        num_workers: int = 0,
        ensure_shared_decoder: bool = True,
    ) -> None:
        if ensure_shared_decoder and not os.path.exists(self.decoder_path):
            self.fit_shared_decoder()

        end = len(self) if limit is None else min(len(self), int(start_index) + int(limit))
        indices = list(range(int(start_index), end))
        if num_workers <= 0:
            for local_i, idx in enumerate(indices, start=1):
                result = self[idx]
                if log_every > 0 and (local_i % log_every == 0 or local_i == len(indices)):
                    print(
                        f"[{idx + 1}/{len(self)}] cached {self.case_dirs[idx].name}: {result['gaussian3d']}",
                        flush=True,
                    )
            return

        subset = Subset(self, indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate,
        )
        for local_i, _ in enumerate(loader, start=1):
            global_i = int(start_index) + local_i - 1
            if log_every > 0 and (local_i % log_every == 0 or global_i + 1 == end):
                print(f"[{global_i + 1}/{len(self)}] cached {self.case_dirs[global_i].name}", flush=True)


__all__ = [
    "DEFAULT_GAUSSIAN3D_CHANNELS",
    "DEFAULT_GAUSSIAN3D_COUNT",
    "DEFAULT_GAUSSIAN3D_K",
    "GaussianOccupancyDecoder",
    "HighSpeedTrainGaussian3DCacheBuilder",
    "fit_case_gaussian3d",
    "gaussian3d_cache_dir",
    "gaussian3d_cache_path",
    "shared_decoder_path",
    "valid_gaussian3d_cache",
]
