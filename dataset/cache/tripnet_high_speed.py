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
    complete_high_speed_raw_case,
    high_speed_case_sort_key,
    normalized_train_type,
)
from dataset.utils.high_speed_train_dataset import _split_high_speed_names_by_geometry
from dataset.utils.load import high_speed_type_dir, list_high_speed_cases
from dataset.utils.sampling import filter_names
from model.utils.triplane import sample_triplanes


DEFAULT_TRIPLANE_CHANNELS = 32
DEFAULT_TRIPLANE_RESOLUTION = 128


def triplane_cache_dir(
    cache_root: str | Path,
    train_type: str,
    channels: int = DEFAULT_TRIPLANE_CHANNELS,
    resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
) -> str:
    name = f"triplane_{int(channels)}x{int(resolution)}x{int(resolution)}_fp32"
    return os.path.join(str(cache_root), normalized_train_type(train_type), name)


def triplane_cache_path(
    cache_root: str | Path,
    train_type: str,
    case_name: str,
    channels: int = DEFAULT_TRIPLANE_CHANNELS,
    resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
) -> str:
    return os.path.join(triplane_cache_dir(cache_root, train_type, channels, resolution), f"{case_name}.npz")


def shared_decoder_path(
    cache_root: str | Path,
    train_type: str,
    channels: int = DEFAULT_TRIPLANE_CHANNELS,
    resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
) -> str:
    return os.path.join(triplane_cache_dir(cache_root, train_type, channels, resolution), "_shared_occupancy_decoder.pt")


def valid_triplane_cache(
    path: str | Path,
    channels: int = DEFAULT_TRIPLANE_CHANNELS,
    resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        data = np.load(path)
        triplane = data["triplane"]
        bounds = data["bounds"]
        x_center = data["x_center"]
        if triplane.shape != (3, int(channels), int(resolution), int(resolution)):
            return False
        if triplane.dtype != np.float32:
            return False
        if bounds.shape != (2, 3) or bounds.dtype != np.float32:
            return False
        if np.asarray(x_center).size != 1:
            return False
        return bool(np.all(np.isfinite(triplane)) and np.all(np.isfinite(bounds)))
    except Exception:
        return False


def _atomic_save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _bounds_tuple_to_array(bounds, padding: float = 0.05) -> np.ndarray:
    lower = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    upper = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float32)
    span = np.maximum(upper - lower, 1e-6)
    pad = span * float(padding)
    return np.stack([lower - pad, upper + pad], axis=0).astype(np.float32)


def _load_centered_surface(case_dir: str | Path):
    import pyvista as pv

    surface = pv.read(Path(case_dir) / "surface.vtk").extract_surface(algorithm="dataset_surface").triangulate()
    points = np.asarray(surface.points, dtype=np.float32).copy()
    x_center = float(0.5 * (points[:, 0].min() + points[:, 0].max()))
    points[:, 0] -= x_center
    surface.points = points
    return surface, x_center


def _sample_occupancy(
    surface,
    bounds: np.ndarray,
    rng: np.random.Generator,
    uniform_points: int,
    near_surface_points: int,
    near_surface_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    import pyvista as pv

    lower = bounds[0]
    upper = bounds[1]
    uniform = rng.uniform(lower, upper, size=(int(uniform_points), 3)).astype(np.float32)
    surface_points = np.asarray(surface.points, dtype=np.float32)
    if surface_points.shape[0] == 0 or int(near_surface_points) <= 0:
        points = uniform
    else:
        idx = rng.integers(0, surface_points.shape[0], size=int(near_surface_points))
        sigma = float(near_surface_sigma) * float(np.max(upper - lower))
        near = surface_points[idx] + rng.normal(0.0, sigma, size=(int(near_surface_points), 3)).astype(np.float32)
        near = np.clip(near, lower, upper)
        points = np.concatenate([uniform, near.astype(np.float32)], axis=0)

    cloud = pv.PolyData(points)
    try:
        if hasattr(cloud, "select_interior_points"):
            selected = cloud.select_interior_points(surface, check_surface=False)
            labels = np.asarray(selected.point_data["selected_points"], dtype=np.float32).reshape(-1, 1)
        else:
            selected = cloud.select_enclosed_points(surface, check_surface=False)
            labels = np.asarray(selected.point_data["SelectedPoints"], dtype=np.float32).reshape(-1, 1)
    except Exception:
        selected = cloud.select_enclosed_points(surface, check_surface=True)
        labels = np.asarray(selected.point_data["SelectedPoints"], dtype=np.float32).reshape(-1, 1)
    return points.astype(np.float32, copy=False), labels


class TriplaneOccupancyDecoder(nn.Module):
    def __init__(self, channels: int = DEFAULT_TRIPLANE_CHANNELS, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        self.channels = int(channels)
        layers: list[nn.Module] = []
        dim = self.channels
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.GELU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, sampled_features: torch.Tensor) -> torch.Tensor:
        if sampled_features.shape[-1] != 3 * self.channels:
            raise ValueError("sampled_features last dimension must equal 3 * channels.")
        features = sampled_features.view(*sampled_features.shape[:-1], 3, self.channels).sum(dim=-2)
        return self.net(features)


def _triplane_regularization(planes: torch.Tensor, tv_weight: float, norm_weight: float) -> torch.Tensor:
    tv_h = (planes[..., 1:, :] - planes[..., :-1, :]).abs().mean()
    tv_w = (planes[..., :, 1:] - planes[..., :, :-1]).abs().mean()
    norm = planes.square().mean()
    return float(tv_weight) * (tv_h + tv_w) + float(norm_weight) * norm


def fit_case_triplane(
    case_dir: str | Path,
    decoder: TriplaneOccupancyDecoder,
    *,
    channels: int = DEFAULT_TRIPLANE_CHANNELS,
    resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
    epochs: int = 70,
    lr: float = 1e-3,
    uniform_points: int = 32_768,
    near_surface_points: int = 32_768,
    near_surface_sigma: float = 0.01,
    bounds_padding: float = 0.05,
    tv_weight: float = 1e-5,
    norm_weight: float = 1e-6,
    seed: int = 0,
    device: str | torch.device | None = None,
    resample_each_epoch: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    surface, x_center = _load_centered_surface(case_dir)
    bounds = _bounds_tuple_to_array(surface.bounds, padding=bounds_padding)
    rng = np.random.default_rng(int(seed))
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    decoder = decoder.to(device_t).float().eval()
    for param in decoder.parameters():
        param.requires_grad_(False)

    planes = nn.Parameter(torch.randn(1, 3, int(channels), int(resolution), int(resolution), device=device_t) * 0.02)
    bounds_t = torch.from_numpy(bounds).to(device_t).view(1, 2, 3)
    optimizer = torch.optim.Adam([planes], lr=float(lr))
    fixed_query = None
    fixed_target = None
    if not resample_each_epoch:
        points, labels = _sample_occupancy(
            surface,
            bounds,
            rng,
            uniform_points=uniform_points,
            near_surface_points=near_surface_points,
            near_surface_sigma=near_surface_sigma,
        )
        fixed_query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
        fixed_target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
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
            query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
            target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
        else:
            query = fixed_query
            target = fixed_target
        sampled, _ = sample_triplanes(planes, query, bounds_t)
        logits = decoder(sampled)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss = loss + _triplane_regularization(planes, tv_weight, norm_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    return planes.detach().cpu().numpy()[0].astype(np.float32), bounds.astype(np.float32), float(x_center), last_loss


def identity_collate(batch):
    return batch


class HighSpeedTrainTriplaneCacheBuilder(Dataset):
    """Build per-case TripNet raw triplane caches for high_speed_train cases."""

    def __init__(
        self,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
        train_type: str = "CRH450",
        cache_root: str | None = None,
        channels: int = DEFAULT_TRIPLANE_CHANNELS,
        resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
        ids_file: str | None = None,
        split: str = "all",
        val_ratio: float = 0.1,
        geometry_count: int | None = None,
        overwrite: bool = False,
        skip_incomplete: bool = True,
        stage1_epochs: int = 500,
        stage1_case_limit: int | None = None,
        stage2_epochs: int = 70,
        lr: float = 1e-3,
        uniform_points: int = 32_768,
        near_surface_points: int = 32_768,
        near_surface_sigma: float = 0.01,
        bounds_padding: float = 0.05,
        decoder_hidden_dim: int = 128,
        decoder_depth: int = 3,
        cache_stage1_surfaces: bool = True,
        resample_stage1_each_epoch: bool = False,
        resample_stage2_each_epoch: bool = False,
        tv_weight: float = 1e-5,
        norm_weight: float = 1e-6,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.channels = int(channels)
        self.resolution = int(resolution)
        self.overwrite = bool(overwrite)
        self.stage1_epochs = int(stage1_epochs)
        self.stage1_case_limit = None if stage1_case_limit is None else int(stage1_case_limit)
        self.stage2_epochs = int(stage2_epochs)
        self.lr = float(lr)
        self.uniform_points = int(uniform_points)
        self.near_surface_points = int(near_surface_points)
        self.near_surface_sigma = float(near_surface_sigma)
        self.bounds_padding = float(bounds_padding)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.decoder_depth = int(decoder_depth)
        self.cache_stage1_surfaces = bool(cache_stage1_surfaces)
        self.resample_stage1_each_epoch = bool(resample_stage1_each_epoch)
        self.resample_stage2_each_epoch = bool(resample_stage2_each_epoch)
        self.tv_weight = float(tv_weight)
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
                print(f"TripNet triplane cache builder: skipped {skipped} incomplete cases.", flush=True)
        os.makedirs(triplane_cache_dir(self.cache_root, self.train_type, self.channels, self.resolution), exist_ok=True)
        print(
            "TripNet triplane cache builder: "
            f"{self.train_type}, {len(self.case_dirs)} cases, "
            f"channels={self.channels}, resolution={self.resolution}.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.case_dirs)

    def _decoder(self) -> TriplaneOccupancyDecoder:
        return TriplaneOccupancyDecoder(
            channels=self.channels,
            hidden_dim=self.decoder_hidden_dim,
            depth=self.decoder_depth,
        )

    @property
    def decoder_path(self) -> str:
        return shared_decoder_path(self.cache_root, self.train_type, self.channels, self.resolution)

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
            planes = nn.Parameter(
                torch.randn(1, 3, self.channels, self.resolution, self.resolution, device=device_t) * 0.02
            )
            if self.resample_stage1_each_epoch and self.cache_stage1_surfaces:
                surface, _ = _load_centered_surface(self.case_dirs[int(idx)])
                bounds = _bounds_tuple_to_array(surface.bounds, padding=self.bounds_padding)
                bounds_t = None
                query = None
                target = None
            elif self.resample_stage1_each_epoch:
                surface = None
                bounds = None
                bounds_t = None
                query = None
                target = None
            else:
                surface, _ = _load_centered_surface(self.case_dirs[int(idx)])
                bounds = _bounds_tuple_to_array(surface.bounds, padding=self.bounds_padding)
                points, labels = _sample_occupancy(
                    surface,
                    bounds,
                    rng,
                    uniform_points=self.uniform_points,
                    near_surface_points=self.near_surface_points,
                    near_surface_sigma=self.near_surface_sigma,
                )
                bounds_t = torch.from_numpy(bounds).to(device_t).view(1, 2, 3)
                query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
                target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
                surface = None
                bounds = None
            case_states.append((self.case_dirs[int(idx)], surface, bounds, bounds_t, query, target, planes))

        params = list(decoder.parameters()) + [item[6] for item in case_states]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        for epoch in range(self.stage1_epochs):
            total = 0.0
            for case_dir, cached_surface, cached_bounds, fixed_bounds_t, fixed_query, fixed_target, planes in case_states:
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
                    bounds_t = torch.from_numpy(bounds).to(device_t).view(1, 2, 3)
                    query = torch.from_numpy(points).to(device_t).view(1, -1, 3)
                    target = torch.from_numpy(labels).to(device_t).view(1, -1, 1)
                else:
                    bounds_t = fixed_bounds_t
                    query = fixed_query
                    target = fixed_target
                sampled, _ = sample_triplanes(planes, query, bounds_t)
                logits = decoder(sampled)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                loss = loss + _triplane_regularization(planes, self.tv_weight, self.norm_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
            if epoch == 0 or epoch + 1 == self.stage1_epochs or (epoch + 1) % 10 == 0:
                print(
                    f"TripNet shared decoder [{epoch + 1}/{self.stage1_epochs}] "
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

    def load_shared_decoder(self) -> TriplaneOccupancyDecoder:
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
        path = triplane_cache_path(self.cache_root, self.train_type, case_dir.name, self.channels, self.resolution)
        if not self.overwrite and valid_triplane_cache(path, self.channels, self.resolution):
            return {"triplane": path, "loss": 0.0}

        decoder = self.load_shared_decoder()
        triplane, bounds, x_center, loss = fit_case_triplane(
            case_dir,
            decoder,
            channels=self.channels,
            resolution=self.resolution,
            epochs=self.stage2_epochs,
            lr=self.lr,
            uniform_points=self.uniform_points,
            near_surface_points=self.near_surface_points,
            near_surface_sigma=self.near_surface_sigma,
            bounds_padding=self.bounds_padding,
            tv_weight=self.tv_weight,
            norm_weight=self.norm_weight,
            seed=self.seed + int(idx),
            device=self.device,
            resample_each_epoch=self.resample_stage2_each_epoch,
        )
        _atomic_save_npz(
            path,
            triplane=triplane,
            bounds=bounds,
            x_center=np.array(x_center, dtype=np.float32),
            resolution=np.array(self.resolution, dtype=np.int32),
            channels=np.array(self.channels, dtype=np.int32),
            loss=np.array(loss, dtype=np.float32),
        )
        return {"triplane": path, "loss": loss}

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
                        f"[{idx + 1}/{len(self)}] cached {self.case_dirs[idx].name}: {result['triplane']}",
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
    "DEFAULT_TRIPLANE_CHANNELS",
    "DEFAULT_TRIPLANE_RESOLUTION",
    "HighSpeedTrainTriplaneCacheBuilder",
    "TriplaneOccupancyDecoder",
    "fit_case_triplane",
    "shared_decoder_path",
    "triplane_cache_dir",
    "triplane_cache_path",
    "valid_triplane_cache",
]
