from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from model.Encoder.flow_surface_encoder import VolumeUniFieldEncoder
from model.utils.checkpoint import load_route_compatible_state_dict


class QueryUniField(nn.Module):
    """Composable surface-encoder + query-decoder UniField."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None,
        default_query_chunk_size: int = 50_000,
        coefficient_head_kwargs: Mapping[str, Any] | None = None,
        coordinate_normalization: Mapping[str, Any] | None = None,
        query_reuses_surface_points: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.default_query_chunk_size = int(default_query_chunk_size)
        self.query_reuses_surface_points = bool(query_reuses_surface_points)
        normalization = dict(coordinate_normalization or {})
        self.coordinate_normalization_mode = str(normalization.get("mode", "none")).lower()
        if self.coordinate_normalization_mode not in {"none", "utonia"}:
            raise ValueError("coordinate_normalization.mode must be 'none' or 'utonia'.")
        self.coordinate_normalization_external_cond_dim = int(
            normalization.get("external_cond_dim", 2)
        )
        self.coordinate_normalization_apply_z_positive = bool(
            normalization.get("apply_z_positive", True)
        )
        self.coordinate_normalization_eps = float(normalization.get("eps", 1e-6))
        if self.coordinate_normalization_external_cond_dim < 0:
            raise ValueError("coordinate_normalization.external_cond_dim must be non-negative.")
        if self.coordinate_normalization_eps <= 0.0:
            raise ValueError("coordinate_normalization.eps must be positive.")
        if self.coordinate_normalization_mode == "utonia":
            model_cond_dim = self.coordinate_normalization_external_cond_dim + 1
            for component_name, component in (("encoder", encoder), ("decoder", decoder)):
                component_cond_dim = getattr(component, "cond_dim", None)
                if component_cond_dim is not None and int(component_cond_dim) != model_cond_dim:
                    raise ValueError(
                        f"{component_name}.cond_dim must be {model_cond_dim} when Utonia "
                        "coordinate normalization appends the radius condition."
                    )
        self.coefficient_head = None
        if coefficient_head_kwargs:
            head_kwargs = dict(coefficient_head_kwargs)
            head_type = str(head_kwargs.pop("head_type", "mean_pool")).lower()
            if head_type == "mean_pool":
                self.coefficient_head = CoefficientHead(**head_kwargs)
            elif head_type == "pose_aware":
                self.coefficient_head = PoseAwareCoefficientHead(**head_kwargs)
            else:
                raise ValueError(f"Unsupported coefficient head_type: {head_type!r}.")

    def _prepare_model_coordinates(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.coordinate_normalization_mode == "none":
            return surface_pos, query_pos, cond, None
        if surface_pos.dim() != 3 or surface_pos.shape[-1] != 3:
            raise ValueError("surface_pos must have shape (B, N, 3) for Utonia normalization.")
        if query_pos.dim() != 3 or query_pos.shape[-1] < 3:
            raise ValueError("query_pos must have shape (B, N, C) with C >= 3.")

        center = surface_pos.mean(dim=1, keepdim=True)
        radius = (surface_pos - center).norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
        radius_safe = radius.clamp_min(self.coordinate_normalization_eps)
        surface_model = (surface_pos - center) / radius_safe

        query_model = query_pos.clone()
        query_model[..., :3] = (query_pos[..., :3] - center) / radius_safe
        if self.coordinate_normalization_apply_z_positive:
            lower = surface_model.amin(dim=1, keepdim=True)
            upper = surface_model.amax(dim=1, keepdim=True)
            shift = torch.cat(
                [
                    (lower[..., :2] + upper[..., :2]) / 2,
                    lower[..., 2:3],
                ],
                dim=-1,
            )
            surface_model = surface_model - shift
            query_model[..., :3] = query_model[..., :3] - shift

        external_dim = self.coordinate_normalization_external_cond_dim
        if cond is None:
            cond = surface_pos.new_zeros(surface_pos.shape[0], external_dim)
        elif cond.shape != (surface_pos.shape[0], external_dim):
            raise ValueError(
                "Utonia coordinate normalization expects external condition "
                f"with shape (B, {external_dim})."
            )
        cond_model = torch.cat([cond.to(surface_pos), radius[:, 0, :].to(surface_pos)], dim=-1)
        return surface_model, query_model, cond_model, radius[:, 0, :]

    def _validate_explicit_query_reuse(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> None:
        if self.query_reuses_surface_points and query_pos.shape[:2] != surface_pos.shape[:2]:
            raise ValueError(
                "query_reuses_surface_points=True requires the query batch and point dimensions "
                "to match surface_pos."
            )

    def _pack_output(self, query_pred: torch.Tensor, memory: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.coefficient_head is None:
            return query_pred
        return {
            "query": query_pred,
            "coefficients": self.coefficient_head(memory),
        }

    def encode_surface(
        self,
        surface_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        geometry_meta: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        cond_t = self.encoder._prepare_cond(surface_pos, cond)
        routes_t = self.encoder._prepare_routes(surface_pos, routes)
        flow = self.encoder._prepare_flow(surface_pos, cond_t, flow_vec)
        encoder_kwargs = {
            "cond": cond_t,
            "normals": normals,
            "area": area,
            "flow_vec": flow,
            "routes": routes_t,
            "return_pyramid": False,
        }
        if geometry_meta is not None:
            encoder_kwargs["geometry_meta"] = geometry_meta
        memory = self.encoder(surface_pos, **encoder_kwargs)
        decoder_cond = memory.get("decoder_cond", cond_t)
        # Preserve physical geometry inputs for coefficient-only heads.  These
        # are normalized coordinates, rotated normals, and quadrature areas;
        # no explicit AoA is supplied or reconstructed outside the network.
        memory["input_points"] = surface_pos
        memory["input_normals"] = normals
        memory["input_area"] = area
        memory["condition"] = decoder_cond
        return memory, decoder_cond, routes_t, flow

    def forward_coefficients(
        self,
        surface_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        geometry_meta: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict global coefficients from geometry, without field decoding.

        Output ordering is defined by the task/config coefficient target list.
        This path retains gradients so the encoder and pooled MLP can train
        together. It accepts no CFD labels or surface pressure query inputs.
        """
        if self.coefficient_head is None:
            raise ValueError("forward_coefficients requires a coefficient_head.")
        surface_pos, _, cond, radius = self._prepare_model_coordinates(
            surface_pos, surface_pos, cond,
        )
        memory, _, _, _ = self.encode_surface(
            surface_pos, cond=cond, routes=routes, normals=normals,
            area=area, flow_vec=flow_vec, geometry_meta=geometry_meta,
        )
        if radius is not None:
            memory["normalization_radius"] = radius
        return self.coefficient_head(memory)

    def forward(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        geometry_meta: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._validate_explicit_query_reuse(surface_pos, query_pos)
        surface_pos, query_pos, cond, normalization_radius = self._prepare_model_coordinates(
            surface_pos,
            query_pos,
            cond,
        )
        memory, cond_t, routes_t, flow = self.encode_surface(
            surface_pos,
            cond=cond,
            routes=routes,
            normals=normals,
            area=area,
            flow_vec=flow_vec,
            geometry_meta=geometry_meta,
        )
        if normalization_radius is not None:
            memory["normalization_radius"] = normalization_radius
        if self.query_reuses_surface_points:
            memory["query_reuses_surface_points"] = True
        if self.decoder is None:
            query_pred = surface_pos.new_empty(surface_pos.shape[0], 0, 0)
        else:
            query_pred = self.decoder(query_pos, memory, cond_t, routes_t, flow)
        return self._pack_output(query_pred, memory)

    @torch.no_grad()
    def forward_test(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        geometry_meta: Optional[Mapping[str, torch.Tensor]] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.decoder is None:
            self._validate_explicit_query_reuse(surface_pos, query_pos)
            surface_pos, query_pos, cond, normalization_radius = self._prepare_model_coordinates(
                surface_pos,
                query_pos,
                cond,
            )
            memory, _, _, _ = self.encode_surface(
                surface_pos,
                cond=cond,
                routes=routes,
                normals=normals,
                area=area,
                flow_vec=flow_vec,
                geometry_meta=geometry_meta,
            )
            if normalization_radius is not None:
                memory["normalization_radius"] = normalization_radius
            query_pred = surface_pos.new_empty(surface_pos.shape[0], 0, 0)
            return self._pack_output(query_pred, memory)
        if not getattr(self.decoder, "supports_chunked_forward", False):
            raise NotImplementedError(
                f"{self.decoder.__class__.__name__} uses query-query interaction and does not support "
                "chunk-equivalent forward_test."
            )
        query_chunk_size = self.default_query_chunk_size if query_chunk_size is None else int(query_chunk_size)
        self._validate_explicit_query_reuse(surface_pos, query_pos)
        surface_pos, query_pos, cond, normalization_radius = self._prepare_model_coordinates(
            surface_pos,
            query_pos,
            cond,
        )
        memory, cond_t, routes_t, flow = self.encode_surface(
            surface_pos,
            cond=cond,
            routes=routes,
            normals=normals,
            area=area,
            flow_vec=flow_vec,
            geometry_meta=geometry_meta,
        )
        if normalization_radius is not None:
            memory["normalization_radius"] = normalization_radius
        if self.query_reuses_surface_points:
            memory["query_reuses_surface_points"] = True
        prepared = self.decoder.prepare_memory(memory, cond_t, routes_t, flow)
        if self.query_reuses_surface_points or getattr(self.decoder, "requires_full_query_forward", False):
            query_pred = self.decoder.decode_with_memory(query_pos, prepared, cond_t, routes_t, flow)
            return self._pack_output(query_pred, memory)

        outputs = []
        for start in range(0, query_pos.shape[1], query_chunk_size):
            query_chunk = query_pos[:, start : start + query_chunk_size]
            outputs.append(
                self.decoder.decode_with_memory(query_chunk, prepared, cond_t, routes_t, flow)
            )
        query_pred = torch.cat(outputs, dim=1)
        return self._pack_output(query_pred, memory)

    def load_route_compatible_state_dict(
        self,
        checkpoint: Mapping[str, Any],
        route_map: Optional[Sequence[int]] = None,
        strict: bool = False,
        partial_load: bool = False,
    ) -> dict[str, list[str]]:
        return load_route_compatible_state_dict(
            self,
            checkpoint,
            route_map=route_map,
            strict=strict,
            partial_load=partial_load,
        )


class CoefficientHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int = 4,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
        output_bias: Sequence[float] | None = None,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(in_dim if hidden_dim is None else hidden_dim)
        layers: list[nn.Module] = [nn.LayerNorm(int(in_dim))]
        dim = int(in_dim)
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = hidden_dim
        output = nn.Linear(dim, int(out_channels))
        if bool(zero_init_output):
            nn.init.zeros_(output.weight)
        if output_bias is not None:
            bias = torch.as_tensor(output_bias, dtype=output.bias.dtype)
            if bias.shape != output.bias.shape:
                raise ValueError(
                    f"output_bias must have shape {tuple(output.bias.shape)}, got {tuple(bias.shape)}."
                )
            with torch.no_grad():
                output.bias.copy_(bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

    def forward(self, memory: dict[str, torch.Tensor]) -> torch.Tensor:
        features = memory["features"]
        pooled = features.mean(dim=1)
        return self.mlp(pooled)


class PoseAwareCoefficientHead(nn.Module):
    """Coefficient regressor that retains signed pose information from geometry.

    The head receives no AoA scalar.  Signed orientation is represented by
    area-weighted point/normal moments, in particular the off-diagonal spatial
    moments that change sign when an inclined vehicle is mirrored.
    """

    _GEOMETRY_DIM = 3 + 6 + 3 + 3 + 9

    def __init__(
        self,
        in_dim: int,
        out_channels: int = 2,
        hidden_dim: int = 1024,
        depth: int = 2,
        dropout: float = 0.0,
        separate_outputs: bool = True,
        output_bias: Sequence[float] | None = None,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        if int(out_channels) != 2:
            raise ValueError("PoseAwareCoefficientHead currently requires [Cd, Cl] outputs.")
        self.in_dim = int(in_dim)
        self.separate_outputs = bool(separate_outputs)
        pooled_dim = self.in_dim + self._GEOMETRY_DIM
        biases = (0.0, 0.0) if output_bias is None else tuple(float(v) for v in output_bias)
        if len(biases) != 2:
            raise ValueError("output_bias must contain [Cd, Cl].")

        def branch(bias: float) -> nn.Sequential:
            layers: list[nn.Module] = [nn.LayerNorm(pooled_dim)]
            dim = pooled_dim
            for _ in range(max(int(depth) - 1, 0)):
                layers.extend([nn.Linear(dim, int(hidden_dim)), nn.GELU()])
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                dim = int(hidden_dim)
            output = nn.Linear(dim, 1)
            if bool(zero_init_output):
                nn.init.zeros_(output.weight)
            with torch.no_grad():
                output.bias.fill_(bias)
            layers.append(output)
            return nn.Sequential(*layers)

        if self.separate_outputs:
            self.cd_branch = branch(biases[0])
            self.cl_branch = branch(biases[1])
            self.shared = None
        else:
            self.cd_branch = self.cl_branch = None
            layers: list[nn.Module] = [nn.LayerNorm(pooled_dim)]
            dim = pooled_dim
            for _ in range(max(int(depth) - 1, 0)):
                layers.extend([nn.Linear(dim, int(hidden_dim)), nn.GELU()])
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
                dim = int(hidden_dim)
            output = nn.Linear(dim, 2)
            if bool(zero_init_output):
                nn.init.zeros_(output.weight)
            with torch.no_grad():
                output.bias.copy_(torch.as_tensor(biases, dtype=output.bias.dtype))
            layers.append(output)
            self.shared = nn.Sequential(*layers)

    @staticmethod
    def _normalized_weights(points: torch.Tensor, area: torch.Tensor | None) -> torch.Tensor:
        if area is None:
            return points.new_full(points.shape[:2], 1.0 / points.shape[1])
        weights = area.to(points).squeeze(-1).clamp_min(0.0)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(points.dtype).eps)

    @classmethod
    def geometry_descriptor(cls, memory: dict[str, torch.Tensor]) -> torch.Tensor:
        points = memory["input_points"]
        normals = memory.get("input_normals")
        weights = cls._normalized_weights(points, memory.get("input_area"))
        weighted = weights[..., None]
        centroid = (points * weighted).sum(dim=1)
        centered = points - centroid[:, None, :]
        x, y, z = centered.unbind(dim=-1)
        second = torch.stack((x*x, y*y, z*z, x*y, x*z, y*z), dim=-1)
        second = (second * weighted).sum(dim=1)
        extent = points.amax(dim=1) - points.amin(dim=1)
        if normals is None:
            normal_mean = points.new_zeros(points.shape[0], 3)
            position_normal = points.new_zeros(points.shape[0], 9)
        else:
            normals = torch.nn.functional.normalize(normals.to(points), dim=-1, eps=1e-6)
            normal_mean = (normals * weighted).sum(dim=1)
            position_normal = torch.einsum("bni,bnj,bn->bij", centered, normals, weights).flatten(1)
        descriptor = torch.cat((centroid, second, extent, normal_mean, position_normal), dim=-1)
        if descriptor.shape[-1] != cls._GEOMETRY_DIM:
            raise RuntimeError("Unexpected pose-aware geometry descriptor width.")
        return descriptor

    def forward(self, memory: dict[str, torch.Tensor]) -> torch.Tensor:
        features = memory["features"]
        pooled = torch.cat((features.mean(dim=1), self.geometry_descriptor(memory)), dim=-1)
        if self.shared is not None:
            return self.shared(pooled)
        assert self.cd_branch is not None and self.cl_branch is not None
        return torch.cat((self.cd_branch(pooled), self.cl_branch(pooled)), dim=-1)


__all__ = ["QueryUniField", "CoefficientHead", "PoseAwareCoefficientHead"]
