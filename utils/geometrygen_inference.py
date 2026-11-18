"""Geometry-only joint-field inference. No CFD files or synthetic truth required.

Input mesh is canonical, uniformly fitted BEFORE CAD and base-cuboid centred.
Angles rotate the mesh exactly once; coordinates remain metres (scale=1).
Geometry preparation uses aerosim and GeometryGen from the AutoCFD checkout
at /data/home/zdhs0017/OpenFOAM/Cases/AutoCFD (override AUTOCFD_ROOT).
"""
from pathlib import Path
import sys
import os
import time

import numpy as np

ROOT = Path(os.environ.get('AUTOCFD_ROOT', '/data/home/zdhs0017/OpenFOAM/Cases/AutoCFD')).expanduser()
REPO = Path(__file__).resolve().parents[1]
for path in (ROOT, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DOMAIN = np.array([[-5., -5., -5.], [10., 5., 5.]])
RUN = Path('GeometryGen_VolumeUniField_Medium_SurfacePUDistance_VolumePUDistance/Steps_100000_GPU_1')
CHECKPOINT = Path.home()/'zoujunhong/model/savemodel/field'/RUN/'latest.pth'
CONFIG = REPO/'log'/RUN/'config.py'


def near_bounds(bounds):
    b = np.asarray(bounds, dtype=float).copy()
    b[0] -= [.5, .5, .5]
    b[1] += [2., .5, .5]
    return np.stack([np.maximum(b[0], DOMAIN[0]), np.minimum(b[1], DOMAIN[1])])


def unpack_fields(raw, ux):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError('Expected finite N x 5 joint-field output')
    velocity = (raw[:, :3] + np.array([1, 0, 0], dtype=np.float32))*ux
    np.testing.assert_allclose(velocity/ux-[1, 0, 0], raw[:, :3], atol=3e-7, rtol=3e-6)
    return dict(Cp=raw[:, 3].copy(), Ustar=raw[:, :3].copy(), U_m_s=velocity,
                speed_ratio=np.linalg.norm(velocity, axis=1)/ux,
                sqrt_distance=raw[:, 4].copy(), distance_m=np.maximum(raw[:, 4], 0)**2)


class JointPredictor:
    """Strict checkpoint loading; one surface encoding, bounded query chunks."""
    def __init__(self, checkpoint=CHECKPOINT, config=CONFIG, chunk=8192):
        import torch
        from model import build_model
        from utils.config import load_main_config
        from utils.checkpoint import load_model_weights
        self.torch = torch
        torch.manual_seed(20261013)
        torch.set_float32_matmul_precision('high')
        cfg = load_main_config(str(config))
        if cfg.get('migration', {}).get('evaluation_requires_legacy_source'):
            raise RuntimeError('This historical variant requires its archived model source; the current model is not an equivalent substitute.')
        if cfg['model']['coordinate_normalization']['mode'] != 'none':
            raise ValueError('This interface requires raw metre coordinates')
        self.model = build_model(cfg['model']).cuda().float().eval()
        load_model_weights(self.model, checkpoint, strict=True,
                           trusted_legacy=bool(cfg.get('test', {}).get('trust_legacy_checkpoint', False)))
        self.chunk = int(chunk)
        self.timing = {}
        print('Loaded fixed joint checkpoint; BF16 batch1, no compile', flush=True)

    def encode(self, inputs, ux):
        torch = self.torch
        a = np.asarray(inputs, dtype=np.float32)
        if a.shape != (8192, 6) or not np.isfinite(a).all() or ux <= 0:
            raise ValueError('Input must be finite 8192 x 6; positive Ux required')
        np.testing.assert_allclose(np.linalg.norm(a[:, 3:], axis=1), 1, atol=2e-4)
        xyz = torch.as_tensor(a[:, :3].copy(), device='cuda')[None]
        normals = torch.as_tensor(a[:, 3:].copy(), device='cuda')[None]
        cond = torch.tensor([[ux, 1.]], device='cuda')
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            memory, self.cond, self.routes, self.flow = self.model.encode_surface(xyz, cond, routes=0, normals=normals)
            self.prepared = self.model.decoder.prepare_memory(memory, self.cond, self.routes, self.flow)
        torch.cuda.synchronize()
        self.timing['encode_seconds'] = time.monotonic()-started
        self.ux = float(ux)
        print(f'Encoded geometry in {self.timing["encode_seconds"]:.2f}s, condition=({ux:.6g},1)', flush=True)

    def predict_raw(self, xyz, chunk=None):
        torch = self.torch
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Query coordinates must be finite N x 3')
        out = np.empty((len(xyz), 5), dtype=np.float32)
        size = self.chunk if chunk is None else int(chunk)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for first in range(0, len(xyz), size):
                block = xyz[first:first+size]
                # Keep GEMM/attention shape identical even for the final block.
                # Queries do not interact, so padding has no physical effect.
                padded = np.zeros((max(self.chunk, len(block)), 3), dtype=np.float32)
                padded[:len(block)] = block
                q = torch.as_tensor(padded, device='cuda')[None]
                pred = self.model.decoder.decode_with_memory(q, self.prepared, self.cond, self.routes, self.flow)
                out[first:first+size] = pred[0, :len(block)].float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite model output')
        if len(xyz) > 8192:
            print(f'Predicted {len(xyz)} queries', flush=True)
        return out

    def verify_chunks(self, xyz):
        torch = self.torch
        q = np.asarray(xyz)[:256]
        a, b = self.predict_raw(q, chunk=256), self.predict_raw(q, chunk=73)
        bf16_error = float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(a, b, atol=.002, rtol=.002)
        # Verify mathematical chunk independence in FP32. BF16 fused kernels
        # are shape-dependent and can amplify rounding near output cancellation;
        # preserve/report that observed error instead of relaxing a failed test.
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision('highest')
        prepared = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                    for k, v in self.prepared.items()}
        def reference(size):
            outputs = []
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                for first in range(0, len(q), size):
                    qt = torch.as_tensor(q[first:first+size].copy(), dtype=torch.float32, device='cuda')[None]
                    value = self.model.decoder.decode_with_memory(qt, prepared, self.cond, self.routes, self.flow)
                    outputs.append(value[0].cpu().numpy())
            return np.concatenate(outputs)
        try:
            c, d = reference(256), reference(73)
            np.testing.assert_allclose(c, d, atol=2e-4, rtol=2e-4)
        finally:
            torch.set_float32_matmul_precision(original_precision)
        return dict(fp32_max_abs_difference=float(np.max(np.abs(c-d))), fp32_atol=2e-4,
                    fp32_rtol=2e-4, fp32_chunk_equivalent=True, query_counts=[256, 73],
                    bf16_max_abs_difference=bf16_error,
                    bf16_chunk_equivalent=True, bf16_atol=.002, bf16_rtol=.002,
                    bf16_note='All production/test blocks padded to fixed8192 to avoid shape-dependent rounding')


def triangulation(mesh):
    t = mesh.triangulate()
    faces = np.asarray(t.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError('Expected all triangles')
    return t, faces[:, 1:]


def refine_surface(mesh, max_edge=.02, max_triangles=250000):
    """Conforming marked-edge bisection; linear triangles retain exact shape.

    Unlike global four-way refinement, only long edges are bisected, avoiding
    excessive growth of the very thin CAD triangles. Shared edges use identical
    midpoint indices, so refinement introduces no T-junctions or cracks.
    """
    import pyvista as pv
    mesh, faces = triangulation(mesh)
    # Remove ONLY redundant perfectly planar triangulation vertices, with VTK's
    # zero absolute-error constraint. Curvature/shape simplification is forbidden.
    # The original geometry still drives FPS, random sampling and solid masks.
    from vtkmodules.vtkFiltersCore import vtkDecimatePro
    zero = vtkDecimatePro()
    zero.SetInputData(mesh); zero.SetTargetReduction(.5)
    zero.PreserveTopologyOn(); zero.SplittingOff(); zero.BoundaryVertexDeletionOff()
    zero.SetFeatureAngle(.001); zero.SetMaximumError(0.); zero.SetErrorIsAbsolute(1)
    zero.Update()
    planar = pv.wrap(zero.GetOutput())
    if planar.n_cells < mesh.n_cells:
        planar, planar_faces = triangulation(planar)
        probes = np.vstack([planar.points, planar.points[planar_faces].mean(axis=1)])
        forward = np.abs(pv.PolyData(probes).compute_implicit_distance(mesh)['implicit_distance'])
        backward = np.abs(pv.PolyData(mesh.points).compute_implicit_distance(planar)['implicit_distance'])
        unchanged = (max(forward.max(), backward.max()) <= 1e-10
                     and np.isclose(planar.area, mesh.area, atol=1e-10, rtol=1e-10)
                     and np.isclose(planar.volume, mesh.volume, atol=1e-10, rtol=1e-10))
        if unchanged:
            print(f'Zero-error planar triangulation cleanup: {mesh.n_cells} -> {planar.n_cells}; shape unchanged', flush=True)
            mesh, faces = planar, planar_faces
        else:
            print('Discarded planar cleanup: strict shape-equality audit failed; retain ORIGINAL triangulation', flush=True)
    points = np.asarray(mesh.points, dtype=np.float64).copy()
    for iteration in range(24):
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
        unique, inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
        lengths = np.linalg.norm(points[unique[:, 0]]-points[unique[:, 1]], axis=1)
        split = lengths > max_edge*(1+1e-10)
        if not np.any(split):
            out = pv.PolyData(points, np.column_stack([np.full(len(faces), 3), faces]).ravel())
            return out
        midpoint_ids = np.full(len(unique), -1, dtype=np.int64)
        midpoint_ids[split] = np.arange(len(points), len(points)+int(split.sum()))
        points = np.vstack([points, (points[unique[split, 0]]+points[unique[split, 1]])/2])
        mids = midpoint_ids[inverse].reshape(-1, 3)
        count = np.sum(mids >= 0, axis=1)
        expected = int(np.sum(count+1))
        print(f'Refinement pass {iteration}: {len(faces)} -> {expected} triangles; longest edge {lengths.max():.6g}m', flush=True)
        if expected > max_triangles:
            raise ValueError(f'Cannot achieve max edge {max_edge} within {max_triangles} triangles: next pass {expected}')
        pieces = [faces[count == 0]]
        for rotation in range(3):
            order = [(rotation+j) % 3 for j in range(3)]
            f, m = faces[:, order], mids[:, order]
            one = (count == 1) & (m[:, 0] >= 0)
            v, h = f[one], m[one, 0]
            pieces += [np.column_stack([v[:, 0], h, v[:, 2]]), np.column_stack([h, v[:, 1], v[:, 2]])]
            two = (count == 2) & (m[:, 0] >= 0) & (m[:, 1] >= 0)
            v, h = f[two], m[two]
            pieces += [np.column_stack([v[:, 1], h[:, 1], h[:, 0]])]
            # Split the residual quad along its SHORTER diagonal. The other
            # diagonal can repeatedly introduce long edges in needle CAD faces.
            diagonal_a = np.linalg.norm(points[v[:, 0]]-points[h[:, 1]], axis=1)
            diagonal_b = np.linalg.norm(points[h[:, 0]]-points[v[:, 2]], axis=1)
            choose_a = diagonal_a <= diagonal_b
            va, ha = v[choose_a], h[choose_a]
            vb, hb = v[~choose_a], h[~choose_a]
            pieces += [np.column_stack([va[:, 0], ha[:, 0], ha[:, 1]]),
                       np.column_stack([va[:, 0], ha[:, 1], va[:, 2]]),
                       np.column_stack([vb[:, 0], hb[:, 0], vb[:, 2]]),
                       np.column_stack([hb[:, 0], hb[:, 1], vb[:, 2]])]
        v, h = faces[count == 3], mids[count == 3]
        pieces += [np.column_stack([v[:, 0], h[:, 0], h[:, 2]]),
                   np.column_stack([h[:, 0], v[:, 1], h[:, 1]]),
                   np.column_stack([h[:, 2], h[:, 1], v[:, 2]]),
                   np.column_stack([h[:, 0], h[:, 1], h[:, 2]])]
        faces = np.vstack(pieces)
        if len(faces) != expected:
            raise AssertionError('Refinement triangulation count mismatch')
    raise RuntimeError('Refinement iteration limit exceeded')


def area_sample(mesh, count, rng):
    t, faces = triangulation(mesh)
    v = np.asarray(t.points, dtype=np.float64)[faces]
    cross = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if np.any(twice_area <= 0):
        raise ValueError('Degenerate triangle in input surface')
    ids = rng.choice(len(faces), count, p=twice_area/twice_area.sum())
    r = rng.random((count, 2)); a = np.sqrt(r[:, 0]); b = r[:, 1]
    xyz = (1-a[:, None])*v[ids, 0] + (a*(1-b))[:, None]*v[ids, 1] + (a*b)[:, None]*v[ids, 2]
    normals = cross[ids]/twice_area[ids, None]
    return xyz.astype(np.float32), normals.astype(np.float32)


def fluid_mask(xyz, surface):
    """VTK topological inside/outside test, never the network distance head."""
    import pyvista as pv
    test = pv.PolyData(np.asarray(xyz, dtype=np.float64)).select_enclosed_points(
        surface, tolerance=1e-7, check_surface=True)
    return np.asarray(test.point_data['SelectedPoints']) == 0


def sample_fluid(surface, bounds, count, rng):
    chunks = []; left = count; drawn = 0; post_cast_rejections = 0
    while left:
        # Classify in the SAME float32 coordinates that the model will query.
        candidates = rng.uniform(bounds[0], bounds[1], size=(max(left+128, int(left*1.15)), 3)).astype(np.float32)
        mask = fluid_mask(candidates, surface)
        accepted = candidates[mask][:left]
        chunks.append(accepted); left -= len(accepted); drawn += len(candidates)
        if drawn > count*50:
            raise RuntimeError('Cannot sample enough fluid points')
    xyz = np.concatenate(chunks).astype(np.float32)
    # Dense CFD skins can give a handful of shape/order-sensitive ray tests.
    # Reject those points and top up; never relax the geometry predicate.
    for validation_pass in range(12):
        final_mask = fluid_mask(xyz, surface)
        if final_mask.all():
            return xyz, dict(drawn=drawn, accepted=count, all_queries_in_fluid=True,
                             post_cast_rejections=post_cast_rejections,
                             final_float32_mask_verified=True, validation_passes=validation_pass+1)
        post_cast_rejections += int((~final_mask).sum())
        xyz = xyz[final_mask]
        needed = count-len(xyz)
        extra = rng.uniform(bounds[0], bounds[1], size=(needed+128, 3)).astype(np.float32)
        drawn += len(extra)
        extra = extra[fluid_mask(extra, surface)][:needed]
        xyz = np.vstack([xyz, extra])
        if len(xyz) != count:
            raise ValueError('Cannot top up stable float32 exterior queries')
    raise ValueError('VTK exterior classification remains unstable after 12 reject/resample passes')


def plane_grid(surface, bounds, axis, resolution=(384, 256)):
    """Direct regular grid, conservative geometric hole; no field interpolation.

    A one-cell-diagonal exclusion band prevents rendered cells spanning a thin
    solid even when their corner points are outside. Boundary outline is exact.
    """
    import pyvista as pv
    axes = [i for i in range(3) if i != axis]
    nx, ny = resolution
    x = np.linspace(bounds[0, axes[0]], bounds[1, axes[0]], nx)
    y = np.linspace(bounds[0, axes[1]], bounds[1, axes[1]], ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    xyz = np.zeros((nx, ny, 3)); xyz[..., axes[0]] = xx; xyz[..., axes[1]] = yy
    grid = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    mask = fluid_mask(grid.points, surface)
    distances = np.abs(grid.compute_implicit_distance(surface).point_data['implicit_distance'])
    diagonal = float(np.hypot(x[1]-x[0], y[1]-y[0]))
    mask &= distances > diagonal
    grid.point_data['fluid_query_mask'] = mask.astype(np.uint8)
    # Explicit VTK point ordering is Fortran order for a StructuredGrid.
    quad = grid.extract_surface().triangulate()
    _, faces = triangulation(quad)
    # extract_surface may reorder vertices: carry the mask through VTK rather
    # than incorrectly indexing it in the original StructuredGrid order.
    keep = np.asarray(quad.point_data['fluid_query_mask'])[faces].all(axis=1)
    result = quad.extract_cells(np.flatnonzero(keep)).extract_surface()
    result.clear_data()
    rejected_cells = 0
    for validation_pass in range(12):
        checked = fluid_mask(result.points, surface)
        if checked.all():
            break
        _, final_faces = triangulation(result)
        keep_final = checked[final_faces].all(axis=1)
        rejected_cells += int((~keep_final).sum())
        result = result.extract_cells(np.flatnonzero(keep_final)).extract_surface()
        result.clear_data()
    else:
        raise ValueError('Plane exterior classification remains unstable after 12 conservative rejection passes')
    return result, dict(resolution=list(resolution), exclusion_band_m=diagonal,
                        axis=axis, points=result.n_points, cells=result.n_cells,
                        validation_rejected_cells=rejected_cells,
                        final_geometry_mask_verified=True, validation_passes=validation_pass+1)


def load_rotated_geometry(path, alpha, beta):
    import pyvista as pv
    from aerosim.case import geometry_rotation
    from GeometryGen.surface_mesh import load_surface_mesh, accepted_closed_surface
    source = load_surface_mesh(Path(path))
    if not accepted_closed_surface(source):
        raise ValueError('Canonical geometry is not closed, oriented, single component')
    if np.any(np.abs(source.vertices) > np.array([1., 1., .5])+5e-5):
        raise ValueError('Canonical input escapes the training normalization envelope; do not rescale after CAD')
    R = geometry_rotation(alpha, beta)
    np.testing.assert_allclose(R.T@R, np.eye(3), atol=1e-12)
    vertices = np.asarray(source.vertices)@R.T
    faces = np.asarray(source.faces)
    normals_rotated = np.asarray(source.face_normals)@R.T
    cross = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    cross /= np.linalg.norm(cross, axis=1, keepdims=True)
    np.testing.assert_allclose(cross, normals_rotated, atol=2e-6)
    np.testing.assert_allclose(vertices@R, source.vertices, atol=1e-12)
    mesh = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    return mesh, dict(rotation_world_from_canonical=R.tolist(), scale=1.,
                     rotation_and_normal_consistent=True, centre_rule='base cuboid, not final bounding box')


def attach_fields(mesh, raw, ux, association='point', prefix=''):
    data = mesh.point_data if association == 'point' else mesh.cell_data
    for name, values in unpack_fields(raw, ux).items():
        data[prefix+name] = values.copy()
    return mesh


def infer_geometry(canonical_mesh_path, mach, alpha_deg, beta_deg, predictor=None,
                   seed=20261013, resolution=(384, 256)):
    """Return real geometry/query coordinates and model fields, without CFD.

    Output dictionary contains surface random queries, refined surface, volume
    point cloud and direct plane meshes. No truth or error fields are created.
    """
    import pyvista as pv
    from aerosim.atmosphere import resolve_freestream
    from dataset.GeometryGenSurface import _farthest_point_indices
    start = time.monotonic(); timings = {}
    rng = np.random.default_rng(seed)
    mesh, validation = load_rotated_geometry(canonical_mesh_path, alpha_deg, beta_deg)
    ux = resolve_freestream(mach).speed
    if not .05 <= mach <= .6:
        raise ValueError('Mach outside supported experiment range')
    candidates, normals = area_sample(mesh, 32768, rng)
    ids = _farthest_point_indices(candidates, 8192)
    inputs = np.column_stack([candidates[ids], normals[ids]])
    surface_xyz, _ = area_sample(mesh, 65536, rng)
    near = near_bounds(np.asarray(mesh.bounds).reshape(3, 2).T)
    full_xyz, full_audit = sample_fluid(mesh, DOMAIN, 65536, rng)
    near_xyz, near_audit = sample_fluid(mesh, near, 65536, rng)
    volume_xyz = np.vstack([full_xyz, near_xyz])
    refined = refine_surface(mesh, max_edge=.02, max_triangles=250000)
    refined, faces = triangulation(refined)
    v = refined.points[faces]
    max_edge = float(max(np.linalg.norm(v[:, a]-v[:, b], axis=1).max() for a, b in [(0, 1), (1, 2), (2, 0)]))
    if refined.n_cells > 250000:
        raise ValueError('Surface subdivision exceeded triangle budget')
    if max_edge > .0200001:
        raise ValueError('Surface edge target not achieved')
    # VTK linear subdivision does not project/smooth vertices or alter shape.
    query_surface = v.mean(axis=1)
    planes = {}; plane_audit = {}
    for extent, bounds in [('full', DOMAIN), ('near', near)]:
        for axis in [1, 2]:
            key = f'{extent}_{"y" if axis == 1 else "z"}0'
            planes[key], plane_audit[key] = plane_grid(mesh, bounds, axis, resolution)
    timings['geometry_sampling_seconds'] = time.monotonic()-start
    predictor = predictor or JointPredictor()
    predictor.encode(inputs, ux)
    validation['chunk_equivalence'] = predictor.verify_chunks(volume_xyz)
    timings.update(predictor.timing)
    start = time.monotonic()
    surface_points = attach_fields(pv.PolyData(surface_xyz), predictor.predict_raw(surface_xyz), ux)
    refined = attach_fields(refined, predictor.predict_raw(query_surface), ux, 'cell')
    volume_points = attach_fields(pv.PolyData(volume_xyz), predictor.predict_raw(volume_xyz), ux)
    volume_points.point_data['sampling_region'] = np.repeat([0, 1], 65536).astype(np.uint8)
    for plane in planes.values():
        attach_fields(plane, predictor.predict_raw(plane.points), ux)
    timings['query_seconds'] = time.monotonic()-start
    validation.update(full_sampling=full_audit, near_sampling=near_audit, planes=plane_audit,
                      surface_max_edge_m=max_edge, surface_triangles=refined.n_cells,
                      surface_edge_target_met=max_edge <= .0200001,
                      negative_sqrt_distance_fraction=float(np.mean(volume_points['sqrt_distance'] < 0)),
                      all_outputs_finite=True, no_cfd=True)
    return dict(surface=refined, geometry=mesh, surface_points=surface_points,
                volume_points=volume_points, planes=planes, inputs=inputs,
                validation=validation, timings=timings, near_bounds=near.tolist(),
                condition=dict(mach=mach, alpha_deg=alpha_deg, beta_deg=beta_deg,
                               Ux_m_s=ux, model_condition=[ux, 1.]), seed=seed)


def save_geometry_prediction(result, output):
    """Persist geometry-only outputs; explicit names cannot be confused with truth."""
    import json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for key, filename in [('surface', 'surface_prediction.vtp'),
                          ('surface_points', 'surface_random_queries.vtp'),
                          ('volume_points', 'volume_prediction.vtp'),
                          ('geometry', 'geometry_rotated.vtp')]:
        result[key].save(output/filename)
    for name, mesh in result['planes'].items():
        mesh.save(output/f'{name}_prediction.vtp')
    np.save(output/'geometry_input.npy', result['inputs'])
    metadata = {key: result[key] for key in ('condition', 'seed', 'validation', 'timings', 'near_bounds')}
    metadata['interpretation'] = 'MODEL PREDICTION ONLY; no CFD truth or accuracy validation'
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--mesh', type=Path, required=True, help='Already normalized canonical centred closed OBJ/STL')
    parser.add_argument('--mach', type=float, required=True)
    parser.add_argument('--alpha-deg', type=float, default=0.)
    parser.add_argument('--beta-deg', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; existing outputs never overwritten')
    args = parser.parse_args()
    result = infer_geometry(args.mesh, args.mach, args.alpha_deg, args.beta_deg,
                            predictor=JointPredictor(checkpoint=args.checkpoint, config=args.config), seed=args.seed)
    save_geometry_prediction(result, args.output)
