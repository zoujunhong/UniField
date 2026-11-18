"""Independent, float64 OpenFOAM-style geometric cell centres.

Never call vtkCellCenters: its polyhedron parametric centre may lie far outside
the cell. Field row indices and original vertices are not changed here.
"""
from __future__ import annotations

import numpy as np

FACE_TEMPLATES = {
    10: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    12: ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
         (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)),
    13: ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    14: ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}
POINT_COUNTS = {10: 4, 12: 8, 13: 6, 14: 5}


def face_geometry_batch(vertices):
    """Fixed-arity polygon batch -> projected-area centroid and area vector."""
    vertices = np.asarray(vertices, dtype=np.float64)
    avg = vertices.mean(axis=1)
    p = vertices - avg[:, None]
    q = np.roll(p, -1, axis=1)
    cross = np.cross(q - p, -p)
    total = cross.sum(axis=1)
    normal = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-300)
    weights = np.einsum('nvi,ni->nv', cross, normal)
    denom = weights.sum(axis=1)
    centre = avg + np.sum(weights[..., None] * (p + q), axis=1) / np.maximum(3 * denom[:, None], 1e-300)
    return centre, total / 2


def _ragged_face_geometry(points, offsets, connectivity, chunk_size=32768):
    centres = np.empty((len(offsets) - 1, 3), dtype=np.float64)
    areas = np.empty_like(centres)
    sizes = np.diff(offsets)
    for arity in np.unique(sizes):
        if arity < 3:
            raise ValueError('Face has fewer than three vertices')
        ids = np.flatnonzero(sizes == arity)
        for start in range(0, len(ids), chunk_size):
            sub = ids[start:start + chunk_size]
            vertices = points[connectivity[offsets[sub, None] + np.arange(arity)]]
            centres[sub], areas[sub] = face_geometry_batch(vertices)
    return centres, areas


def _finish(estimate, moment, volume3, closure, area_sum, lower, upper):
    safe = np.where(np.abs(volume3) > 1e-300, volume3, 1)
    centres = estimate + .75 * moment / safe[:, None]
    diag = np.linalg.norm(upper - lower, axis=1)
    tolerance = np.maximum(1e-6, 1e-4 * diag)
    valid = (np.isfinite(centres).all(axis=1)
             & (np.abs(volume3) > 1e-12 * diag**3)
             & (np.linalg.norm(closure, axis=1) <= 1e-6 * np.maximum(area_sum, 1e-300))
             & (centres >= lower - tolerance[:, None]).all(axis=1)
             & (centres <= upper + tolerance[:, None]).all(axis=1))
    return centres, np.abs(volume3) / 3, valid


def volume_cell_geometry(mesh, chunk_size=32768):
    """Return (centres, volumes, valid_mask, audit), all in original row order."""
    from vtk.util.numpy_support import vtk_to_numpy

    points = np.asarray(mesh.points, dtype=np.float64)
    offsets = np.asarray(mesh.offset)
    connectivity = np.asarray(mesh.cell_connectivity)
    types = np.asarray(mesh.celltypes)
    unsupported = set(np.unique(types)) - set(FACE_TEMPLATES) - {42}
    if unsupported:
        raise ValueError(f'Unsupported volume cell types: {unsupported}')
    centres = np.empty((mesh.n_cells, 3), dtype=np.float64)
    volumes = np.empty(mesh.n_cells, dtype=np.float64)
    valid = np.zeros(mesh.n_cells, dtype=bool)
    old_bad = 0
    old = np.asarray(mesh.cell_data.get('cell_center_world_m', np.full_like(centres, np.nan)))

    def vertex_bounds(ids):
        counts = offsets[ids + 1] - offsets[ids]
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        flat = np.repeat(offsets[ids] - starts, counts) + np.arange(counts.sum())
        vertices = points[connectivity[flat]]
        return np.minimum.reduceat(vertices, starts, axis=0), np.maximum.reduceat(vertices, starts, axis=0)

    def assign(ids, result, lower, upper):
        nonlocal old_bad
        centres[ids], volumes[ids], valid[ids] = result
        tol = np.maximum(1e-6, 1e-4 * np.linalg.norm(upper - lower, axis=1))
        old_bad += int(np.count_nonzero(~np.isfinite(old[ids]).all(axis=1)
            | (old[ids] < lower - tol[:, None]).any(axis=1)
            | (old[ids] > upper + tol[:, None]).any(axis=1)))

    for kind, templates in FACE_TEMPLATES.items():
        ids = np.flatnonzero(types == kind)
        for start in range(0, len(ids), chunk_size):
            sub = ids[start:start + chunk_size]
            vertices = points[connectivity[offsets[sub, None] + np.arange(POINT_COUNTS[kind])]]
            geom = [face_geometry_batch(vertices[:, face]) for face in templates]
            fc = np.stack([x[0] for x in geom], axis=1)
            sf = np.stack([x[1] for x in geom], axis=1)
            est = fc.mean(axis=1)
            weights = np.sum(sf * (fc - est[:, None]), axis=2)
            lower, upper = vertices.min(axis=1), vertices.max(axis=1)
            result = _finish(est, np.sum(weights[..., None] * (fc - est[:, None]), axis=1),
                             weights.sum(axis=1), sf.sum(axis=1),
                             np.linalg.norm(sf, axis=2).sum(axis=1), lower, upper)
            assign(sub, result, lower, upper)

    ids = np.flatnonzero(types == 42)
    if len(ids):
        faces = mesh.GetPolyhedronFaces()
        locations = mesh.GetPolyhedronFaceLocations()
        fo = vtk_to_numpy(faces.GetOffsetsArray())
        fi = vtk_to_numpy(faces.GetConnectivityArray())
        co = vtk_to_numpy(locations.GetOffsetsArray())
        ci = vtk_to_numpy(locations.GetConnectivityArray())
        if len(co) != mesh.n_cells + 1:
            raise ValueError('Unexpected VTK polyhedron face location layout')
        face_centres, face_areas = _ragged_face_geometry(points, fo, fi, chunk_size)
        for start in range(0, len(ids), chunk_size):
            sub = ids[start:start + chunk_size]
            counts = co[sub + 1] - co[sub]
            if np.any(counts < 4):
                raise ValueError('Polyhedron has fewer than four faces')
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            flat = np.repeat(co[sub] - starts, counts) + np.arange(counts.sum())
            refs = ci[flat]
            fc, sf = face_centres[refs], face_areas[refs]
            est = np.add.reduceat(fc, starts, axis=0) / counts[:, None]
            delta = fc - np.repeat(est, counts, axis=0)
            weights = np.einsum('ni,ni->n', sf, delta)
            lower, upper = vertex_bounds(sub)
            result = _finish(est, np.add.reduceat(weights[:, None] * delta, starts, axis=0),
                             np.add.reduceat(weights, starts),
                             np.add.reduceat(sf, starts, axis=0),
                             np.add.reduceat(np.linalg.norm(sf, axis=1), starts), lower, upper)
            assign(sub, result, lower, upper)
    audit = {'cells': int(mesh.n_cells), 'old_centres_outside_own_bbox': old_bad,
             'new_invalid_cells': int((~valid).sum()),
             'method': 'float64 OpenFOAM-style projected face centroids and volume-weighted face pyramids'}
    return centres, volumes, valid, audit


def surface_cell_geometry(mesh):
    from vtk.util.numpy_support import vtk_to_numpy
    polys = mesh.GetPolys()
    if polys.GetNumberOfCells() != mesh.n_cells:
        raise ValueError('Expected polygon-only body surface')
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    centres, vectors = _ragged_face_geometry(np.asarray(mesh.points, dtype=np.float64), offsets, connectivity)
    areas = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(centres).all(axis=1) & np.isfinite(areas) & (areas > 0)
    return centres, areas, valid
