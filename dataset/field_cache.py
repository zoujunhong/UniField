"""Read per-case fields in stored physical units; transforms stay in readers."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np

# Historical identifiers only; row counts and dtypes are metadata-defined.
SURFACE_DTYPE = np.dtype([('xyz', '<f2', (3,)), ('pressure', '<f2'), ('normal', '<f4', (3,))])
VOLUME_DTYPE = np.dtype([('xyz', '<f2', (3,)), ('pressure', '<f2'), ('velocity', '<f2', (3,)), ('wall_distance', '<f4')])
DEFAULT_ROOT = Path('/data/home/zdhs0017/zoujunhong/data/DrivAerNet/cases')
_VECTOR_FIELDS = {'xyz': 3, 'normal': 3, 'velocity': 3}


@lru_cache(maxsize=8)
def manifest(root):
    path = Path(root) / 'manifest.json'
    metadata = json.loads(path.read_text())
    if metadata.get('schema') not in {'standalone_fields_v2', 'field_case_collection_v1'} or not metadata.get('complete'):
        raise ValueError(f'{path}: incomplete or unsupported field-cache manifest')
    metadata['by_name'] = {record['name']: record for record in metadata.get('records', [])}
    if len(metadata['by_name']) != metadata.get('cases'):
        raise ValueError(f'{path}: duplicate or missing cases')
    return metadata


def case_metadata(directory):
    """Return this pool's contract; legacy compact manifests remain readable."""
    directory = Path(directory)
    path = directory / 'metadata.json'
    if path.is_file():
        metadata = json.loads(path.read_text())
        if metadata.get('schema') == 'field_case_v1':
            return metadata
        if metadata.get('schema') != 'field_case_metadata_v1':
            raise ValueError(f'{path}: unsupported field metadata schema')
        # The initial cleanup added descriptive metadata to the exact compact
        # pool. Its root manifest remains the authoritative row/checksum proof.
    old_manifest = directory.parent / 'manifest.json'
    if old_manifest.is_file():
        collection = manifest(directory.parent)
        if collection['schema'] == 'standalone_fields_v2':
            record = collection['by_name'].get(directory.name)
            if record is None:
                raise ValueError(f'{directory}: absent from frozen cache manifest')
            return {'schema': 'field_case_v1', 'case': directory.name,
                    'profile': 'Surface65536_Volume200000', 'coordinate_frame': 'raw',
                    'surface': {'shape': [record['surface_points']], 'sha256': record['surface_sha256'],
                                'row_order': 'compact_surface65536_v1'},
                    'volume': {'shape': [record['volume_points']], 'sha256': record['volume_sha256']}}
    raise FileNotFoundError(f'Missing per-case field contract: {path}')


def _named_dense(array, record, path):
    fields = record.get('fields')
    if array.ndim != 2 or not isinstance(fields, dict) or not fields:
        raise ValueError(f'{path}: dense fields need explicit metadata fields/columns')
    dtype, columns = [], {}
    for name, descriptor in fields.items():
        selected = descriptor.get('columns') if isinstance(descriptor, dict) else None
        if isinstance(selected, int):
            selected = [selected]
        if not isinstance(selected, list) or not selected or any(not isinstance(i, int) or i < 0 or i >= array.shape[1] for i in selected):
            raise ValueError(f'{path}: invalid column mapping for {name!r}')
        columns[name] = selected
        dtype.append((name, array.dtype, (len(selected),)) if len(selected) > 1 else (name, array.dtype))
    named = np.empty(len(array), dtype=np.dtype(dtype))
    for name, selected in columns.items():
        named[name] = array[:, selected] if len(selected) > 1 else array[:, selected[0]]
    return named


def _append_auxiliary_fields(array, record, path, mmap_mode):
    auxiliary = record.get('auxiliary_fields', {})
    if not auxiliary:
        return array
    values, dtype = {}, list(array.dtype.descr)
    for name, descriptor in auxiliary.items():
        if name in array.dtype.names:
            raise ValueError(f'{path}: duplicate auxiliary field {name!r}')
        value = np.load(path.parent / descriptor['file'], mmap_mode=mmap_mode, allow_pickle=False)
        if len(value) != len(array):
            raise ValueError(f'{path}: auxiliary {name!r} has a different row count')
        if 'shape' in descriptor and list(value.shape) != descriptor['shape']:
            raise ValueError(f'{path}: auxiliary {name!r} shape disagrees with metadata')
        if 'dtype' in descriptor and str(value.dtype) != descriptor['dtype']:
            raise ValueError(f'{path}: auxiliary {name!r} dtype disagrees with metadata')
        if value.shape == (len(array), 1):
            value = value[:, 0]
        values[name] = value
        dtype.append((name, value.dtype, value.shape[1:]) if value.ndim > 1 else (name, value.dtype))
    named = np.empty(len(array), dtype=np.dtype(dtype))
    for name in array.dtype.names:
        named[name] = array[name]
    for name, value in values.items():
        named[name] = value
    return named


def read_fields(path, verify=False, required=(), *, mmap_mode=None):
    """Return named fields, preserving row order, count and precision.

    Optional quantities remain absent. Tasks declare their requirements so
    missing normal/area/pressure cannot silently become zeros or ones.
    """
    path = Path(path)
    if path.stem not in {'surface', 'volume'}:
        raise ValueError(f'{path}: expected surface.npy or volume.npy')
    record = case_metadata(path.parent).get(path.stem)
    if not isinstance(record, dict):
        raise ValueError(f'{path}: metadata does not declare this field pool')
    array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if 'shape' in record and list(array.shape) != record['shape']:
        raise ValueError(f'{path}: shape {array.shape} disagrees with metadata {record["shape"]}')
    if 'dtype' in record:
        actual = array.dtype.descr if array.dtype.names else str(array.dtype)
        if json.loads(json.dumps(actual)) != record['dtype']:
            raise ValueError(f'{path}: dtype {array.dtype} disagrees with metadata')
    if array.dtype.names is None:
        array = _named_dense(array, record, path)
    if array.ndim != 1 or 'xyz' not in array.dtype.names:
        raise ValueError(f'{path}: fields must be a row array containing xyz')
    array = _append_auxiliary_fields(array, record, path, mmap_mode)
    for name in array.dtype.names:
        descriptor = record.get('fields', {}).get(name, record.get('auxiliary_fields', {}).get(name, {}))
        components = descriptor.get('components', _VECTOR_FIELDS.get(name)) if isinstance(descriptor, dict) else _VECTOR_FIELDS.get(name)
        expected = (len(array), int(components)) if components and int(components) > 1 else (len(array),)
        if array[name].shape != expected or not np.issubdtype(array[name].dtype, np.number):
            raise ValueError(f'{path}: invalid {name} field shape/dtype')
    for name in required:
        if name not in array.dtype.names:
            raise KeyError(f'{path}: required field {name!r} is unavailable in this pool')
    if verify:
        expected = record.get('sha256')
        if expected is None:
            raise ValueError(f'{path}: no SHA256 recorded for verification')
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'{path}: checksum mismatch')
    return array


def load_surface_fields(directory):
    array = read_fields(Path(directory) / 'surface.npy', required=('pressure', 'normal'))
    return np.column_stack((array['xyz'], array['pressure'])), np.ascontiguousarray(array['normal'])


def load_surface(path, **kwargs):
    path = Path(path)
    if (path.parent / 'metadata.json').exists() or (path.parent.parent / 'manifest.json').exists():
        array = read_fields(path, required=('pressure',), **kwargs)
        return np.column_stack((array['xyz'], array['pressure']))
    # Explicit old flat readers retain their documented dense format.
    return np.load(path, allow_pickle=False, **kwargs)


def load_volume(path, **kwargs):
    path = Path(path)
    if (path.parent / 'metadata.json').exists() or (path.parent.parent / 'manifest.json').exists():
        array = read_fields(path, required=('pressure', 'velocity'), **kwargs)
        return np.column_stack((array['xyz'], array['pressure'], array['velocity']))
    return np.load(path, allow_pickle=False, **kwargs)


def load_normals(directory):
    return np.ascontiguousarray(read_fields(Path(directory) / 'surface.npy', required=('normal',))['normal'])


def load_wall_distance(directory):
    return np.ascontiguousarray(read_fields(Path(directory) / 'volume.npy', required=('wall_distance',))['wall_distance'])
