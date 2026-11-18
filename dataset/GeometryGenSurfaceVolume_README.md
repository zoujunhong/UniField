# GeometryGen normalized joint-field data

Dataset root: `~/zoujunhong/data/GeometryGen`.

- `results/case_000001`–`case_004000`: two complete 2000-case CFD batches, including surface and volume VTK, geometry, conditions and aerodynamic coefficients.
- `cache/surface_volume`: unified target specification is 8192 input points, up to 16384 surface points and 65536 volume points, without source indices. On 2026-10-26 a rebuild was requested for local cases 1–4000; use `progress.json`, `index.json` and the compact rebuild report to check completion and current counts. The pre-rebuild cohort had 12299 accepted records and 2000 uncached VTK cases.
- Imported exports retain 8192 xyz+normal input points, up to 16384 surface and 65536 volume query points, and no source indices. Their normalization/column ordering match the full-point cache; sampling remains random from each case's available query pool.
- Total inventory is 14310 simulations: 4000 with local VTK and 10310 cache-only exports. Imported cases do not have source VTK or mesh connectivity; VTK-based visualization cannot treat them as full CFD archives.
- `campaigns/`: campaign provenance copies. Original campaign/attempt logs remain under AutoCFD.
- `manifest.json`: merged case inventory and cache coverage.
- Old raw-case paths and the old cache root remain compatibility symlinks. No duplicate CFD files are stored.

Use `config/dataset/GeometryGenSurfaceVolume/surface_volume.py` or the existing joint-field experiment. Defaults can be overridden using `GEOMETRYGEN_ROOT`, `GEOMETRYGEN_JOINT_SOURCE_ROOT`, and `GEOMETRYGEN_JOINT_CACHE_ROOT`. FPS inputs, normalization and full-surface wall-distance reference remain unchanged. Query pools are randomly subsampled without replacement with deterministic seeds; smaller pools retain all points. Source indices are removed on compact rebuild. The older `GeometryGenSurface` dataset is a different surface-only cohort and is not redirected.

To explicitly extend the cache to all raw cases, run on an allocated compute node from the repository root:

```bash
python -m dataset.GeometryGenSurfaceVolume --workers 2
```

This reuses existing records, preserves compact imported cases, and builds missing VTK cases; it may take substantial time and storage. Import does not start this job. The default dataset now includes imported records; the pre-import index is backed up at `campaigns/import_1026/index_before_import.json`. For reproducing a historical run, use its historical case list rather than the expanded default cohort. Imported pressure-warning flags are preserved; importing alone does not impose a new numerical quality filter.
