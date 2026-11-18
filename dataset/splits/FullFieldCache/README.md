# 列车与导弹全流场 cache

> 2026-10-26：用户确认传输完成后，已按要求删除本地两版缓存目录及交付 ZIP、SHA256 文件。
> 下文保留历史生成结果与格式说明；本地仅保留清单、脚本和分析报告，原始 CFD 数据未删除。
> 删除记录见 `temp/20261026_114056_zip_full_field_cache/visualization/deletion_after_transfer.json`。

## 精简交付 ZIP（无索引、无表面面积）

交付包：`surface32768_volume100000_fields.zip`；校验文件：同名 `.zip.sha256`。
ZIP 大小 **33300935302 字节（33.30 GB，31.01 GiB）**。
解压后顶层目录为 `surface32768_volume100000_fields/`。
每例只有 `surface.npy`、`volume.npy`，全包共 **15555 个文件**：
7776 × 2 个数组和统一的 `cases.csv`、`schema.json`、`README.md`。

点数、点顺序及 float32 精度沿用紧凑缓存；两域表面删除 `area` 列，
不包含 `surface_indices.npy` / `volume_indices.npy` 或逐例 JSON。
列车表面 13 列，导弹表面 8 列；空间字段不变。
`cases.csv` 集中记录相对文件路径、工况、划分、导弹质量标签和 Cd/Cl 末 500 条统计。
力系数的参考面积 `coefficient_Aref` 是归一化常数，保留在 CSV 中，与删除的逐面面积不同。
列车缓存没有力系数，对应单元格为空。
实际列顺序以包内 `schema.json` 为准；下文带 `area` 和索引的布局仅描述本地保留的历史缓存。

重新打包脚本及校验结果：`temp/20261026_114056_zip_full_field_cache/`。
本地历史缓存保留用于追溯，交付包不包含它们的索引、面积及审计文件。

## 当前紧凑版本

缓存目录：`data/FullFieldCache/surface32768_volume100000_fp32/`。
每例表面最多 **32768** 点、空间最多 **100000** 点；不足上限全部保留，无放回采样、不补点。
字段保持 float32，原始行号索引经范围与回读检查后无损存为 int32。
工况、划分、排除项、物理量单位及列顺序与下述旧版相同。

2026-10-26 已完成 **7776 例**（列车 497、导弹 7279）；本批所有工况均达到上述新点数上限。
数组及索引总计 **44935053952 字节（44.94 GB）**，较旧版 143.38 GB 减少 **68.66%**。
全量文件回读与所选父缓存行逐值相等，int32 索引转换无损；数据划分及排除项核对一致。

```bash
python -m dataset.cache.compact_full_field --workers 12
python temp/20261026_113239_compact_full_field_cache/summarize.py
```

新版本从旧 cache 无放回选取行，组合索引后仍可追溯到原始 VTK/NPZ（列车空间仍为旧空间 cache）行号。
`record.json` 记录父缓存来源，字段文件独立保存，可直接使用 `np.load`。
`validation.json` 保存全量写入后回读结果，`float16_audit.json` 保存两域各字段的溢出、下溢、量化误差；
`float16_cases.csv` 保存每例异常计数。float16 试验是直接转换，不做缩放或截断；溢出元素单独计数，误差指标只计算转换后仍有限的元素。

原始绝对压力和部分 omega 无法直接以 float16 表示，因此没有把含 inf 的试验数组发布为正式缓存。
全量直接转换审计：7279 个导弹工况均出现压力溢出，6809 个工况出现空间 omega 溢出；
列车表面压力、表面 omega、空间 omega 分别在 2、19、6 个工况中出现溢出。
小面积、k、nut 等还存在非零值归零；坐标有精度损失。
逐字段计数及误差见 `float16_audit.json`，详细分析和实际 float16 文件试验见
`temp/20261026_113239_compact_full_field_cache/report.md`。

另经原始 NPZ 核对，`g172/M020/Ap10`（C_unconverged）的 omega 原本即含约
`1.12695e36` 的极端值；它在 float32 中有限，不能据有限性检查认定物理质量合格。
本次保留既有质量标签和纳入规则。
原版较大 cache 当前保留。

## 原版 100000 / 300000

旧缓存目录：`data/FullFieldCache/surface100000_volume300000/`。

- 列车：原始 CRH450 200 例、Maglev 300 例的清单，沿用既有审核结果，缓存 497 例（198 / 299）。两例表面缺失、一例已隔离，原因及路径见 `excluded.json`。
- 导弹：`/data/home/zdhs0017/zoujunhong/data/rocket/extracted/` 的 7281 例，沿用人工排除 `g086/M020/Ap10`、`g170/M020/Am02` 的结果，缓存 7279 例。保留 A/B/C 原始质量标签。
- 每例表面至多 **100000** 点、空间至多 **300000** 点。低于上限保留全部有效点，不重复补齐。超过上限按原始行等概率、无放回采样；不做空间裁剪。保存 float32。
- 坐标为原始物理坐标（米），速度为 m/s；不进行旋转、尺度拟合或训练归一化。列车 `p` 是运动学压力 m²/s²，导弹压力是源文件的绝对压力 Pa，不能混用数值定义。
- 保留原有 train/val/test 划分和导弹末 500 条 Cd/Cl 算术均值标签。列车原有留出集标为 `test`。

2026-10-26 已完成全部 7776 例并通过全量回读校验：

| 域 | 可用工况 | 每例表面点数范围 | 表面不足 10 万、全部保留的工况 | 每例空间点数 |
| --- | ---: | ---: | ---: | ---: |
| 列车 | 497 | 49493–100000 | 86 | 300000 |
| 导弹 | 7279 | 37342–100000 | 3927 | 300000 |

数组及索引合计 143375183544 字节（约 143.4 GB），不含 JSON/CSV 元数据。
所有数组有限、点数符合上限、索引无重复，几何分组未跨数据划分；6 个代表工况的源字段抽查逐值相等。

## 文件布局

```text
surface100000_volume300000/
  contract.json             # 上限、seed、采样规则及清单来源
  index.json                # ready=true 才表示全量完成，含全部逐例记录
  cases.csv                 # 点数和划分汇总
  excluded.json             # 5 个既有排除/缺失记录
  progress.json             # 生成进度及错误
  validation.json           # 全量回读与独立源字段比对结果
  train/{CRH450,Maglev}/{case}/
  rocket/{geometry}/{Mach}/{AoA}/
    surface.npy
    volume.npy
    surface_indices.npy
    volume_indices.npy
    record.json
    source_metadata.json    # 仅导弹，源工况信息
    source_field_data.json  # 仅导弹，源字段和单位定义
```

每个域、每种字段的列顺序由 `record.json` 的 `surface_columns` / `volume_columns` 明确指定：

| 域 | 数组 | 列顺序 |
| --- | --- | --- |
| 列车 | surface | x, y, z, p, Ux, Uy, Uz, k, omega, nut, nx, ny, nz, area |
| 列车 | volume | x, y, z, p, Ux, Uy, Uz, k, omega, nut, surface_distance |
| 导弹 | surface | x, y, z, pressure, Cp, nx, ny, nz, area |
| 导弹 | volume | x, y, z, p, Ux, Uy, Uz, k, omega, nut |

导弹表面仅保存源文件提供的压力及几何量。`area` 是原面面积 m²；抽样后的面积未重新加权。列车 `surface_distance` 沿用原缓存的完整表面面心最近距离（米）。

列车表面从 `surface.vtk` 重建，以修复旧缓存不足 10 万点时重复采样的问题。空间复用已逐例检查有限性、数量及坐标唯一性的 `volume_random_300000_fp32`；原始列车网格均超过 30 万单元。其 `volume_indices.npy` 指向**原空间 cache 的行号**，不是原始 CFD 单元编号，来源文件写在 `volume_index_reference`。导弹两种索引均指向源 NPZ 的原始行号；表面先应用源 `valid_mask`，空间保留全计算域作为候选。`surface_source_points` / `surface_available_points` 区分原始面数与有效面数；列车另存 `volume_raw_mesh_points`。

## 读取

```python
import json
from pathlib import Path
import numpy as np

root = Path("data/FullFieldCache/surface32768_volume100000_fp32")
index = json.loads((root / "index.json").read_text())
assert index["ready"]
r = index["cases"][0]
case_dir = (root / "train" / r["family"] / r["case"]
            if r["domain"] == "train" else root / "rocket" / r["case"])
surface = np.load(case_dir / "surface.npy", mmap_mode="r")
volume = np.load(case_dir / "volume.npy", mmap_mode="r")
velocity = volume[:, [r["volume_columns"].index(k) for k in ("Ux", "Uy", "Uz")]]
```

## 生成与校验

在 n30152 的 zoujunhong 环境中运行：

```bash
python -m dataset.cache.full_field --workers 12
python temp/20261026_100000_full_field_cache/verify.py
```

生成器支持中断后重跑，检查源文件 size/mtime 和缓存参数，逐例 `record.json` 在数组原子写入完成后才生成。全部完成才输出 `index.json`。正式清单快照在 `manifests/train500.csv`、`manifests/rocket7281.json`，分别来自 `temp/20261025_155815_training500_mesh_statistics/visualization/per_case.csv` 与 `temp/20261013_195027_rocket_joint_scratch/cache/manifest.json`。

此次运行的日志、独立源数据比对和全量回读结果位于 `temp/20261026_100000_full_field_cache/`。这些点云 cache 适合字段监督；原始网格拓扑仍保存在源数据中。
