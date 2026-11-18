# 数据与缓存

原始 CFD 数据在 `/lustre/home/zdhs0017/data/`，公共 DrivAerNet++ 原始文件保持 `/lustre/group/project1/CFD/DrivAerNet++/`。所有缓存位于个人 `~/zoujunhong/data/`，DrivAerNet++ 正式缓存根为 `~/zoujunhong/data/DrivAerNet/`。源码仓库只保存读取/构建程序、冻结 split、schema 和小型元数据。

| 数据集 | 原始数据（相对 /lustre/home/zdhs0017/data） | 个人缓存 |
|---|---|---|
| DrivAerNet++ | 公共项目目录（例外） | DrivAerNet/cases/<case>/ |
| HighSpeedTrain | HighSpeedTrain/ | HighSpeedTrain/cases/<case>/ |
| AircraftWing / FullAircraft | Flight/ | AircraftWing/cases/、FullAircraft/cases/ |
| ThingiCFD | ThingiCFD/ | ThingiCFD/cases/ |
| GeometryGen 表面 | GeometryGen/CanonicalSurface/ | GeometryGen/surface/cases/ |
| GeometryGen 联合场 | GeometryGen/results/ 及原始 campaign | GeometryGen/volume/cases/ |
| RocketCFD 表面 | RocketCFD/source/ | RocketCFD/cases/ |
| Rocket 联合场 | rocket/extracted/ | RocketCFD/volume/cases/ |

每例一个文件夹，含 surface.npy、可用时的 volume.npy 与 metadata.json。JSON 记录字段顺序、单位/压力约定、工况、已有力系数、来源和 split；input.npy、原始索引、几何拓扑等辅助文件必须记录语义。缓存不等于原始数据：有限采样、坐标变换和精度必须显式说明，禁止把 Cp、Pa 和运动学压力互相当作同一标签。

DrivAerNet++ 的 `DrivAerNet/cases/<case>/` 保存原始物理量；主池保留现有实验的混合精度和行序，另保留的随机/空间均匀100000点表面池位于每例 `pools/<profile>/`，通过 `cache_profile` 显式选择。新建池采用 FP32 named fields，字段数量和点数不与模型绑定。详见 [CACHE_FORMAT.md](CACHE_FORMAT.md)。固定测试点在个人 `DrivAerNet/` 下的 test_point_sets、query_normals、shared_points_set00；冻结划分在 dataset/splits/DrivAerNet。

## 读取接口

组合 dataset 返回：

```python
surface_input, surface_query, surface_target, volume_query, volume_target, cond, route, coefficient_target
```

原生模型需要时可附加第九项 geometry metadata，例如 query_normals。未监督域通常返回空目标；ABUPT 的双域锚点上下文可以有 query 而没有监督量。dataset/surface 和 dataset/volume 管理采样/标签，dataset/cache 管理构建，dataset/utils 放公共解析和几何工具。

验证通过 dataset/validation.py 统一设置：完整有效划分，各受监督域一组10000随机 query；在任何归一化前筛除物理 Cp < -10 的整行。其他物理量不额外设阈值。几何输入不按标签筛选。训练及测试不自动继承验证阈值，测试保留固定点集。

缓存构建是独立任务，先依 AGENTS 选择没有训练的计算节点，再通过 `python -m dataset.cache.<builder>` 执行。常规评估读取缓存；显式全点评估直接读取原始 VTK。常规模式缺缓存明确报错，不静默补建。过去交付后按用户要求删除的 FullFieldCache 大数组不重新伪造，现存 manifest 与构建说明在 [splits/FullFieldCache](splits/FullFieldCache/README.md)。

高铁主 FP32 随机池保存在 `HighSpeedTrain/cases/<case>/surface.npy`（100000 点）与 `volume.npy`（300000 点）。FP16、不同采样方式/点数、triplane 和 Gaussian 等表示保存在同例 `auxiliary/<profile>/data.npy` 或 `data.npz`，例如 `surface_random_100000_fp16`、`triplane_32x128x128_fp32`。每例 `metadata.json` 的 `auxiliary` 索引记录文件、数组 shape/dtype；共享 occupancy decoder 仅保存于 `HighSpeedTrain/shared/<CRH450或Maglev>/<profile>/_shared_occupancy_decoder.pt`，由索引引用。构建器和 reader 使用统一路径函数，不保留旧平铺目录链接。只有辅助表示、缺主场的案例标为 `cache_only`，不进入冻结的 497 例主训练 cohort。

GeneratedHighSpeedTrain 的历史 surface/query/support pool 同样在 `cases/<case>/auxiliary/<profile>/`，不覆盖主 Cp 表面数组。`fv_face_shards_v1` 属于 PINN 的局部物理采样缓存，metadata 明示 owner、partial 与完整性；第二例只有 15 shards、缺原元数据，不能当作完整训练缓存使用。

RocketCFD 表面的新默认 train/val/test 使用已发布的互斥子集：train 保持 train；val 为全部 aoa_interpolation；test 为 aoa_extrapolation、geometry_generalization、ood_geometry 的并集。此处不再额外做 hash 子划分。历史结果保留原 subset 选择。联合场 RocketSurfaceVolume 有其独立按几何冻结的 train/val/test manifest，不能混用两者划分。
