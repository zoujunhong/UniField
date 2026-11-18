# DrivAerNet++ 通用缓存约定

正式位置固定为 `~/zoujunhong/data/DrivAerNet/`，仅原始数据保留 `/lustre/group/project1/CFD/DrivAerNet++/`。代码直接使用新缓存路径，旧位置不保留软链接。所有数据集缓存均放 `~/zoujunhong/data/`，遵循每例目录、物理语义显式、采样归一化由 reader 控制的原则。

```text
DrivAerNet/
  cases/
    <case>/
      surface.npy
      volume.npy
      metadata.json
      pools/
        <profile>/
          surface.npy             # 仅有此域时保留此文件
          volume.npy              # 仅有此域时保留此文件
          metadata.json
          <必要的原始索引或对齐辅助量>.npy
      auxiliary/
        <geometry_profile>/       # 拓扑、几何层次、原始顶点索引等
    manifest.json                 # 默认主池的冻结 cohort
  metadata/                       # 数据集级清单、统计、来源记录
  test_point_sets/                 # 历史十套固定 query
  query_normals/
  shared_points_set00/
```

## 数组和物理语义

新建缓存使用无 pickle 的 NPY，推荐 FP32 structured dtype（按名称访问字段）。每个字段有自己的精度，点数不写死。旧 dense NPY 无须为整理重编码，但必须在 metadata 中声明每个字段的列号。`dataset.field_cache.read_fields` 返回统一的 named structured array；保留原行序、实际精度与物理数值。

| 字段 | 维度 | 存储含义 |
|---|---:|---|
| xyz | 3 | 原始坐标，m；不居中、不缩放 |
| pressure | 1 | DrivAerNet++ 运动学表压，m²/s² |
| velocity | 3 | 速度，m/s |
| normal | 3 | 与本池表面行一一对齐的法向；缺失则不提供 |
| area | 1 | 面积，m²；必须区分原始面/顶点面积与采样后积分权重 |
| wall_distance | 1 | 距离，m；注明最近顶点距离、三角面距离或其他定义 |

压力系数由 `(p-p_ref)/(0.5*U_ref**2)` 得到，当前数据 `p_ref=0, U_ref=30 m/s`。存储不做标准化，不把 Cp 与运动学压力混用。几何归一化、输出量选择、训练采样数量和验证极值筛选在 dataset/config 中指定。

`metadata.json` 的 schema 为 `field_case_v1`，至少记录 case、原始坐标系、压力定义和各可用域的 file、shape、dtype、fields。fields 为名字到单位/分量数的映射；dense 存储额外提供 `columns`。记录已知工况、Cd、来源、采样规则/seed、原始索引和已有 SHA256。辅助数组须明确与哪个点池和行序对应。不得把另一个点池的 normal、area、distance 拼接过来。

缺少所请求的 pressure、normal、area 等字段时直接报错，不补零、不补一。主池没有面积，不可直接用于面积积分。`auxiliary` 几何池即使点数也是65536，也不能据此认为与主池行序相同。

已退役几何辅助池的坐标/面积约定存于维护档案，不能把其旧索引当成现存数据。当前训练坐标统计文件仍在 `metadata/coordinate_statistics_train_center_x.json`。

## 现有点池

主池保留已完成实验的 **8119例，65536表面点、200000体场点**。xyz/p/U 为原始 FP16，法向和距离为 FP32；整理不改变任何数组字节。有限点数与量化是历史数据能力限制，不能通过转为 FP32 恢复丢失精度。

| cache_profile | 内容 |
|---|---|
| `None`（默认） | 上述历史主池，表面 p/normal，体场 p/U/distance |
| `SurfaceRandom100000_FP16` | 10万随机表面点及已有属性，依该池 metadata 读取 |
| `SurfaceUniform100000_FP16` | 10万空间均匀表面点及已有属性 |

当前只保留上述三组。主池体场明确包含 `pressure`、三分量 `velocity`、`wall_distance`。两个表面池各覆盖8129例；额外10例不混入8119例主池的冻结划分。一旦给定 ids/split，缺任何请求案例或监督域都报错。compact 的孤立顶点排除索引只属于该行序，不能套到其他点池。

原生 FP32 大池、300000 体场池及三种几何层次缓存已按用户要求删除；历史规格和来源在维护档案中保留。旧评估如果依赖被删除的精确点序，需要从原数据重建；不能以主池新结果冒充旧指标。常规测试读取现存缓存，显式全点测试直接读取原始 VTK，不再为全点测试长期保留大点池。

配置字段示例（在默认配置上覆盖；模型输入也须与所选字段匹配）：

```python
CONFIG['dataset_kwargs']['cache_root'] = '/data/home/zdhs0017/zoujunhong/data/DrivAerNet/cases'
CONFIG['dataset_kwargs']['cache_profile'] = None  # 默认主池
```

表面和体场还可分别指定点池，例如100000点表面池和默认200000点体场池：

```python
CONFIG['dataset_kwargs']['surface_cache_profile'] = 'SurfaceRandom100000_FP16'
CONFIG['dataset_kwargs']['volume_cache_profile'] = None
```

域级选项未指定时继承 `cache_profile`；上例公共profile为None，因此体场使用主池。混合池的表面和体场是独立采样，不能据此假定行号相互对应。实际 config 可能按多个 dataset/stage 组织，覆盖对应的 kwargs 即可。修改输入字段时，同步 dataset、task 与模型输入通道。选择模型所需的字段和点池，不把模型名称写入缓存路径。当前 DSAPT 接续任务保持默认主池，保证数据协议不变。

## 构建与扩展

构建器是独立非训练任务，按 AGENTS 使用空闲计算节点。需要重建时显式指定目标和采样参数；生成独立 FP32 点池、保留原始行索引，并同时读取体场 p/U。下例仅说明重建接口，不是要求额外保留一份缓存；不得覆盖正在用于接续训练的主池。

```bash
python -m dataset.cache.drivaernet \
  --cache-profile rebuild_surface65536_volume200000_fp32 \
  --surface-cache-points 65536 --volume-cache-points 200000 \
  --include-wall-distance --include-topology --num-workers 4
```

点数设0表示保留该域全部原始行。变更点数、seed或采样方式应使用新的 profile；同名池参数不符会报错。重启部分构建保持按案例/域固定的采样序列。构建器支持 point/cell 场关联，必要时明确记录 cell-to-point 插值；拓扑三角形索引指向独立 geometry_points，绝不指向随机监督点。

这个格式覆盖点模型、Transformer和几何编码模型的场数据需求；使用网格连接、积分或有限体积算子的任务还需对应的拓扑、积分权重、owner/neighbour等附加数据，不能从通用点场缓存推定其存在。有限体积物理逻辑由 PINN 仓库维护。
