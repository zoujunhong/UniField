# 模型目录

当前支持七个模型族。模型定义、官方依赖和执行优化均在本目录，正式运行不得导入 `temp/` 源码。

| 模型族 | 配置规模 | 实现 |
| --- | --- | --- |
| ABUPT | Medium | 保留主实验使用的官方 AB-UPT，192 维、3 头 |
| SMART | Medium | 保留主实验使用的官方 SMART，256 维、6 组 encoder/decoder |
| TripNet | Medium | triplane encoder + query decoder |
| Transolver3 | Medium / Large | 官方 Transolver-3，分别 12 / 24 层 |
| UniField | Small / Medium / Large | 原 UniFieldV2，参数键保持不变 |
| VolumeUniField | Small / Medium / Large | 原 UniField_cross_attention |
| DSAPT | TinyG2222L2111 | Dual-Scale Aerodynamic Point Transformer |

DSAPT 的 `G2222` / `L2111` 表示两条分支的四次 pooling stride，不是深度；Tiny backbone 的 stage depths 是 `(2,2,2,6,2)`，channels 是 `(18,36,72,144,216)`。其 query decoder 为 216 维、6 头、6 层。RoPE、IDW、LoRA 系数和训练配方在实验副配置记录，不扩展模型族名称。

## 目录与接口

- `Encoder/`、`Decoder/`：UniField、VolumeUniField、TripNet 的可组合模块。
- `Baseline/`：官方基线适配器；`third_party/`：随主实验迁入的官方依赖与许可证。
- `DSAPT/`：双分支网络、decoder、IDW 与已采用的执行优化。精确续训接口保留独立 encoder/decoder state_dict。
- `utils/`：共享 attention、FFN、采样、插值和权重装载逻辑。
- `build_model` 接受组合配置 `encoder_name/decoder_name`，或完整模型配置 `native_model_name/native_model_kwargs`。

统一入口为 `forward(surface_pos, query_pos, cond, routes=..., normals=..., area=..., geometry_meta=...)`，`forward_test` 同样接受这些参数。query 最后一维为类型标记，0 表面、1 体场。官方基线和 DSAPT 输出 `[p,Ux,Uy,Uz]`，表面只有第 0 通道有效；不同物理量的监督切片由 task 配置明确指定。

## 数据契约

- 官方 ABUPT、SMART、Transolver3 使用原始物理坐标。DrivAerNet dataset 设置 `coordinate_frame='raw'`。ABUPT/SMART 适配器内部执行各自主实验的坐标归一化，Transolver3 直接使用 raw xyz。
- Transolver3 与 DSAPT 的表面查询法向必须与 query 逐点对齐，由 `geometry_meta['query_normals']` 提供；不得用不对应的 encoder 法向代替。
- ABUPT 保留原 batch=1 几何接口，表面预测也需要体域 anchor/context。模型间不得用近似重写版本加载官方 checkpoint。
- DSAPT 的 encoder 与 query 使用相同居中和固定 global scale；`geometry_meta['point']` 包含官方 GridSample 的 coord、grid_coord、feat、offset。目前每 GPU 一例。专用 dataset/训练实现位于 `dataset/dsapt/` 与 `utils/training_dsapt.py`，仍由根 `train.py --config ...` 启动。
- 新模型配置仅保存结构和固定规模。监督字段、训练采样数量、归一化、优化器、损失、GPU 数等改动在 experiment 或对应 log 副配置显式记录。

## 权重兼容

UniFieldV2 → UniField 和 UniField_cross_attention → VolumeUniField 只是正式命名整理，没有重排保留架构的 state_dict。ABUPT/SMART 已改为主实验使用的官方实现，原 clean-room 权重不与官方模型兼容。官方基线的 `load_route_compatible_state_dict` 可读取历史 `model` / `models` 容器；Transolver3 自动映射独立分支及旧 `net.` 包装前缀，严格检查全部张量。也可通过 `load_branch_state_dict` 单独装载分支。DSAPT 可严格装载历史独立 encoder/decoder 字典，EMA 使用完整普通权重叠加 shadow 和 buffers。

更早的 CAquery/tfquery 等同名旧模型须以历史来源和实际键结构识别；目录更名不构成兼容性证明。迁移审计中保留了删除前的源码与配置。
