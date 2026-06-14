# Config 使用说明

主线入口只接受一个 config 地址：

```bash
python train.py --config config/experiment/drivaernet_surfaceP_volumeU.py
python test.py --config config/experiment/drivaernet_surfaceP_volumeU.py
python visualization.py --config config/experiment/drivaernet_surfaceP_volumeU.py
```

所有 config 文件都是 Python 文件，并且只需要导出一个 `CONFIG` 字典：

```python
CONFIG = {...}
```

命令行参数不会覆盖 config 内容；新实验建议复制一个 config 文件后修改。

## 目录规则

```text
config/
  dataset/<DatasetName>/<Option>.py
  model/<ModelName>/<Size>.py
  experiment/<ExperimentName>.py
```

`experiment` config 里不写完整路径，而是写名称：

```python
"dataset": "DrivAerNet",
"dataset_option": "surface_volume",
"model": "UniField_cross_attention",
"model_size": "Medium",
```

加载器会自动解析为：

```text
config/dataset/DrivAerNet/surface_volume.py
config/model/UniField_cross_attention/Medium.py
```

## Dataset Config

单数据集：

```python
CONFIG = {
    "name": "DrivAerNet",
    "kwargs": {...},
}
```

多数据集合并：

```python
CONFIG = {
    "name": ["CRH450", "Maglev"],
    "common_kwargs": {...},
    "kwargs": {
        "CRH450": {"route": 1},
        "Maglev": {"route": 2},
    },
}
```

常用字段：

- `name`：必须和 `dataset/<Name>.py` 同名。
- `use_surface` / `use_volume`：是否启用表面/体场分支。
- `num_surface_points`：输入 encoder 的表面点数量。
- `num_surface_query_points`：计算表面 target loss 的查询点数量。
- `num_query_points`：体场查询点数量。
- `surface_input_list`：encoder 输入通道，默认常用 `["xyz", "normal", "area"]`。
- `surface_target_list`：表面监督字段，例如 `["p"]`。
- `volume_target_list`：体场监督字段，例如 `["U"]`。
- `normalization`：`"standard"`、`"physical"` 或 `"none"`。
- `ids_file` / `split`：数据划分；DrivAerNet 常用 `ids_file`，高铁数据常用 `split`。
- `route`：多 route 模型的分支标识。
- `repeat`：重复采样次数。

合并数据集时，`surface_input_list`、target list 和采样点数必须一致，否则会在构建阶段报错。

## Model Config

```python
CONFIG = {
    "encoder_name": "UniFieldEncoder",
    "encoder_kwargs": {...},
    "decoder_name": "CrossAttentionDecoder",
    "decoder_kwargs": {...},
    "default_query_chunk_size": 50000,
}
```

常用字段：

- `encoder_name`：必须由 `model/Encoder/__init__.py` 导出，例如 `"UniFieldEncoder"`。
- `encoder_kwargs`：encoder 构造参数，例如 `embed_dims`、`depths`、`num_heads`、`window_sizes`、`down_ratios`、`cond_dim`、`num_routes`、`use_flow_attention`、`use_flow_ffn`。
- `decoder_name`：必须由 `model/Decoder/__init__.py` 导出，例如 `"CrossAttentionDecoder"`、`"AnchorDecoder"`、`"MLPQueryDecoder"`、`"SelfAttentionDecoder"`。
- `decoder_kwargs`：decoder 构造参数，例如 `out_channels`、`query_in_dim`、`query_dim`、`depth` 或 `query_depth`、`num_heads` 或 `query_heads`、`num_routes`、`use_flow_attention`、`use_flow_ffn`。
- `decoder_kwargs.memory_dim`：通常不用写，`model.build_model` 会从 `encoder_kwargs["embed_dims"][-1]` 自动推断；只有 decoder 使用非 encoder 最后一层 memory 时才需要手动指定。
- `default_query_chunk_size`：测试/可视化时 chunk 推理默认点数。

模型预设目录名，例如 `config/model/UniField_cross_attention/Medium.py`，只是实验预设名称。它不再要求 `model/UniField_cross_attention.py` 存在；实际完整模型统一由 `QueryUniField(encoder, decoder)` 拼接。

AdaField / UniFieldV1 旧模型也使用同一格式：

```python
CONFIG = {
    "encoder_name": "AdaFieldEncoder",
    "encoder_kwargs": {
        "depth": (4, 4, 6, 12, 8),
        "channels": (64, 128, 256, 512, 1024),
        "num_points": (1024, 256, 64, 16),
        "cond_dim": 2,
    },
    "decoder_name": "SurfacePressureDecoder",
    "decoder_kwargs": {"out_channels": 1},
}
```

这类 legacy surface 模型的 `memory_dim` 会从 `encoder_kwargs["channels"][0]` 自动推断。

模型规模文件名固定为 `Small.py`、`Medium.py`、`Large.py`。

## Experiment Config

Experiment config 负责组合 dataset、model、task、runtime、train、test、visualization。

### 顶层选择

```python
"dataset": "DrivAerNet",
"dataset_option": "surface_volume",
"model": "UniField_cross_attention",
"model_size": "Medium",
```

### task

```python
"task": {
    "surface_input_list": ["xyz", "normal", "area"],
    "surface_target_list": ["p"],
    "volume_target_list": ["U"],
    "query_type_channel": True,
    "losses": [
        {"domain": "surface", "field": "p", "pred_slice": (3, 4), "weight": 1.0},
        {"domain": "volume", "field": "U", "pred_slice": (0, 3), "weight": 1.0},
    ],
    "eval_fields": [...],
    "denormalize": {...},
}
```

- `surface_input_list`：必须和 dataset config 一致。
- `surface_target_list` / `volume_target_list`：用于从 target tensor 中按字段取真值。
- `query_type_channel=True`：表面 query 拼 `[x,y,z,0]`，体场 query 拼 `[x,y,z,1]`。
- `losses`：
  - `domain`：`"surface"` 或 `"volume"`。
  - `field`：target 字段名，例如 `"p"`、`"U"`。
  - `pred_slice`：模型输出中的通道切片，左闭右开。例如 `(3, 4)` 表示第 4 个通道。
  - `weight`：该 loss 权重。
- `eval_fields`：测试和可视化要输出指标/图片的字段；格式和 `losses` 相同，可不写 `weight`。

### denormalize

`denormalize` 只用于 `test.py` 和 `visualization.py`，把模型空间的预测/target 转回 raw-space 做指标和画图。

如果 dataset 使用 `normalization="standard"`，写 `type="affine"`：

```python
"denormalize": {
    "p": {"type": "affine", "mean": [-94.5], "std": [117.25]},
    "U": {
        "type": "affine",
        "mean": [21.725932072168813, -0.23861807530826643, 0.6522818340397506],
        "std": [12.373392759706936, 3.8865940380244175, 3.9738297581062536],
    },
}
```

含义是：

```python
raw = normalized * std + mean
```

如果 DrivAerNet 使用 `normalization="physical"`，写：

```python
"denormalize": {
    "p": {"type": "physical", "dataset": "DrivAerNet", "u_ref": 30.0},
    "U": {"type": "physical", "dataset": "DrivAerNet", "u_ref": 30.0},
}
```

对应 dataset 中的反变换：

```python
p_raw = p_norm * (0.5 * u_ref ** 2)
U_raw = U_norm * abs(u_ref) + [u_ref, 0, 0]
```

如果 CRH450/Maglev 使用 `normalization="physical"`，写：

```python
"denormalize": {
    "p": {"type": "physical", "dataset": "HighSpeedTrain"},
    "U": {"type": "physical", "dataset": "HighSpeedTrain"},
    "k": {"type": "physical", "dataset": "HighSpeedTrain"},
}
```

它会从 batch 的 `cond=[Ux/100, Uy/20]` 反推出：

```python
free_stream = [cond0 * 100, cond1 * 20, 0]
speed = norm(free_stream)
p_raw = p_norm * (0.5 * speed ** 2)
U_raw = U_norm * speed + free_stream
k_raw = k_norm * speed ** 2
```

`omega` 和 `nut` 在当前 high-speed physical 归一化中不做缩放，因此反变换也是原值。

如果 dataset 使用 `normalization="none"`，可以省略该字段。

### runtime

- `seed`：随机种子。
- `matmul_precision`：通常为 `"high"`。
- `amp`：是否启用 fp16 autocast。
- `compile`：是否启用 `torch.compile`。
- `distributed.gpu_num`：GPU 数。
- `distributed.start_gpu`：起始 GPU id。
- `distributed.port`：DDP 端口。

### train

- `batch_size` / `workers`：DataLoader 配置。
- `lr` / `weight_decay`：优化器配置。
- `total_epoch` / `resume_epoch`：训练轮数与恢复位置。
- `save_step`：checkpoint 保存间隔。
- `clip_grad`：梯度裁剪阈值。
- `final_lr_ratio` / `warmup_iters`：cosine scheduler 配置。
- `saveroot`：checkpoint 输出目录。
- `checkpoint`：可选预训练权重。
- `route_map`：route 兼容加载时使用。
- `dry_run_iters`：大于 0 时只跑指定 iteration，用于 smoke test。

### test

- `checkpoint`：测试权重。
- `batch_size` / `workers`：DataLoader 配置。
- `query_chunk_size`：chunk 推理点数。
- `max_batches`：大于 0 时只测试前几个 batch。
- `log_interval`：日志间隔。

### visualization

- `checkpoint`：可视化权重。
- `query_chunk_size`：chunk 推理点数。
- `max_points`：每张图最多绘制点数。
- `output_dir`：为空时默认写到 `temp/<timestamp>_main_visualization/visualization/`。
