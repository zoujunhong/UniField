# whole_field

数据驱动流场预测仓库。物理信息任务、OpenFOAM finite-volume 算子与 PINN 在相邻的 `../PINN` 仓库维护。物理里程碑及loss闭合限制记录在 [PINN/PhisicsBase.md](../PINN/PhisicsBase.md)。项目管理规则以 [AGENTS.md](AGENTS.md) 为准；历史论文信息见 [PaperInformation.md](PaperInformation.md)。

## 正式训练

```bash
conda activate zoujunhong
python train.py --config config/experiment/DrivAerNet_Transolver3_Medium_SurfaceX_VolumePU.py
# 修改实验：配置保存在结果目录，直接使用该文件重现
python train.py --config log/DrivAerNet_DSAPT_TinyG2222L2111_SurfaceP_VolumePU/Loss_RelL2_Optimizer_Ranger_GPU_3/config.py
python test.py --config log/<主配置名>/<副配置名>/config.py
```

训练只接受 `--config`，不在 `temp/` 启动模型训练。普通入口和 DSAPT 使用同一配置、目录、步数、验证及 checkpoint 约定；DSAPT 的稀疏点模型专用计算循环位于 `utils/training_dsapt.py`。运行前按 AGENTS 的资源约定选择计算节点。

保留模型：ABUPT、SMART、TripNet、Transolver3、UniField（原 V2）、VolumeUniField（原 cross-attention）、DSAPT（Dual-Scale Aerodynamic Point Transformer）。具体实现与规模见 [model/README.md](model/README.md)。Transolver3 Medium/Large 分别为12/24层；DSAPT TinyG2222L2111 的数字表示两路步幅，不能解释为网络层数。

## 配置、记录与存储

主配置名为 `数据集_模型_规模_Surface预测量_Volume预测量.py`；无监督域用 X。损失、优化器、学习率、GPU数、数据比例等放副配置，详见 [config/README.md](config/README.md)。

```text
config/experiment/<主名>.py
log/<主名>/<副名>/config.py          # import 主配置后覆盖改动
log/<主名>/<副名>/README.md          # 简短改动说明
log/<主名>/<副名>/train.log
~/zoujunhong/model/savemodel/field/<主名>/<副名>/
    config.py, README.md, latest.pth, best.pth
    latest_ema.pth, best_ema.pth      # 开启 EMA 时
~/zoujunhong/model/pretrain/         # 下载的预训练权重
```

默认副名为 `default`；变体示例 `Loss_RelL2_GPU_3`。最优权重由验证集确定。不存在的历史 best 不补造；只有权重的旧模型用于初始化或推理，不能精确接续。无法重建的历史架构在其配置中明确禁止训练，并保留原始配置和来源记录。

新训练默认200000次优化器更新。未特别指定时，验证间隔为 `max(10000, ceil(total_steps / 10))`：200000步每20000步验证，50000步每10000步验证；显式间隔优先。每次验证使用完整有效验证集。每个受监督域采用单组10000随机 query；在归一化前排除 Cp < -10 的整行，速度等同点标签同步排除。验证不改变几何输入，不把该阈值用于测试。DrivAerNet++ 当前冻结划分为5817训练、1148验证、1154测试；旧32车监控结果不参与新 best 比较。DSAPT 测试保留原十套固定10000点。通用入口默认十组可复现随机 query，在 summary 中标记采样协议；这些新指标不能冒充历史固定点结果。评分区分模型空间与原始物理空间，后者仅对配置了反归一化的字段报告。

完整 checkpoint 保存模型、optimizer、AMP scaler、EMA、步数、数据游标与各 rank 随机状态。正式完成训练并释放进程后自动进行配置指定的测试；主动停止和异常退出不触发完成测试。历史中止模型的测试仍从 `test.py --config` 显式启动。

## 代码与数据

- `train.py`、`test.py`、`visualization.py`：入口主干。
- `utils/`：训练循环、配置合并、恢复、指标及测试流水线。
- `dataset/`：读取、缓存、采样、归一化、冻结划分；同数据集 surface/volume 配置放同目录。
- `model/`：模型结构及必要的官方依赖；`loss/`：监督学习损失。
- `temp/YYYYMMDD_HHMMSS_function_name/`：临时测试、分析和可视化，带脚本、report.md、log/ 与 visualization/。
- `docs/maintenance/20261118/`：本地保存的清理映射、验证证据及历史源码归档，入口见[本地整理报告](docs/maintenance/20261118/report.md)。

根目录的 `docs/`、`paper/`、`tests/`、`visualization/` 保留在本地，不通过 Git 跟踪或上传 GitHub；这些目录中的报告与材料不包含在远端源码仓库中。正式入口 `visualization.py` 继续跟踪。

原始 CFD 数据统一 `/lustre/home/zdhs0017/data/`，DrivAerNet++ 公共原始数据保持 `/lustre/group/project1/CFD/DrivAerNet++/`。所有缓存位于 `~/zoujunhong/data/`，DrivAerNet++ 缓存根为 `~/zoujunhong/data/DrivAerNet/`，每例目录包含 `surface.npy`、可用时的 `volume.npy` 和 `metadata.json`；几何输入与索引作为有说明的辅助文件。详见 [dataset/README.md](dataset/README.md) 和 [DrivAerNet++ 缓存规范](dataset/CACHE_FORMAT.md)。训练、测试不依赖 temp 中的缓存或源文件。

常规测试使用缓存；需要全点测试时，显式读取原始 VTK。DrivAerNet++ 当前仅保留 65536 表面点/200000 体场点主池，以及随机和空间均匀两套 100000 表面点池；主池体场包含 P、U、wall_distance。固定测试点与划分保留。
