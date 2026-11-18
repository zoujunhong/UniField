# 配置规范

所有训练通过 `python train.py --config <文件.py>` 启动；文件导出 `CONFIG` 字典。主配置使用 `config/defaults.py` 的工厂组合已登记的数据集、模型和统一训练协议。

```text
config/dataset/<数据集>/surface.py 或 volume.py（联合任务可有 surface_volume.py）
config/model/<模型族>/<规模>.py
config/experiment/<数据集>_<模型>_<规模>_Surface<量或X>_Volume<量或X>.py
```

物理量代码为 P、U、Normal、K、Omega、Nut、Distance；顺序与 task 的字段顺序一致。规模由模型结构决定，学习率、GPU数量、训练比例和损失类型不写入主名。

## 派生实验

每个变体放 `log/<主名>/<副名>/config.py`。默认副名 `default`，仅导入主配置；其他副名由 `调整项_目标` 组成。必须先 deepcopy，避免修改导入对象。

```python
from copy import deepcopy
from config.experiment.DrivAerNet_Transolver3_Medium_SurfaceX_VolumePU import CONFIG as BASE
CONFIG = deepcopy(BASE)
CONFIG['experiment'].update(variant='Loss_RelL2_GPU_3', description='损失改为相对L2，使用3张GPU。')
CONFIG['runtime']['distributed']['gpu_num'] = 3
for loss in CONFIG['task']['losses']:
    loss['loss_type'] = 'relative_l2'
```

同目录 README.md 只写相对默认的改动。启动器会保存解析配置、来源、train.log，并将 config.py 和 README.md 备份至对应 checkpoint 目录。历史迁移配置另保留 legacy_config、来源路径及缺失状态说明。

## 关键字段

- `experiment.name/variant/description`：主名、副名和改动说明。
- `dataset/dataset_option/dataset_kwargs`：数据源及公共参数；`train/validation/test.dataset_kwargs` 为阶段覆盖。
- `task.surface_input_list/surface_target_list/volume_target_list`：输入和监督量；`task.losses` 中设置 domain、field、pred_slice、weight、loss_type。
- `runtime.distributed.gpu_num`：GPU数；`runtime.seed/amp/amp_dtype/compile`：运行设置。
- `train.total_steps`：优化器更新预算；`train.validation.interval` 和 `checkpoint_interval_steps` 均以步计数。
- `train.optimizer/lr/weight_decay/warmup_iters/final_lr_ratio`：优化配置；`train.ema={'enabled': True, 'decay': 0.9999}`。
- `train.resume_checkpoint`：完整状态接续；`train.checkpoint`：权重初始化。二者不能混淆。
- `train.auto_test`：正常完成后运行测试；`test.checkpoint` 可显式选择权重。
- `test.query_sets`：通用测试的随机采样组数，默认10；DSAPT 使用原始十套固定索引协议。
- `task.denormalize.<domain>.<field>`：按表面/体场分别指定反归一化；同名压力场可以有不同 mean/std。未提供变换时只报告模型空间指标。

验证始终完整划分、固定单组10000 query/监督域、物理Cp<-10整行过滤。历史特殊训练设置写在副配置中；训练阶段的原过滤规则不应冒充验证或测试规则。若数据集历史上只有 train/val，不能把 val 宣称为独立 test，应关闭自动测试并在配置中记录。

常规测试使用缓存；显式全点测试直接读取原始 VTK，由相应评估程序分块预测并记录全点协议。全点结果、历史固定点结果和默认随机 query 结果分别标记，不能互换。当前通用 `test.py` 是缓存采样入口，不应仅把 query 数量调大就称为原始全点测试。

DSAPT 使用 `train.engine='dsapt'`；公共 train/runtime/loss 字段同步进入专用计算循环，稀疏结构与数据细节在 `CONFIG['dsapt']`。TripNet 依赖 triplane 输入，使用 HighSpeedTrain 的专用 dataset option。不同原生 baseline 的几何标准化由其适配器处理，不额外叠加其他模型的坐标变换。
