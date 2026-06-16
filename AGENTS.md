# Codex 项目约定

这个仓库当前主线研究 high_speed_train / motorBike 上的 OpenFOAM-style finite-volume physics loss。请优先保持仓库结构整洁。

## 目录组织

- 根目录尽量只放主线训练脚本、对训练模型的可视化脚本、README、requirements、git 配置等项目级文件。
- `loss/` 放 OpenFOAM FV operator、物理损失、边界条件 loss 等共用 loss 代码。
- `dataset/` 放 high_speed_train / motorBike 数据读取、VTK/cache 读取和 OpenFOAM 导出数组读取逻辑。
- `model/` 放模型定义。
- `utils/` 放训练、测试、可视化共用的非模型/非数据集工具，例如 config 解析与合并、checkpoint、distributed、metric、optimizer、scheduler、task tensor 处理等。
- `log/` 尽量只放主线训练结果、checkpoint 相关日志和正式训练可视化。
- 小型分析、临时诊断、一次性可视化实验放入 `temp/`。

## 主线入口职责

- `train.py` 只保留训练主干：加载主 config、构建训练 dataset/model/optimizer/scheduler、训练循环、checkpoint 保存和分布式启动。
- `test.py` 只保留评估主干；`visualization.py` 只保留正式可视化主干。
- 多入口共享的逻辑不要放在根目录脚本中。比如 experiment config 组合、阶段级 dataset kwargs 覆盖、路径解析等应放在 `utils/config.py` 或更合适的 `utils/` 模块。
- dataset 划分、cache/VTK 读取、采样和归一化逻辑优先放在 `dataset/`；模型结构和 attention/encoder/decoder 逻辑优先放在 `model/`。
- 根目录脚本不承担临时实验和一次性诊断职责；这类内容放入 `temp/YYYYMMDD_HHMMSS_function_name/`。

## 临时分析目录规范

新建临时分析时，使用：

```text
temp/YYYYMMDD_HHMMSS_function_name/
```

每个分析目录尽量包含：

- 运行用的 `.py` 脚本。
- `report.md`，记录目的、命令、输入数据、关键指标、结论。
- `log/`，记录运行日志、stdout/stderr、配置等。
- `visualization/`，记录图片、CSV、JSON summary 等可视化和分析输出。

临时分析脚本如果需要 import 仓库模块，应显式把仓库根目录加入 `sys.path`，避免从 `temp/` 下直接运行时 import 失败。

## 修改习惯

- 进行任何调试时，进入可用的计算节点进行调试，一般我都已经通过sleep.sh申请好计算节点，在这里我会标记当前申请的节点名称：**n30152**。 
- 在计算节点上，如果发现没有环境的话，请通过. /data/home/zdhs0017/OpenFOAM/of13-compute-env.sh加载OpenFOAM环境，以及通过conda activate zoujunhong加载python环境
- 不要把一次性分析脚本继续放在根目录。
- 不要把临时分析输出写入主线 `log/`；默认写入该分析目录自己的 `log/` 或 `visualization/`。
- 如果临时分析后来成为主线能力，再把它整理回根目录或合适的模块，并同步更新 README。
- 小实验默认继续保留在 `temp/`。当用户明确指定某个实验为里程碑时，才把该实验整理进主目录/模块/`log/`，并在整理完成后清空 `temp/`。
- 每次里程碑整理后，同步更新 `PhisicsBase.md`，说明该次实验的物理基础、OpenFOAM 对齐程度、误差、缺失条件和当前 loss 的自由度问题。

## 可视化习惯

- 对物理 loss、残差分布、优化场或手工构造场做可视化时，默认把 OpenFOAM 真值场、预测/构造场、二者误差和 loss/residual 分布放在同一组图中；不要只单独画 loss 分布，避免低残差但明显错场的情况被隐藏。
- 对流场切面做可视化时，默认同时给出近物体局部切面和全计算域切面。局部切面用于看车体/壁面附近细节，全场切面用于确认远场、入口/出口和整体误差分布。
