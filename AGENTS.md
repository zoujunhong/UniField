# whole_field 仓库管理规定

本仓库职责：数据驱动流场模型、监督学习与正式训练框架。

## 实验与结果管理（2026-11-18）

本节为当前规范，优先于仓库内历史说明和实验脚本中的旧约定。

- `~/zoujunhong/code/whole_field` 负责数据驱动流场任务；`../PINN` 负责物理信息任务、OpenFOAM FV operators、方程/边界条件 loss 与基础 PINN。不得在 PINN 新增纯监督模型实验。
- 根入口只负责运行主干。共用训练、checkpoint、config、metric 等逻辑放 `utils/`；数据读取、划分、采样、归一化和缓存放 `dataset/`；模型放 `model/`。临时诊断不放根目录。
- 所有模型训练都从正式入口、可执行 config 启动：`python train.py --config config/experiment/<主名>.py` 或 `python train.py --config log/<主名>/<副名>/config.py`。禁止在 temp 中放置新的模型训练入口或用隐藏 CLI 参数改变训练。测试、可视化和一次性分析可放 temp。
- `config/dataset/<数据集>/` 汇总该数据集的 `surface.py`、`volume.py`；必要的联合任务选项同目录存放。不同数据形态不能另起数据集目录。数据划分和缓存 manifest 不以首次使用的模型命名。
- whole_field 模型族仅保留 ABUPT、SMART、TripNet、Transolver3、UniField、VolumeUniField、DSAPT。UniField 指原 UniFieldV2；原 V1 不再使用；VolumeUniField 指原 cross-attention 系列。DSAPT 全称 Dual-Scale Aerodynamic Point Transformer。不得用同一名字混淆官方实现与不同的自制实现。
- 模型规模由结构定义，在 model README 中登记。Transolver3 Medium=12层、Large=24层；DSAPT TinyG2222L2111 指 Tiny 主干、global/coarse 分支步幅2222与local/fine分支步幅2111，这些数字不是层数。结构细节及原名写入配置和迁移记录。
- 主配置名必须为 `数据集_模型_规模_Surface预测量_Volume预测量.py`；无该域监督填 X。量采用 P、U、Normal、K、Omega、Nut、Distance 等已登记符号。名称不写 loss、优化器、GPU数、学习率、数据比例或跨类划分。
- 结果目录为 `log/<主名>/<副名>/`。默认副名 `default`；变体按 `调整项_目标` 串联，如 `Loss_RelL2_GPU_3`。目录必须有 `config.py`（import主配置后显式覆盖）、简短 `README.md`（仅写改动）和 `train.log`。实际解析配置、来源哈希与原始机器指标可作为附加记录。训练日志采用 `[epoch][当前进度/总进度], lr: ..., time: ..., Loss: ..., surface_p: ...`，实际 global_step 同时记录在机器指标与 checkpoint；epoch或批次游标仅作诊断，不控制预算。
- 新训练使用 `total_steps` 和按优化器更新计数的验证/保存间隔，默认200000步、每20000步验证。精确接续保持原预算、学习率位置、优化器、AMP scaler、EMA、各rank RNG与数据游标；仅有模型权重只能声明初始化，不能宣称精确接续。
- 每次验证使用完整有效验证划分，不得限制32车、max_batches或静默跳过缺缓存样例。默认各受监督域单组10000随机query点，使用固定seed便于比较；在物理Cp空间、归一化前排除Cp < -10的点。整行排除使同点速度和其他监督量同步排除。其他物理量暂不设极值阈值。几何输入不依赖标签筛选；训练与测试不自动继承该验证阈值。
- 最优checkpoint只能由验证集选择，默认逐样例相对L2均值、越小越好；记录指标空间、点集规则、有效样例数及步数。验证协议改变后重建best比较记录，历史32车成绩不可与新完整验证成绩混用。
- checkpoint统一存 `~/zoujunhong/model/savemodel/field/<主名>/<副名>/`：`latest.pth` 为完整续训状态、`best.pth` 为验证最优；使用EMA时另存 `latest_ema.pth`、`best_ema.pth`。原子保存，保留独立可推理的完整参数与buffer。配置和改动说明同时备份。缺失的历史best/optimizer/RNG不得补造或把latest改名冒充best。
- 下载的预训练权重只放 `~/zoujunhong/model/pretrain/`，记录来源与哈希。checkpoint不得混在源码仓库、原始数据或temp中。
- 原始CFD数据只放 `/lustre/home/zdhs0017/data/`；DrivAerNet++公共原始数据保留 `/lustre/group/project1/CFD/DrivAerNet++/`。所有数据缓存放 `~/zoujunhong/data/`，DrivAerNet++正式缓存为 `~/zoujunhong/data/DrivAerNet/`，仅原始数据保留公共目录（2026-11-18用户最终更正）。每例一个目录，含 `surface.npy`、可用时的 `volume.npy` 与 `metadata.json`。元数据记录字段顺序/精度/单位、工况、可得的力系数、split、来源及校验值；输入几何/必要索引等辅助文件须说明语义。不能将同名p字段的Cp、Pa、运动学压力混用。
- 常规测试/评估读取缓存；显式要求全点测试时直接读取原始VTK，不为全点测试长期保留额外大点池。常规缓存模式缺文件明确报错，补建为独立任务。DrivAerNet++保留完整1154例测试划分与既有十套固定10000点，历史复查按原点集并保留set00对照。新增评估必须明示采样协议；通用入口当前十组可复现随机query不能冒充历史固定点。跨模型正式对比须显式选择并复用同点集。只过滤有拓扑证据的孤立顶点；不删除正常独立部件，不因压力大小筛测试点。
- 六项表面压力测试指标：MSE/MAE按所有评分点汇总，RMSE=sqrt(pooled MSE)，R²使用全部点的统一真值均值；relative L1/L2逐车等权平均。每套独立计算，再汇总十套均值、最小、最大、样本标准差(ddof=1)。同时记录原始与标准化空间；十套标准差只是点采样波动。
- 正式训练达到预算、最终checkpoint完整保存且训练进程退出后，自动在释放的训练GPU接续测试，默认最终EMA。持久记录等待/测试中/完成/失败并加锁；异常退出或主动停止不触发完成测试，测试失败可恢复。
- temp仅用于非训练临时任务，命名 `temp/YYYYMMDD_HHMMSS_function_name/`，包含脚本、`report.md`、`log/`、`visualization/`。脚本显式将仓库根加入sys.path。未经新的清理授权，不删除后续实验；里程碑整理要移出temp、验证源码与数据依赖，再删除旧目录。迁移清单及已无法恢复的历史信息写入 `docs/maintenance/`。
- PINN物理里程碑每次整理后更新 `PhisicsBase.md`，记录方程、OpenFOAM对齐、误差、缺失闭合条件与自由度；低残差不能代替对真值流场的验证。
- whole_field 根目录的 `docs/`、`paper/`、`tests/`、`visualization/` 仅在本地保存，不纳入 Git 跟踪；取消跟踪不得删除本地内容。根入口 `visualization.py` 仍属于正式源码。代码、配置与仓库级管理文档继续同步 GitHub。

## 计算资源与检查

- 调试、训练、缓存构建、测试均在已申请计算节点运行。当前可用节点以`squeue`和节点进程为准，不能仅相信历史节点名；历史常用n30152。必要时source `/data/home/zdhs0017/OpenFOAM/of13-compute-env.sh`，Python使用conda `zoujunhong`。
- 临时非训练任务启动前检查所有已申请节点是否正在训练，优先空闲节点；都在训练则另申请。GPU测试通常1GPU，CPU任务4–8核，按余量调整并保留余量。训练结束后原GPU自动接续测试允许同节点其他GPU仍训练。
- 默认直接正式启动并短暂观察进展；只为实际故障或实质改动做必要检查，不强加无关预检。目录/代码重构涉及恢复状态时须做针对性的恢复一致性与依赖检查。
- 路径迁移后同步修改正式源码、配置与必要manifest；旧位置不保留兼容软链接。历史档案中的原路径仅作溯源，不能作为运行依赖。
- 大型跨盘迁移逐文件校验后再移除源，保存源→目标清单；清理前确认所有要保留权重、指标、配置、源码和固定测试点集均可访问。正在运行的任务先安全保存退出，按原步数接续。

## 可视化与论文

- 物理loss、残差或构造场图默认同组展示OpenFOAM真值、预测/构造场、误差和loss/residual。切面同时给近物体局部与全域。
- 压力用coolwarm，速度及非负量用viridis，参照 `/data/home/zdhs0017/OpenFOAM/Cases/high_speed_train_debug/visualization_CRH450_geo_19_000002_Ux_98.79_Uy_16.29`。全域线性色标P1–P99并标注截断；局部覆盖窗口全部极值，必要时正值log、带符号symlog。不标绿色十字、极值圈或箭头。
- 论文LaTeX在登录节点限制线程编译。每次修改正文、补充、图表、引用或排版源码后立即重编相关PDF并检查错误、未定义引用及分页。默认未署名main.pdf和补充；仅明确要求时编main_signed.pdf（build.sh --signed）。
