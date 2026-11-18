# FullAircraft cohort

2026-11-18 枚举完整 raw 文件集合共 3461 例。冻结的默认 cohort 包含 2661 例有有效压力行的数据；800 例原始 CFD 无有效数据，逐例原因、源文件和有效点数见 `invalid_source.json`。没有按模型误差或压力极值剔除整例。

`complete_3461.txt` 保存完整 raw 名单；`complete_valid_2661.txt` 是当前完整可用名单；`historical_cached_2661.txt` 保存清理前已缓存的 2661 例，历史结果使用该清单以固定原划分含义。训练/验证从缓存根 `case_manifest.json` 读取冻结 cohort，缺少任何所需缓存都会报错。

Tecplot 的 `-nan(ind)` 被识别为 NaN，仅过滤非有限行，避免因一个非有限值拒绝整例。
