# 环肽主链 N-甲基化的位点选择

在 PyCharm 或 Anaconda Prompt 运行本工程，见 [本地启动说明](LOCAL_RUN.md)。

把一个主链酰胺氮甲基化后，环肽的被动通透性会改变多少？同一母体若有多个可选位点，优先测哪个？本仓库从 CycPeptMPDB 的 PAMPA 数据重建这两个决策。原 v0.1.0 绝对值基线在 `results/legacy/` 中提供逐行复核。

## 先看三份结果

- [真实双位点结构](results/pairs/examples/kelly_site_choice.png)给出同一实验中的母体 #6618、D-Leu4/NMe 的 #6645 和 L-Leu6/NMe 的 #6622。后两者实测 log₁₀P 为 −6.85 和 −5.48。[模型解释](results/model/example/score_trace.json)记录了显式模型偏向前者的实际预测。
- [编辑清单](results/pairs/edits.csv)含母体、变体、化学系列、实验协议、立体化学 SMILES、编辑氮、原子对应和实验值。完整筛选由 6,941 条 PAMPA 记录得到 3,410 个有已测母体的增量对和 1,112 个可排序母体。后者中有 437 个缺同条件可定量母体。8 个保守系列跨 68 个实验子库/条件，917 个完全去目标 N-甲基后的骨架。[数据复核说明](results/pairs/README.md)
- [逐母体指标与总表](results/references/comparison.csv)及[按母体明细](results/references/comparison_by_parent.csv)含所有正式比较。固定测试包括 460 个可排序母体、1,028 个实测增量、1,545 个候选选项；其中 Kelly 贡献 458 个排序母体、Le Roux 贡献 2 个。共同表中 50 个无法对齐残基链的训练编辑已移出。所有方法均保留完整测试选项。

## 固定测试

首选损失 = 该母体实测最佳位点的 log₁₀P 减去模型首选位点的实测 log₁₀P。每个母体等权，预测并列时取并列选项的期望值。同组中未实测的母体只能比较位点，无法计算 Δ。下表列出固定测试的 11 种正式比较。第 12 项为 CycDelta 作者发布权重，作为训练数据有重叠的文献参照单独计入快速复评分。

| 方法 | 首选损失 ↓ | 首选命中率 ↑ | Δ MAE ↓ |
|---|---:|---:|---:|
| 随机选位，Δ 恒报 0 | 0.2393 | 47.97% | 0.4436 |
| 随机选位，Δ 恒报训练中位数 | 0.2393 | 47.97% | 0.3746 |
| v0.1.0 predict_pair 绝对值作差 | 0.2362 | 47.83% | 0.4161 |
| ExtraTrees 绝对值作差 | 0.2071 | 50.76% | 0.3993 |
| Chemprop D-MPNN 绝对值作差 | 0.2757 | 43.04% | 0.4320 |
| 化学分支 | 0.2114 | 52.83% | 0.3836 |
| 普通融合 | 0.2247 | 50.65% | 0.8239 |
| 化学分支＋编辑位置 | 0.2428 | 48.04% | 0.4287 |
| 显式编辑融合 | 0.2511 | 46.09% | 0.3715 |
| CycDelta 同折重训 | 0.2660 | 41.30% | 0.3861 |
| ncAA-RepDistill 冻结编码器＋同折读出 | 0.1527 | 63.04% | 0.4114 |

显式模型在固定测试中约 99% 的编辑预测为正。把 8 个系列逐一留出，其首选损失为 0.2145，随机选位为 0.2177，差值 −0.0032 的按系列重采样 95% 区间为 [−0.0124, +0.0928]。显式模型的 Δ MAE 为 0.4654，训练中位数参照为 0.3692。[逐系列预测与不确定性](results/model/series_sensitivity/)

CycDelta 发布权重的训练集中出现过 1,545 个测试选项中 1,467 个选项的一端或两端，因此该权重的结果在比较表中标为 `reference_only`。ncAA-RepDistill 的原监督读出与测试结构也有重叠，正式比较重新训练读出。其无标签适配阶段接触过部分测试结构，完整剔除这些母体组后，只余 29 个可排序母体，首选损失 0.1945、命中率 51.72%。两套上游数据池均未接触的完整组仅 3 个，有定量母体的编辑仅 1 对。另行逐系列留出的 ncAA-RepDistill 在 1,103 个母体上首选损失为 0.2225、命中率为 47.96%，Δ MAE 为 0.4464。同一八折 CycDelta 的首选损失为 0.1931、命中率为 49.68%，Δ MAE 为 0.3805。它与显式模型的首选损失差为 −0.0214，按 6 个可排序系列成对重采样的 95% 区间为 [−0.0288, +0.0693]。[全部方法的逐系列复核与同折比较](results/references_series/README.md)。两项方法的参照母体、原基准的交集、原训练与适配重叠、跨运行变化均见[文献方法复跑](results/references/README.md)。

## 在相同分子上比较构象

5,160 条 4D 记录与原有分子和标签对应。3,196 个定量母子对及 652 个完整排序组有母体和全部相关变体的双溶剂描述符。严格覆盖的固定测试含 216 个排序母体，均属 Kelly。2D、加水相、再加正己烷的首选损失为 0.23285、0.22970、0.23403。逐系列结果和恢复 Kelly 低传输记录后的敏感性见[构象消融](results/conformation/README.md)。

## 复跑

使用 Python 3.12。以下命令从 `B_PAMPA` 目录运行。CPU 版 PyTorch 独立安装，随后安装固定版本的数值计算、RDKit 和编码器依赖。首跑所需的配对、样例和预测已随仓库提供。需要原始数据库 CSV 时，`fetch_data.py` 从作者固定版本下载并核对校验和。首次提取特征还需联网下载两个固定版本的预训练编码器。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/requirements_edit_cpu.txt
python scripts/fetch_data.py
python scripts/prepare_pampa.py
python scripts/pair_data.py
python scripts/check_round.py --check-atoms
```

快速查看一个真实决策，运行 `python scripts/replay_site_choice.py`。它从随包的两个未标准化的融合向量和拟合权重现场推理，可离线得到 D-Leu4 / L-Leu6 的 +0.344 / +0.332 与首选损失 1.37。[两行实测和预测](results/model/example/kelly_two_site_replay.csv)。安装依赖后，检查全部随包预测可执行上面的最后一行。它验证 4,322 个编辑的原子映射、去甲基回构、分折和全部正式对照的测试分母，并从逐编辑预测重新计算三个主指标。重建历史与新模型：

```bash
python scripts/run_baseline.py --data data/processed/pampa.csv --output .cache/legacy --seed 2026 --trees 200 --jobs 2
python scripts/audit_legacy.py --refit .cache/legacy
python scripts/encoder_cache.py --weights .cache/weights --fetch --cache .cache/encoders --demo-data data/processed/pampa.csv
python scripts/pair_chemistry.py data/raw/cycpeptmpdb_all.csv .cache/graph.json
python scripts/prime_edit_cache.py --graph .cache/graph.json --records data/processed/pampa.csv --weights .cache/weights --cache .cache/encoders
python scripts/fit_edits.py --weights .cache/weights --cache .cache/encoders --threads 2
python scripts/conform_ablation.py
```

- [预训练编码器、改造位点对齐和八系列留一](docs/edit_model.md)给出批量缓存、冷启动、拟合头重载及实际示例命令。主线使用冻结的 ESM-2 8M 和 ChemBERTa-77M-MLM 编码器。原子图保留手性、闭环和局部氮环境。
- [文献实现](results/references/README.md)给出 ncAA-RepDistill、CycDelta 和 D-MPNN 所需的额外依赖、上游提交和权重。下载的大型第三方模型按固定来源加载。
- [原始数据来源与许可](data/sources.json)、[历史模型审计](results/legacy/audit.json)和[本次独立重计算](results/round1/rescore.json)。历史来源留出 ExtraTrees 的绝对值 MAE 为 0.881469，训练均值为 0.851938。固定测试表的 Δ MAE 评价一次改造的增量。
