# 外部模型在八个化学系列上的配对比较

所有模型共用已冻结的八组外部测试系列、内层训练和验证角色。每个分子的所有角色及一个母体的完整选项均随系列划分。每次从头拟合读出或预测网络，测试标签在报告实测 Δ、最优位点和损失时才读取。每个方法须预测相同的 4,272 个选项，才能进入逐母体或逐系列比较。随机选位给每个候选相同的权重，Δ 的常数参照用各内层训练折的中位数。

## ncAA-RepDistill

使用作者 Phase2 E0_B0 的 768 维图注意力 CLS 表征，读取完整立体化学 SMILES。每一外折仅用五个训练系列内按协议去重的实测分子拟合同一 256 单元 ReLU 读出。另两个系列的绝对测量 RMSE 决定早停轮次。作者原生训练函数独立核对所选择轮次的验证误差。评分对指定母体与每一变体的两个绝对预测相减；外折含未实测母体时，同样从结构计算共同参照。

完整配对中有 3,709 条至少一端出现在作者已知的 Phase2 训练或验证适配结构库。这里是无标签结构暴露。若某母体任一选项任一端曾出现，则移除整个母体组。余下 286 个选项、56 个可排名母体及 220 条实测 Δ。上游发布的监督读出未参与拟合。Phase1 已公开的 219,150 条训练记录及 24,350 条验证记录为多组分结构，与本任务的单连通整分子无完整身份匹配。教师模型全部原始预训练样本仍不完全可核查。

## CycDelta

复用已核对的残基层级 Uni-Mol／描述符缓存。用每一对固定的化学母体作参照。每一外折仅从训练折的定量 Δ 对取得描述符标准化量，再以随机种子 0 训练原生 CycGNN，以两个验证系列中的可定量 Δ-MAE 选轮次。测试时采用作者的零点校正，用于母体未测的位点组时仍可生成预测。原发布监督网络的权重未用于此处的八折拟合。

## 复跑

从 B 方法项目根目录运行。在 results/references/README.md 所述环境中使用对应的 RDKit 版本，EXTERNAL 指向已经核对提交与权重的公开工程。ncAA 可直接使用随包保存的完整 Phase2 CLS 特征矩阵。CycDelta 的固定单体缓存已随此目录保存，也可由主参考实验 refs_cycdelta.py 重新生成。

~~~bash
python scripts/series_ncaa.py --external "$EXTERNAL" --pack results/references/ncaa_native_features.npz --out results/references_series/ncaa
PYTHONPATH="$EXTERNAL/rdkit-cyc${PYTHONPATH:+:$PYTHONPATH}" python scripts/series_cycdelta.py --external "$EXTERNAL" --cache results/references_series --out results/references_series/cycdelta
python scripts/series_reference_compare.py
~~~

`comparison_metrics.csv`、`comparison_per_parent.csv`、`comparison_per_series.csv` 和 `comparison_uncertainty.csv` 汇集了两种文献方法和八种本地方法／基础参照。`reference_differences.csv` 收录与每种文献方法成对的差异及区间。用归档参数逐一复算 4,272 个预测的命令如下，ncAA 使用本项目的 RDKit 环境，CycDelta 使用其作者指定的 2026.3.4 环境。两者记录到 `verify_saved_weights.csv` 的最大绝对差均为 2.22×10⁻¹⁶。

~~~bash
python scripts/verify_series_weights.py --model ncaa --external "$EXTERNAL"
PYTHONPATH="$EXTERNAL/rdkit-cyc${PYTHONPATH:+:$PYTHONPATH}" python scripts/verify_series_weights.py --model cycdelta --external "$EXTERNAL"
~~~

每一方法的 all/ 提供逐母体和逐系列指标，以及按同一组化学系列成对重采样 2,000 次得出的区间。reference_minus_baselines.csv 的方向为该方法减列名指定的参照。损失越小越好，命中率越大越好。若指标由少于三个系列提供，成组区间留空，逐系列结果另列。ncAA 的未暴露子集在 phase2_unseen_complete_groups/。权重、轮次和内层标签计数存于各方法 folds/、fitted_folds.csv 和 run.json。

## 八折实测比较

| 完全相同选项上的模型 | 1,103 母体首选损失 | 1,103 母体命中率 | 3,360 Δ 的 MAE |
|---|---:|---:|---:|
| ncAA-RepDistill | 0.2225 | 47.96% | 0.4464 |
| CycDelta，按指定母体从头训练 | 0.1931 | 49.68% | 0.3805 |
| 显式编辑融合 | 0.2145 | 50.50% | 0.4654 |
| 随机选位、Δ 恒为零 | 0.2177 | 48.10% | 0.4408 |
| 所有位点并列、Δ 恒报内层训练中位数 | 0.2177 | 48.10% | 0.3692 |

ncAA 减显式的首选损失差为 +0.0080（95% 系列成对区间 −0.1692 至 +0.0489），ncAA 减随机为 +0.0048（−0.0855 至 +0.0364）。6 个可排序系列的母体数依次为 1、5、1、636、458、2；按系列等权时，ncAA 和显式的首选损失分别为 0.1371 和 0.3251。主测试的结果与八系列的结果都应保留，并说明每种加权把哪个问题作为评价对象。

在 56 个完全去除 Phase2 适配结构暴露的可排序母体中，ncAA、显式和随机的首选损失为 0.3014、0.1804、0.2080，命中率为 32.14%、53.57%、49.70%。此子集只有两个系列可排序，Townsend 27 个母体，Kelly 29 个母体。220 条可测 Δ 涉及 5 个系列，三种方法的 Δ-MAE 分别为 0.4429、0.4936、0.3840；训练中位数为 0.3485。八个外折的独立原生验证与导出读出之间，最佳验证 RMSE 最大偏差为 5.54×10⁻⁸。

CycDelta 减显式模型的首选损失为 −0.0214（95% 系列成对区间 −0.0288 至 +0.0693），减随机为 −0.0246（−0.0293 至 +0.1282），减 ncAA 为 −0.0294（−0.0651 至 +0.1875）。CycDelta 的 Δ-MAE 比内层训练中位数高 0.0113（−0.0032 至 +0.0467）。CycDelta 验证折每次只有 12–17 个可测增量，ncAA 有 18–26 个绝对测量；每一外折每一方法只训练一个种子。固定权重的重新推理已经核对，随机初始化重新训练的环境敏感性另见主参考实验。
