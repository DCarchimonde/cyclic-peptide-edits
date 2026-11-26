# 从实际编辑位点预测 PAMPA 改造效果

主试验使用三个训练、三个验证和两个测试化学系列。所有方法共用 4,272 个变体选项，测试的 1,545 个选项全部可编码。50 个训练选项的标准 α 氨基酸片段不连通，已排除并逐项记录。覆盖的 4,455 种分子有 32,240 个编码残基位置，其中 7,957 个化学片段投影为 X。

## 复跑命令

Python 3.12.14，PyTorch 2.5.1+cpu，RDKit 2026.09.2。预训练权重来自固定提交，文件哈希在 `configs/edit_encoders_sha256.json`。以下命令从 B 工程根目录执行。初始化两套权重约需下载 46 MB；编码与训练均在 CPU 上完成。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/requirements_edit_cpu.txt
python scripts/encoder_cache.py --weights .cache/weights --fetch --cache .cache/encoders --demo-data data/processed/pampa.csv
python scripts/pair_chemistry.py data/raw/cycpeptmpdb_all.csv .cache/graph.json
python scripts/prime_edit_cache.py --graph .cache/graph.json --records data/processed/pampa.csv --weights .cache/weights --cache .cache/encoders
python scripts/fit_edits.py --edits results/pairs/edits.csv --weights .cache/weights --cache .cache/encoders --out results/model --threads 2
python scripts/edit_invariance.py --edits results/pairs/edits.csv --model-dir results/model --cache .cache/encoders
python scripts/edit_example.py --edits results/pairs/edits.csv --model-dir results/model --cache .cache/encoders
python scripts/series_sensitivity.py --edits results/pairs/edits.csv --weights .cache/weights --cache .cache/encoders --out results/model/series_sensitivity --work .cache/series_fits --threads 2
```

`prime_edit_cache.py` 重放按分子字符串长度组织的固定批次，在几种浮点实现间可能仍有末位差异。归档权重与全部预测已经随包交付。使用已训练头处理新配对时，给出 `pair_id,group_id,parent_smiles,variant_smiles,parent_n_idx,variant_n_idx`；氮原子索引为所在 SMILES 从零开始的重原子顺序，程序对单独加回／除去一个主链 N-甲基进行检验。

```bash
python scripts/fit_edits.py --predict-head results/model/fitted/explicit_edit.pt --edits new_pairs.csv --weights .cache/weights --cache .cache/encoders --out new_predictions.csv
```

## 编码器和特征

完整分子进入冻结的 DeepChem/ChemBERTa-77M-MLM，77M 表示 PubChem 预训练化合物数。通过 SMILES 字符位置与 RDKit 原子的精确匹配，取得每个原子的 384 维向量。化学分支提供原子均值、保留手性及闭环信息的 Morgan 指纹和 12 项描述符。原分词器可能漏读方括号手性符号，显式手性图保留这些差别。编辑分支分别读取两个分子的编辑氮、与该氮相隔不超过两键的原子、半径三键的手性 Morgan 环境。每个模块均提供母体向量与该变体减母体的向量。

残基分支以 N–Cα–C(=O) 有向图定位肽链。经参考残基图匹配后，D/L 与主链 N-甲基形式投影到对应标准字母，化学图另保留其身份。无法确认为标准残基的片段用 X，包括 Abu 及部分桥接片段。ESM-2 8M 对 320 维残基隐状态编码。完整头尾肽环选字母序最小的循环移位，重复序列有多个等价起点时逆对齐取均值。有侧链闭环的结构保留有向主链路径，闭环键保存在化学图中。每条配对单独取改造氮在母体与变体中的上下文。参见 [ESM 官方模型](https://github.com/facebookresearch/esm) 和 [ChemBERTa-2](https://arxiv.org/abs/2209.01712)。

## 比较与评价

直接学习增量的四个模型均含 32 个隐藏单元、GELU 与 0.1 dropout。输入分别为全局化学特征、化学加全局 ESM、化学加编辑位置、全局融合加编辑位置与对齐 ESM。标准化只用训练折。AdamW 学习率 0.001，权重衰减 0.001。共有两个等权损失，分别拟合同一协议母体组的已测 Δ 和组内中心化的变体结果。后者也能利用母体未测的排序组。第 5、10、20、35、50 轮在独立验证折（6 个排序母体）比较首选损失，以 Δ-MAE 破平，种子 2026。绝对值基线在训练折去重的分子–协议观测上拟合。原 v0.1.0 的 predict_pair 对真实全局特征重新拟合，单分子编辑输入为零。另有重新训练的 80 棵 ExtraTrees，采用完整 2,048 位手性指纹与 12 项描述符后预测作差。

每个母体的首选损失为候选中实测最佳减去模型首选的实测值。1e−7 内预测并列，取均匀任选的期望损失与命中。若同一母体有多条协议，先算组内结果再对该母体平均。主分析按母体等权，另给各系列等权的结果。对独立系列成组重采样 2,000 次，在同一批系列上计算模型差。主留出仅两个系列，报告其各自表现，95% CI 留空。另做八系列内部验证早停、外部逐一留出，保存每个外折的数据角色、模型选择和预测。

| 主测试方法 | 首选损失 | 首选命中 | Δ-MAE |
|---|---:|---:|---:|
| 随机选位；Δ 恒为 0 | 0.2393 | 47.97% | 0.4436 |
| 同位点并列；Δ 恒报训练中位数 +0.28 | 0.2393 | 47.97% | 0.3746 |
| ExtraTrees 绝对值相减 | 0.2071 | 50.76% | 0.3993 |
| 原 predict_pair | 0.2362 | 47.83% | 0.4161 |
| 化学分支 | 0.2114 | 52.83% | 0.3836 |
| 普通融合 | 0.2247 | 50.65% | 0.8239 |
| 化学加位点 | 0.2428 | 48.04% | 0.4287 |
| 显式编辑融合 | 0.2511 | 46.09% | 0.3715 |

固定留出含 460 个排序母体和 1,028 个定量增量。Kelly 提供 458 个排序母体，Le Roux 提供 2 个。显式模型给约 99% 的测试边正向预测。实测中有 764 个上升、11 个不变、253 个下降；恒判上升的方向正确率为 75.12%，显式模型为 74.93%。模型主要学到了群体中的典型增量。证据不足以支持在未见系列上可靠选择改造位点。

`results/model/record1.json` 核对第一条真实记录的 11 个残基。记录 1 的完整预训练张量、真实母体 #6618 和两种变体的结构图、实际输入、实测与预测列在 `results/model/example/`。#6645 和 #6622 的预测增量分别约 +0.344 与 +0.332，实测 +0.39 与 +1.76，首选损失 1.37。相反的正确例 #5680 的两个已测变体 #5683 和 #5732 的实测值分别为 −5.99 和 −6.69。积分梯度追溯模型的打分差，不能解释为物理因果关系。

绝对预测与直接增量模型通过同一测试集评价。`results/model/predictions.csv` 有 34,176 行（每种模型 4,272 行）。同一母体和同一系列相互连接的配对都在同一折。逐组结果保存于 `per_group.csv`，化学母体等权、系列等权和成组抽样区间分别见 `per_parent.csv`、`per_series.csv`、`uncertainty.csv`、`paired_method_differences.csv`。Kelly 与 Le Roux 的显式首选损失分别为 0.2517、0.1000；ExtraTrees 分别为 0.2078、0.0500。

## 八系列敏感性

八个外折每次只测试一个独立系列。6 个系列有 1,103 个可排序母体，8 个系列提供 3,360 个可定量的 Δ，每个方法给出 4,272 个外折预测。显式模型首选损失 0.2145，随机选择 0.2177，系列成对差 −0.0032，95% 区间 −0.0124 至 +0.0928。显式命中率 50.50%。显式 Δ-MAE 0.4654，外折训练中位数 0.3692，成对差 +0.0962（+0.0233 至 +0.1352）。该结果未显示稳定的择点收益。一级固定留出中的微小 Δ 误差优势也未跨系列保持。详细折分及逐母体、逐系列结果见 `results/model/series_sensitivity/`。
