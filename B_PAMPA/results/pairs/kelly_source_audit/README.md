# Kelly 原始实验表核对和纳入敏感性

三个作者工作簿与数据库的 1,519 条记录按实验库和带立体化学的原始序列一一对应。全部 log₁₀Papp 与原表的差值落在小数点后两位取整误差内，最大为 0.004998。用于主分析的 1,018 个有实测增量的母子对和 458 个位点组中，原表的氘标记位置在组内均一致。

41 条记录报告 Avg%T = 0，同时作者表中的最终 Papp 均大于零。作者说明，自动处理赋零的峰在可行时经人工重积分，不能积分的峰剔除。L5 中有 13 条共用 0.14 × 10⁻⁶ cm/s 的数值，这些行的个体定量下限仍待原始峰核实。冻结主分析把零 Avg%T 作为保守的数据质量筛选。每条补充记录的原表、行号和单位可见 source_zero_transmission.csv。公开来源包括[支持材料 S14–S18](https://doi.org/10.1021/jacs.0c06115.s001)及[库 2–4](https://doi.org/10.1021/jacs.0c06115.s002)、[库 5](https://doi.org/10.1021/jacs.0c06115.s003)、[库 1](https://doi.org/10.1021/jacs.0c06115.s004)的三个实验表。

恢复全部 41 条报告正值的样本后，Kelly 有 1,057 个实测编辑增量和 461 个可比较位点组。在有变体模拟数据的完整选项上，2D、水相、双环境的首选损失分别为 0.24048、0.24488、0.23939。冻结分析对应为 0.22847、0.22814、0.23072。结果见 variant_only_sensitivity_metrics.csv。两次训练样本保持一致，共有选项的逐行得分差低于 10⁻¹³。

```bash
python scripts/audit_kelly_source.py --expanded-pairs build/kelly_expanded.csv
python scripts/conform_ablation.py --variant-only --pairs build/kelly_expanded.csv --output build/kelly_sensitivity_4d
```
