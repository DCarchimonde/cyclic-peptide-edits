# 绝对通透性历史基线复核

原始代码从 6,941 条 PAMPA 数据重新计算特征并训练 ExtraTrees，随机种子 2026，共 200 棵树。按原来源留出方案训练 5,828 条、测试 730 条，其 MAE 为 0.881469，对照训练均值 0.851938，Spearman 0.021290。训练与留出来源及精确立体化学结构无交集。逐行复评分见 aligned_predictions.csv，逐来源见 source_metrics.csv。

```bash
python scripts/run_baseline.py --data data/processed/pampa.csv --output build/legacy_refit --trees 200 --jobs 2 --seed 2026
python scripts/audit_legacy.py --refit build/legacy_refit
```
