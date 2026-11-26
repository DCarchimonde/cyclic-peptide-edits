# 归档输入和参照来源

| 文件（相对于 R0_Cho/source） | 来源和用途 |
|---|---|
| `outputs/{default50,short15,known_TP1,known_TP2,known_TP3,known_TP4}/fold/` | 本项目生成的六套 HighFold 复合物。每套包括未松弛坐标、逐残基置信度 JSON、环状索引记录和给出 rank 1 ipTM 的折叠日志。每个上级目录中的 `verified_scores.json` 是交接时保存的指标。 |
| `outputs/{short15,default50}/6y6c_0.pdb` | 本项目两次示例采样生成的骨架，用于几何对齐。固定骨架的序列记录分别位于 `mpnn/seqs/` 和 `repeat_fixed_backbone/`。 |
| `outputs/prepared/6y6c.pdb` | 本项目从 6Y6C 的 TREM2 A 链准备的受体，将原作者编号 20–137 重编号为 1–118。对应关系见同目录的 `6Y6C_A_mapping.tsv`，原始与处理后文件的指纹见 `input_hashes.json`。[结构 6Y6C](https://www.rcsb.org/structure/6Y6C) · [PDB 数据使用条款](https://www.rcsb.org/pages/policies) |
| `outputs/published/figure2_and_esi_transcription.csv` | 从 Cho 等图 2 与补充图 S1/S3/S5/S7 记录的序列和数值，用于与本项目复折结果并排比较。出版社文件的地址与指纹在同目录的 `sources_sha256.tsv`。[论文](https://doi.org/10.1016/j.bmcl.2025.130512) · [补充材料](https://ars.els-cdn.com/content/image/1-s2.0-S0960894X25004214-mmc1.docx) |
| `outputs/cache/cached_summary.json` | 由作者公开 notebook 的已存输出汇总，其中 6Y6C 有 170 条模型记录，覆盖 34 个骨架与序列组合。它与上面六个独立复跑复合物分别计数。[作者 notebook 固定版本](https://github.com/hongliangduan/AI-Guided-Design-of-Cyclic-Peptide-Binders-Targeting-TREM2-Pipeline/blob/7acdbbbb707f256cacd5f6f3e838bbe63a62dd40/scripts/rf_mpnn_highfold_code.ipynb) |

归档复折使用公开 HighFold，环肽在前，采用单序列、无模板、`model_1_multimer_v3`、一个 recycle、种子 0。论文研究评估了 1,500 个计算复合物。论文相应的逐例坐标以及到 TP1–TP4 的历史编号映射，在本次交接材料中缺失。TP4 的实验结合证据来自原文。[Cho 等](https://pubmed.ncbi.nlm.nih.gov/41435973/)

| 归档计算使用的上游实现 | 固定版本 |
|---|---|
| Cho 等的公开流程 | [7acdbbb](https://github.com/hongliangduan/AI-Guided-Design-of-Cyclic-Peptide-Binders-Targeting-TREM2-Pipeline/tree/7acdbbbb707f256cacd5f6f3e838bbe63a62dd40) |
| CycleDesigner | [0fa160e](https://github.com/hongliangduan/CycleDesigner/tree/0fa160e6def0ad4c971bdcba0ac15265b1e285ff) |
| RFdiffusion 兼容驱动 | [820bfdf](https://github.com/RosettaCommons/RFdiffusion/tree/820bfdfaded8c260b962dc40a3171eae316b6ce0) |
| ProteinMPNN | [8907e66](https://github.com/dauparas/ProteinMPNN/tree/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57) |
| HighFold | [278aab2](https://github.com/hongliangduan/HighFold/tree/278aab210114af9c28b00f95664f70c9f4056dfd) |

## 可选的原始资料

重评分直接使用 `outputs/` 中的文件。若要查看作者 notebook 的逐行输出，可在 `R0_Cho` 中执行下方命令。脚本从上述固定提交下载 notebook 至 `source/upstream/rf_mpnn_highfold_code.ipynb`，核对 SHA-256，再从已保存的单元格文本导出表格至 `runs/author_cache/`。也可自行下载后放在同一路径。[notebook 原文件](https://raw.githubusercontent.com/hongliangduan/AI-Guided-Design-of-Cyclic-Peptide-Binders-Targeting-TREM2-Pipeline/7acdbbbb707f256cacd5f6f3e838bbe63a62dd40/scripts/rf_mpnn_highfold_code.ipynb)

```bat
conda run -n cyc_r0 python -X utf8 source\scripts\audit_cache.py source\upstream\rf_mpnn_highfold_code.ipynb runs\author_cache
```

如要查看沉积时的完整受体，可将下方原文件分别保存为 `source/upstream/6Y6C.pdb` 和 `source/upstream/5UD7.pdb`。首跑读取的处理后 6Y6C 坐标另存于 `outputs/prepared/6y6c.pdb`。[6Y6C 原文件](https://files.rcsb.org/download/6Y6C.pdb) · [5UD7 原文件](https://files.rcsb.org/download/5UD7.pdb)
