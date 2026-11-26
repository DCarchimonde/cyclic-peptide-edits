# 单处主链 N-甲基化配对和位点排序

从 v0.1.0 的 6,941 条 PAMPA 输入出发，17 条未确定立体信息，其余 6,924 条共有 6,872 个完整带手性分子图。按实验子库和骨架对齐，共得到 3,410 个有同协议母体测量的增量对、1,112 个至少有两位点实测的排序母体，对应 2,377 个排序选项。母体是否定量不影响组内排序。675 个排序母体有同协议的定量读数，395 个在整个快照中没有母体，42 个同协议母体无法定量。对虚拟母体的改变量留空。

| 数据 | 系列 | 可用编辑行 | 定量母子对 | 排序母体 |
|---|---:|---:|---:|---:|
| 训练 | 3 | 2,751 | 2,358 | 646 |
| 验证 | 3 | 26 | 24 | 6 |
| 测试 | 2 | 1,545 | 1,028 | 460 |
| 总计 | 8 | 4,322 | 3,410 | 1,112 |

冻结数据来自八个化学系列、68 个实验条件，覆盖 917 个全去目标 N-甲基后仍区分手性的骨架。测试集的 458 个排序母体属 Kelly，2 个属 Le Roux。跨系列最高带手性 Morgan 半径 2／2048 位 Tanimoto 相似度为 0.78125。相同母体、变体和各帧服从 molecule_split.csv。排序组内包括 965 个双位点、142 个三位点、4 个四位点和 1 个五位点。

每条数据含两份完整异构 SMILES、从母体到变体的逐原子映射和两侧编辑氮。原子索引从零开始。甲基化必须通过恢复端基甲基重建原始分子图。端基、立体化学、环闭合和骨架保持，等价选择并为一个。该定义覆盖 N 与 Cα–C(=O) 相邻的主链酰胺。全部确认正值取 log₁₀[P/(cm·s⁻¹)]；同协议同图重复时取对数中位数，保留各记录 ID。−10、−4、限值和表面传输为零分别追查，部分显示零传输的作者正值按冻结质量规则排除。附 source_counts、records、parent_evidence 和 coverage_stages 行级清单。

从B_PAMPA 目录执行以下命令。

~~~bash
python -m pip install rdkit==2026.9.2
python scripts/pair_data.py
sha256sum results/pairs/edits.csv
~~~

冻结编辑表 SHA-256 为 c35bc14c0ae71ac2f73a2f4cc51a39d776e87b53546e19bea706647fdcd6f735。运行需 configs/wang2015_pampa.csv 和原包中的 data/raw/cycpeptmpdb_all.csv。原始 CSV SHA-256 为 16d09a19e1a7251f165fdaf89992a878028268a032ca0279269bf6aecf58c2cb。协议、浓度、介质、来源和复核位置见 configs/pair_protocols.json。

## 实验核查、检测限和适用性

- 40 个结构在快照中有多条记录，19 个跨来源。按来源和手性结构归并会减少 21 条来源内重复，最终同实验面板的定量单元没有重复记录。五个可归属 PAMPA 的明确数值上限均无同面板定量单编辑对端。两端均可量化时才记改造增量。详见 qualification.json、interval_edges.csv。
- Townsend 作者的 3,086 条工作簿记录，逐条通过编号、立体化学、未简化分子图校验。36 个重建子库与作者 Expt 一一对应，2,866 个正值取 log 后在原库两位小数内，205 个零值对应 −10，15 个空格对应 −4。该系列 2,212 个实测对都是保持手性的 Leu→NMe-Leu，作者所列质量差距 CH₂ 小于 0.00006 Da。645 个排序组内变体的质量字段全相同。[论文与公开工作簿](https://doi.org/10.26434/chemrxiv.13335941.v1) · [子库和实验方法](https://escholarship.org/uc/item/6c45v4cz)
- Kelly 的 1,519 条均连回作者原表。对应的稳定同位素标记位置在 1,018 个增量对及 458 个排序母体内一致。显示传输为零的 41 个化合物仍有正数原表通透性，作为质量筛选敏感性恢复时，Kelly 来源为 1,057 个改造对、461 个排序组。[作者正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8514148/) · [补充材料](https://doi.org/10.1021/jacs.0c06115.s001)
- Furukawa 原表的 688 个名称全部对应，680 个定量实测与数据库一致。8 个原表破折号在数据库记作 −10。两个显示零传输仍有定量点的训练候选按冻结规则排除。Wang 的 62 个编号据补充表自身的 log 列和 SEM 重录，54 个有数值。化合物 12、16 有低回收率；主分析采用其余 52 个数值。[Furukawa 补充表](https://doi.org/10.1021/acs.jmedchem.6b01246.s001) · [Wang 补充表](https://ars.els-cdn.com/content/image/1-s2.0-S022352341530012X-mmc1.pdf)
- 独立系列以同一论文的相关化学库为起点。合并完整图相同、去 NMe 骨架相同、带手性指纹相似度达到 0.80 的系列。当前八个系列两两都通过比较并保持独立。此固定划分以组数和选项规模安排训练、验证、测试。脚本会在任意两系列达到阈值时提示重新合并，防止将新增近邻直接纳入旧划分。

## 真实输入与同协议选择

examples/record1_graph.npz 是 CycPeptMPDB_record:1 的真实输入示例，含 85×7 的原子特征、170 条有向边和每条边的 3 个属性。与序列对应的 11 个单体和闭环边见 record1_residues.csv 和 record1_graph.json。该条为环孢素 A，在 2006_Rezai_1 的 PAMPA 测量为 −6.60。

Kelly L1 子库 7 的同一母体 #6618 为 −7.24，D-Leu4 位 N-甲基化 #6645 为 −6.85（增量 +0.39），L-Leu6 位 #6622 为 −5.48（+1.76）。examples/kelly_site_choice.csv/.png/.sdf 提供逐位点对照、原子定位和二维结构。本例据稳定组 ID 在查看模型结果前选定。

Hosono v1 的 CP1 与三个变体的完整手性结构在 Wang 表中也有测量。Leu4 位在 Hosono 中由 −6.52 至 −5.66（+0.86），在 Wang 中由 −5.91 至 −6.12（−0.21）。Wang 所在的系列已用于验证集，此处用于说明实验协议对结论的影响。[Hosono v1](https://doi.org/10.26434/chemrxiv.12272861.v1) · [Hosono 正式版](https://doi.org/10.1038/s41467-023-36978-z)

补充核查与制图可从B_PAMPA 目录复跑。原表由脚本从作者公开附件获取，也可指定已下载的 PDF 和工作簿目录。

~~~bash
python -m pip install openpyxl
python scripts/audit_pair_series.py
python scripts/audit_pair_qualification.py
python scripts/audit_townsend_source.py
python scripts/audit_furukawa_source.py
python scripts/show_pair_examples.py
python -m unittest discover -s tests -p test_pair_data.py -v
~~~
