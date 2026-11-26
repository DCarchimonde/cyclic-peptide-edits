# 第一批真实数据与准备方式

配对构建和构象分析使用两份上游 CSV。仓库提供派生输入和预测，原始 CSV 由 `scripts/fetch_data.py` 从作者仓库的固定提交获取。

## 已取得的数据

| 文件 | 实际内容 | 固定来源 |
| --- | --- | --- |
| `data/raw/cycpeptmpdb_all.csv` | 7,451 行、244 列；含分子结构、来源、多个实验终点及检测限备注 | CycPeptMP 作者仓库，commit `e7160fad03d1f43cd3c0179a540e445a8b12979b` |
| `data/raw/cycpeptmpdb_4d.csv` | 5,160 行、25 列；含分子结构、PAMPA、构象描述符和文件对应 ID | CycPeptMPDB-4D 作者仓库，commit `5772a0b3d013ebd7c1f2c0768cf90a5705d59fe5` |

下载 URL、SHA-256、文件大小和引用保存在 `data/sources.json`。两份上游仓库的 MIT 声明保存在 `data/raw/`。下载脚本复用校验通过的本地文件。4D CSV 的 MD5 为 `e4b957982bd572aba5029963804e0524`，与 Zenodo v3 页面公布值一致。按记录 ID 与原始全集对齐后，5,160 条的 SMILES 文本与 PAMPA 数值均一致。

引用：

- CycPeptMPDB 原始数据库：[10.1021/acs.jcim.2c01573](https://doi.org/10.1021/acs.jcim.2c01573)。
- 原始数据镜像所在的作者方法仓库：[akiyamalab/cycpeptmp](https://github.com/akiyamalab/cycpeptmp)，方法论文 [10.1093/bib/bbae417](https://doi.org/10.1093/bib/bbae417)。
- 构象数据：[Gobliu/CycPeptMPDB-4D](https://github.com/Gobliu/CycPeptMPDB-4D)，论文 [10.1038/s41597-026-07984-9](https://doi.org/10.1038/s41597-026-07984-9)，[Zenodo 数据页](https://zenodo.org/records/21274695)。

## 已完成的准备

运行：

```bash
python scripts/fetch_data.py
python scripts/prepare_pampa.py
```

第一条命令检查已保存文件，缺失时从固定 commit 下载；第二条命令生成：

- `data/processed/pampa.csv`：6,941 条有结构和 PAMPA 数值的记录，来自 36 个来源。
- `data/processed/pampa_4d.csv`：5,160 条构象子集的测量记录，沿用原始实验标签。
- `data/processed/data_summary.json`：实际计数与处理规则。

主表含 6,889 个上游 `Structurally_Unique_ID`，4D 子表含 5,140 个。这些是数据提供者的结构标识计数；准备脚本未用 RDKit 做化学标准化或独立结构去重。重复测量保留，不能把记录数或构象帧数当作新的独立标注分子数量。

## 主要字段

| 字段 | 含义 |
| --- | --- |
| `molecule_id` | 上游结构标识；相同分子的多次记录使用相同 ID |
| `record_id` | 上游测量条目的 ID，保留原始记录对应关系 |
| `smiles` | 原始表的异构 SMILES；后续需解析并保留立体化学 |
| `target` | PAMPA 通透系数的十进制对数，单位基于 cm/s；不是脂水分配系数 logP |
| `source` | 原始文献或专利来源标签 |
| `group` | 与 `molecule_id` 相同，用于保持同一上游结构 ID 的记录在同一划分 |
| `endpoint` | 固定为 `PAMPA_log10_cm_per_s`，未混入 Caco-2 等终点 |
| `label_relation` | 数值点、可能的上下界或待复核备注 |
| `detection_limit_notes` | 原表检测限说明，保留原文；4D 子表通过记录 ID 补入 |

## 对测量界限的处理

首轮主表的标记计数为：

| 标记 | 条数 | 初始回归怎样使用 |
| --- | ---: | --- |
| `eq` | 6,604 | 可作为首轮探索性回归输入 |
| `floor_or_left_censored` | 240 | 保留原始值，先不当作精确回归标签 |
| `ceiling_or_right_censored` | 18 | 保留原始值，先不当作精确回归标签 |
| `requires_assay_review` | 79 | 原表有检测限备注，待按对应实验确认 |

`eq` 的含义只是“数据库给出数值，且本轮没有发现界限标记”，不是已经逐条核实了实验精度。脚本将 -10、-4 边界和检测限备注显式保留。原表的两列检测限备注不总能直接对应到 PAMPA，因此本轮不从文字自行推算新的精确界限。4D 子表有 4,925 条 `eq` 记录。

## 接下来直接用于什么

1. 用 `eq` 子集建立一个简单的化学结构基线，先看到实际误差与不同来源的表现。
2. 进行来源留出时，用 `source` 选定来源，再按化学身份处理跨来源重复。仅按来源拆分仍可能把同一分子放到两边。
3. 在正式模型比较前，用 RDKit 解析、标准化保留手性的结构，建立最终分子身份；同一身份的不同 SMILES、重复测量、构象不得跨训练与测试集。
4. 后续统计母体—改造物对。现有 6,941 条记录不等于 6,941 个改造对，当前尚未统计配对数据。
5. 将 4D 作为原数据库子集的空间信息扩展。其主表长度为 6、7、10 个残基。

这里尚未创建正式训练/验证/测试划分。初始按来源留出的实验用于项目启动，不自动等同于最终跨骨架评测。
