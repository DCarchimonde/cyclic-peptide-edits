# 在 PyCharm 运行 B 线

在 Anaconda Prompt 的仓库根目录运行 `SETUP_WINDOWS.cmd B`。用 PyCharm 单独打开 `B_PAMPA`，绑定 `B_ENV_OK` 后显示的 Python 3.12 解释器，运行 `B_01_Quick`。以下是从 `B_PAMPA` 目录手动安装和运行同一个 CPU 环境的等价命令。

```bat
conda create -n cyc_b --override-channels -c conda-forge python=3.12 pip -y
conda activate cyc_b
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs\requirements_edit_cpu.txt
python -X utf8 run_b.py quick
```

| PyCharm 配置 | 计算 | 新输出 |
|---|---|---|
| B_01_Quick | 用已拟合编辑模型重算 Kelly 两个位点，并给 12 种方法已有预测重计指标 | kelly.csv、rescore.json |
| B_02_Baseline | 从 PAMPA 表重训原有 200 棵 ExtraTrees | audit/audit.json |
| B_03_Pairs | 从实测表和化学图重建母子配对与排序组 | pairs/edits.csv、pairs/summary.json |
| B_04_Train_CPU | 下载并校验冻结 ESM-2、ChemBERTa，构建位点特征，重训本研究轻量模型和基线 | encoder_run.json、model/metrics.csv、model/per_parent.csv、model/uncertainty.csv |
| B_05_Series_CPU | 按 8 个独立化学系列逐折重训本研究模型 | series/metrics.csv |
| B_06_Conformation | 在相同分子和完整候选集合上重建构象消融 | conformation/ |
| B_07_Train_CUDA | 在支持的 GPU 上提取两种编码，再于 CPU 拟合同一组模型 | encoder_run.json、model/metrics.csv |
| B_08_External | 核查 Ly 等发表的三个候选面板，用冻结的四个模型预测符合定义的位点 | summary.json、candidate_audit.csv、frozen_overlap.csv、encoder_exclusions.csv、predictions.csv |
| B_09_Measured_Squares | 在冻结实测表内找齐同方案的母体、两个单点和双点，比较四者的 PAMPA | summary.json、squares.csv |
| B_10_Square_Selection | 已知两个单点的实测值时，对同一母体的多个双点候选进行盲选对照 | summary.json、by_series.csv |
| B_11_Source_Panels | 在冻结表以外的文献来源中寻找单点位点对照，保存四个冻结模型的预测 | summary.json、source_panels.csv、source_options.csv、predictions.csv |

每次运行都会打印独立的 runs 子目录。quick 和 baseline 的输入已随仓库提供。pairs、train、series 会按需从作者固定版本获取原始测量表，conformation 会获取 4D 描述符表。两个 CSV 的直接下载地址、落盘路径和 SHA-256 均见 [data/sources.json](data/sources.json)。train 和 series 的首次运行需要从 Hugging Face 获取两个模型；权重约 44 MiB，后续沿用本机 cache。可以从能访问模型仓库的机器复制完整 cache/weights 目录，代码会核验权重。CPU 与 GPU 的特征分别缓存，配对构建和两种训练作业按顺序复用同一份缓存。要重新生成特征，可将对应 features 目录改名后再运行。

固定测试共有 1,545 个选项、460 个可排名母体、1,028 条可度量增量。历史对照和当前模型比较见 README.md。外部模型的原生拟合各有专用环境、训练结构重叠表及命令，见 results/references/README.md（固定留出）和 results/references_series/README.md（八系列）。quick 会读入已归档的外部预测统一重评分。4D 模式读取已发布的构象描述符，原子轨迹不在这个计算步骤中。

如使用 RTX 4090 的 Linux 实例，安装同版本 torch==2.5.1+cu124，索引改为 https://download.pytorch.org/whl/cu124，随后运行 `python -u run_b.py train --device cuda --threads 2`。查看 encoder_run.json 中的 GPU 名称、CPU/GPU 单分子特征差值和输出路径。

## Ly 等的外部位点比较

在 B 窗口用 `cyc_b` 选择 `B_08_External`，或在本目录运行 `python -X utf8 run_b.py external`。首次运行尝试从美国国家医学图书馆 PubMed Central 的公开存档取得作者结构 CSV。如果网络不可用，在论文的 Supporting Information 下载 **SMILE data (CSV)**，原样另存为 `data/raw/ly2024_smiles.csv`，重跑同一入口。[Ly 等，J. Med. Chem. 2024，DOI 10.1021/acs.jmedchem.3c02162](https://doi.org/10.1021/acs.jmedchem.3c02162) · [PMC 原文及附件](https://pmc.ncbi.nlm.nih.gov/articles/PMC10946398/)

表 3 的 14a 和表 4 的 18a、19a 各与同编号 c、d、e 三个实测选项比较。逐个核对手性结构、可逆的单处主链 N-甲基化和对称等价位点。每个母体须有至少两个不同的合适选项，只有模型能表示整组时才输出该组预测。未解决的化学结构或编码问题分别列在 candidate_audit.csv 和 encoder_exclusions.csv。这些面板彼此相关，同属一篇研究。

对冻结八系列分别比较整分子和完全去目标 N-甲基的骨架。任一精确重叠或指纹相似度达到原方案的 0.80 即标记有关联，另查 6,941 条原 PAMPA 记录中的精确重叠。该结果覆盖本项目的实验数据。四个头读取原提交 `19fe374` 的拟合权重。输入步骤只使用结构，数值预测写入独立的 `runs/external_…/predictions.csv`。在原文核对实验时，表中是 **−log₁₀Pe**，相对母体的 log₁₀Pe 增量取母体表值减去改造体表值。优先比较同一母体内的首选位点与首选损失，并结合原表的重复测量误差解读。

控制台末行 `B_EXTERNAL_OK` 表示核查已完成。rankable 为有至少两个合适位点的面板数，scored 为完整获得预测的面板数，linked_series 为与受查结构或骨架关联的冻结系列数，snapshot_matches 为在原 PAMPA 表中完全重合的受查化合物数。summary.json 保存来源与指纹。原始第三方 CSV 仅在用户本机使用。

## 实测双位点组合

用现成的 `cyc_b` 环境运行 `B_09_Measured_Squares` 或 `python -X utf8 run_b.py squares`。入口只读取已随包分发的冻结配对表，逐条沿用原单点实验的质控、系列、划分与测量方案。一个可用的四分子组包括同一方案下都有定量实测的起始母体、在它两个不同位置分别新增一个甲基所得的两个分子，以及由这两条路径得到的同一个双点产物。起始母体可能已带其他 N-甲基。沿两条路径均须能用归档原子映射对上另一个位点；未解决的映射单列计数。

`squares.csv` 保留四个分子、四条原始配对的标识及实测 log₁₀P 值。每组记录双点相对母体、相对两个单点中较好者的增量，以及双点增量和两个单点增量之和的差。后者大于零表示超过对数标度下的简单相加。汇总按原系列与数据划分列出，多个组可共用分子；次数仅描述这批实测数据。表中“提高”按已记录的点估计和数值精度计算。各比较未设实验显著性阈值。零个合格组也会正常报告。


## 用已测单点选择双点

在 PyCharm 打开 `B_PAMPA`，项目解释器选 `cyc_b`，运行 `B_10_Square_Selection`。手动配置时脚本选 `run_b.py`，参数填 `square-selection`，工作目录选 `B_PAMPA`。使用本地 CPU 和随包冻结的配对表，已有 B_09 的运行目录可以保留。控制台完成行以 `B_SQUARE_SELECTION_OK` 开头，回传此行及本次的 `summary.json`。

在同一来源、系列、测量协议、骨架和母体内，至少有两个不同双点产物可选才计入排序。三种方法分别按对数通透性可加、按较好的单点和按均匀随机选择。两个单点已有实测是本分析的输入条件。双点实测值仅用于事后计算选中最优的概率和相对最优的通透性损失。相同预测随机打破平局，使用其精确期望。多个路径指向同一产物时先合并，再计一次选项。先对同一母体的多个测量面板取均值，再按系列和数据划分报告。单一选项的面板可用于描述可加性偏差。Kelly、Townsend 等来源各自的样本数见 `interactions.csv`。本项是 B 论文的组合改造扩展分析。

## 从新增文献寻找单点位点对照

在 PyCharm 的 B_PAMPA 项目中选择 `B_11_Source_Panels`，点击运行。沿用 `cyc_b` 和本地 CPU。手动配置使用脚本 `run_b.py`、参数 `source-panels`、工作目录 `B_PAMPA`。

程序比较原始冻结表的全部来源与 LiteFold 镜像中 CycPeptMPDB 的 PAMPA 专表，原作者的结构、文献化合物编号、检测限和 PAMPA 字段用于还原候选对照。同篇文献中，同一化学母体至少要能通过两种不同的单点主链 N-甲基化得到带实测记录的产物。允许母体本身未测，此时仍可以比较子体。同结构的记录若存在不同实验值，保留原值供原文回查。只使用能往返验证的结构编辑。

在至少两个选项有点估计的面板上，四个冻结模型给全部可表示的选项打分。预测输入只含结构和位点。先保留预测，再回原论文核对测量方法、实验分组、单位及检测下限。`priority_review` 计数要求面板内全部有效结构选项有点估计、母体和选项在旧原始表中均无精确重复，并通过与八个冻结系列的整分子及去甲基骨架相似性筛查（Tanimoto < 0.80）。B_08 已研究的 Ly 2024 单独标记。来源尚待核实的面板及编码失败记录也会落盘。数据卡的行级 train/test 划分不参与评测。

首次自动下载固定版本的 PAMPA 专表，约 45 MB。若下载失败，可将下方固定文件原样保存为 `B_PAMPA/data/raw/litefold_pampa.jsonl`，然后再次点击该配置。SHA-256 为 `af478487cffd11d79d0554662fccff40d4bdba9377d70d9ea6087a46db33db20`。该步只在本地生成结果。请回传 `B_SOURCE_PANELS_OK` 行和本次 `summary.json`；具体原文复核可使用 `source_options.csv` 中的文献编号。

[PAMPA 专表固定下载](https://huggingface.co/datasets/LiteFold/CycPepMPDB/resolve/0a897997d32c66c1570490d77b15ffbc4abf3410/tables/peptide_cycpeptmpdb_CycPeptMPDB_Peptide_Assay_PAMPA.csv.jsonl) · [镜像数据卡](https://huggingface.co/datasets/LiteFold/CycPepMPDB) · [CycPeptMPDB 数据库原文](https://doi.org/10.1021/acs.jcim.2c01573)

## B_12 同母体位点成对排序

在 PyCharm 的 B_PAMPA 项目中选择 `B_12_Pairwise_Rank`。脚本为 `run_b.py`，形参为 `pairwise-rank`，工作目录为 `B_PAMPA`。调用原有 `cyc_b` 环境和 B_04 的 CPU 编码缓存。在 GitHub ZIP 中更新文件时，合并到原仓库可复用 `cache/` 和已有运行记录。

本轮比较两个训练头的首选位点。两者使用相同的分折、网络宽度、随机种子、对照和训练目标。一个读取原子化学与编辑位点，另一个增加全局和位点对齐的肽序列表征。对同一母体、同一实验条件下的两个实测变体，用分数差学习哪个位点较好。实测并列提供对称目标，每个可排序母体的训练总权重相同。训练、早停与测试分别使用原八折的 train、valid、test 系列。早停由验证母体首选损失决定。训练出的分数只表示优先顺序。

首要比较为融合排序头相对同折重训 CycDelta 的逐母体首选损失差。负值表示融合排序头在这组母体上损失较小。另报相对仅化学排序头、原回归头和随机选位的差异、按系列成对重采样的区间、各系列明细。八个已用于既往比较的系列属于开发证据，独立来源的整组实测可用于下一步验证。输出 `site_scores.csv`、`comparison.csv`、`per_series.csv`、`paired_regret.csv` 和各折参数。请回传 `B_PAIRWISE_RANK_OK ...`、`comparison.csv` 和 `paired_regret.csv`。
