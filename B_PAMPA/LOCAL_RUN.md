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
