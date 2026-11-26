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

每次运行都会打印独立的 runs 子目录。quick 和 baseline 的输入已随仓库提供。pairs、train、series 会按需从作者固定版本获取原始测量表，conformation 会获取 4D 描述符表。两个 CSV 的直接下载地址、落盘路径和 SHA-256 均见 [data/sources.json](data/sources.json)。train 和 series 的首次运行需要从 Hugging Face 获取两个模型；权重约 44 MiB，后续沿用本机 cache。可以从能访问模型仓库的机器复制完整 cache/weights 目录，代码会核验权重。CPU 与 GPU 的特征分别缓存，配对构建和两种训练作业按顺序复用同一份缓存。要重新生成特征，可将对应 features 目录改名后再运行。

固定测试共有 1,545 个选项、460 个可排名母体、1,028 条可度量增量。历史对照和当前模型比较见 README.md。外部模型的原生拟合各有专用环境、训练结构重叠表及命令，见 results/references/README.md（固定留出）和 results/references_series/README.md（八系列）。quick 会读入已归档的外部预测统一重评分。4D 模式读取已发布的构象描述符，原子轨迹不在这个计算步骤中。

如使用 RTX 4090 的 Linux 实例，安装同版本 torch==2.5.1+cu124，索引改为 https://download.pytorch.org/whl/cu124，随后运行 `python -u run_b.py train --device cuda --threads 2`。查看 encoder_run.json 中的 GPU 名称、CPU/GPU 单分子特征差值和输出路径。
