# Cho 等的 TREM2 环肽复现 · 归档结构重评分

`R0_01_Rescore` 读取六个已归档的复合物。四个来自论文指定序列 TP1–TP4 的 HighFold 复折，两个来自本项目对作者示例的 15 步和 50 步扩散复跑。TP4 是论文报告有实验结合信号的参照。这里的坐标、置信度与日志由本项目的复跑产生。[论文](https://doi.org/10.1016/j.bmcl.2025.130512) · [文件来源与固定版本](source/SOURCES.md)

在仓库根目录的 Anaconda Prompt 执行 `SETUP_WINDOWS.cmd R0`。在 PyCharm 单独打开 `R0_Cho`，选择 `R0_ENV_OK` 后显示的解释器及运行配置 `R0_01_Rescore`。命令行等价入口从本目录运行：

```bat
conda run --no-capture-output -n cyc_r0 python -X utf8 run_r0.py rescore
```

入口先核对归档文件 SHA-256，从折叠日志读取 ipTM，从置信度矩阵计算双向跨链 PAE 均值，将 Å 值舍入至两位小数后除以 31，并从 PDB 重算几何指标。两项结构门槛为 `ipTM > 0.8` 和 `normalized_ipAE < 0.3`。程序将新计算的指标与归档值核对，并写入 `runs/rescore_*/current_inference_summary.csv`；四个论文序列的并排比较在同一目录的 `paper_vs_known_refold.csv`。

```text
R0_RESCORE_OK complexes=6 passed=0/6
```

`0/6` 是本项目这六套归档结构同时达到两项门槛的数量。论文中 TP4 的结合证据来自独立的实验测量。此次重评分的全部输入随仓库提供，只需 Python 3.12 和 NumPy。
