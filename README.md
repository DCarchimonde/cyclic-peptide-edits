# cyclic-peptide-edits

| PyCharm 打开的子目录 | 研究任务 | Conda 环境 | 首次运行配置 |
|---|---|---|---|
| [R0_Cho](R0_Cho/README.md) | 按 Cho 等的结构筛选门槛重评分六个归档 TREM2 复合物，实验参照为 TP4 | `cyc_r0` | `R0_01_Rescore` |
| [B_PAMPA](B_PAMPA/README.md) | 给同一母体的多个主链 N-甲基化位点排序，与实测 PAMPA 通透性比较 | `cyc_b` | `B_01_Quick` |

## Windows 首次运行

在 Anaconda Prompt 中执行。以下命令把仓库放在 `D:\Research`，使用已安装的 Git for Windows 和 Conda。

```bat
if not exist D:\Research mkdir D:\Research
cd /d D:\Research
git clone https://github.com/DCarchimonde/cyclic-peptide-edits.git
cd cyclic-peptide-edits
SETUP_WINDOWS.cmd R0
```

安装程序创建或复用 Python 3.12 的 `cyc_r0`，安装 NumPy，并在 `R0_ENV_OK` 后打印解释器路径。用 PyCharm 2023.3 打开 `D:\Research\cyclic-peptide-edits\R0_Cho`。按 `Ctrl+Alt+S`，进入 `Project: R0_Cho → Python Interpreter → Add Interpreter → Add Local Interpreter`。在 Conda 的 `Existing environment` 中选 `cyc_r0`，也可通过 `System Interpreter` 直接选择刚才打印的 `python.exe`。右上角选择 `R0_01_Rescore`，点击 ▶。预期末行为 `R0_RESCORE_OK complexes=6 passed=0/6`。

然后回到 Anaconda Prompt 的仓库根目录执行 `SETUP_WINDOWS.cmd B`。在 PyCharm 用 `File → Open → New Window` 单独打开 `B_PAMPA` 子目录，绑定 `B_ENV_OK` 后的 `cyc_b` 解释器，运行 `B_01_Quick`。预期末行为 `B_QUICK_OK edits=4322 test_options=1545 methods=12`。两个首跑都在 CPU 上运行，实验输入已随提交提供。

如果下拉框未显示预设，在相应窗口右键顶层的 `run_r0.py` 或 `run_b.py`，选择 Run。两个脚本默认分别执行 `rescore` 和 `quick`。同一入口也可在仓库根目录的 Anaconda Prompt 单独运行：

```bat
conda run --no-capture-output -n cyc_r0 python -X utf8 R0_Cho\run_r0.py rescore
conda run --no-capture-output -n cyc_b python -X utf8 B_PAMPA\run_b.py quick
```

## 以后更新

```bat
cd /d D:\Research\cyclic-peptide-edits
git pull --ff-only
```

每次运行在对应工程的 `runs` 子目录写入新结果。两个工程的本机运行目录和 PyCharm 设置均由 Git 忽略。
