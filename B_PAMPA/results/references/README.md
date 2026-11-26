# 同一母体与未见系列上的文献模型比较

测试集含 1,545 个候选位点、460 个可排序母体和 1,028 条有实测增量的边。Kelly 提供 458 个可排序母体，Le Roux 提供 2 个。每项正式比较均用项目的预先固定分折和共同覆盖清单。各方法在相同母体下给同一批位点排序。首选损失按母体等权计算，观测并列最佳均计命中，预测并列时取并列选项的期望损失、期望命中率。两个独立测试系列分别报告，不用分子重采样拟造系列层面的置信区间。

## 已保存逐位点预测的复跑结果

| 文献模型 | 首选损失 ↓ | 首选命中率 ↑ | 增量 MAE ↓ |
|---|---:|---:|---:|
| ncAA-RepDistill，冻结 Phase2 并按项目分折训练读出 | 0.1527 | 63.04% | 0.4114 |
| 同折重训 CycDelta | 0.2660 | 41.30% | 0.3861 |
| Chemprop 全分子 D-MPNN 作差 | 0.2757 | 43.04% | 0.4320 |
| 项目显式编辑模型 | 0.2511 | 46.09% | 0.3715 |
| CycDelta 已发布原权重，监督训练有重叠 | 0.1191 | 63.26% | 0.3406 |

**ncAA-RepDistill。** 使用作者的 Phase2 E0_B0、词表、图距离注意力和 768 维 CLS 表征。分子保持单一 SMILES 窗口。2,720 个训练分子测量和 35 个验证分子测量均按来源协议及结构去重。训练折标准化后，按附录协议使用单隐层 256 单元读出和随机种子 42。验证集选择轮次，自建导出过程与作者原生训练函数在同一验证集数值交叉核对。最后将两个绝对预测相减。原 nc-CPP 的有标签读出没有用于该榜单。Phase2 无标签训练结构中，1,209/1,545 个测试选项至少一端出现过；合并 Phase2 验证后为 1,303/1,545。原 nc-CPP 监督训练名单也覆盖 1,303 个测试选项的一端或两端，可按逐边记录核对。去除一组内任一端有适配暴露的整组后，剩 107 个选项、29 个可排序母体和 67 条实测 Δ。在交付的逐位点预测中，首选损失 0.1945，命中率 51.72%，Δ MAE 0.4241。已核对 Phase1 两份结构 CSV，共 219,150 条训练结构和 24,350 条验证结构。所有 SMILES 均由多个独立组分组成，与本项目完整单连通环肽的整分子重叠为零。覆盖核对不延伸至教师模型的全部早期训练语料。

原作者 ncAA-Scan 把已观察的单 token 替换扩展成双向边，在 Murcko 骨架内比较位点编辑的 Δ 相关和前十项改善比例。本文测试边中有 21 个也满足其已发布八种骨架、原测试集合和循环序列（允许反向）相邻规则。20 个同时具有相同的两端实测值。原文与代码见 [Yu 等](https://doi.org/10.1186/s13321-026-01311-5) 与 [固定提交](https://github.com/cihebiyql/ncAA-RepDistill/tree/c88857132c028ee463fafb9167a51bf54177c6f2)。

**CycDelta。** 使用作者有显式氢的单体 Uni-Mol、31 个残基描述符、原生 CycGNN 与 PAMPA 实验标记。逐边把实际共同化学母体作为参照，通过原生编码计算 f(变体−母体)−f(0)，并与公开推理函数比对。虚拟母体无需填写实验值。原版模型覆盖全部测试选项，但其原有监督 OD_train 已包含 1,467/1,545 个测试选项的一端或两端。两端均未出现的完整分组只余 24 个可排序母体、78 个选项与 54 条 Δ，原版的损失 0.2600、命中率 33.33%、Δ MAE 0.4458。同折版本从种子 0 重训同一原生网络，在训练结构上重新拟合描述符均值和方差，训练使用 2,308 个指定母体的实测增量。24 个验证增量选择最佳训练轮次。相对于作者的大规模按来源随机选择母体，此处改变配对和验证量以匹配指定问题。

[Qian 等预印本](https://doi.org/10.64898/2026.09.18.752569) · [CycDelta 固定提交](https://github.com/cyclontx/CycDelta/tree/0a686c93f4b10fd3318e39f38893cdc9cbd0a848) · [Uni-Mol 权重](https://huggingface.co/dptech/Uni-Mol-Models/tree/9f19c45c718192888a1c8a1c905f69f0755ea502)。

**Chemprop。** 用 Chemprop 2.2.1 原生有向键消息传递与均值池化。全分子图保留手性和闭环，128 维隐藏层、三层消息传递。种子 42 从训练折绝对值拟合，使用验证集 RMSE 选轮次。用于相同系列同母体的绝对预测作差比较。参考[环肽渗透性系统基准](https://doi.org/10.1186/s13321-025-01083-4)。

## 首轮与恢复运行的对应

首轮保留下了上游提交、权重哈希、固定输入、评价量与逐预测文件指纹。执行主机更换后从这些输入重建了适配程序与当轮预测。分析当前交付的逐母体结果时，使用 `comparison.csv` 的可复跑数值。下表为第一次运行的汇总，可与以上当轮逐位点结果对照。

| 文献模型 | 首轮首选损失 ↓ | 首轮命中率 ↑ | 首轮 Δ MAE ↓ |
|---|---:|---:|---:|
| ncAA-RepDistill，冻结 Phase2 并同折重训读出 | 0.1527 | 63.04% | 0.4114 |
| Chemprop 全分子 D-MPNN 作差 | 0.2270 | 50.22% | 0.4236 |
| 同折重训 CycDelta | 0.2318 | 48.48% | 0.4094 |
| 项目显式编辑模型 | 0.2511 | 46.09% | 0.3715 |
| CycDelta 已发布原权重，监督训练集存在重叠 | 0.1191 | 63.26% | 0.3406 |

CycDelta 公开权重两轮均为 0.1191／63.26%／0.3406。恢复运行的 CycDelta 重训为 0.2660／41.30%／0.3861，Chemprop 为 0.2757／43.04%／0.4320（依次为损失、命中率、Δ MAE）。CycDelta 两轮的最佳验证轮次从第 1 轮变成第 2 轮。Chemprop 两轮都选第 3 轮，验证 RMSE 分别为 0.83190 和 0.83229。恢复运行在同一主机又独立重训一次，重新生成 CycDelta 单体。三种重训方法以及原 CycDelta 权重各自的 4,272 行预测均与复跑的第一遍逐字节相同。ncAA 读出两轮各选第 4 个训练轮次。恢复运行的验证 RMSE 为 1.848352，与作者原生函数的 1.848352 相符。ncAA 重训重复固定了本次原生编码器输出。Chemprop 和 ncAA 的逐轮验证曲线也相同。CycDelta 双线程两遍的前两轮验证曲线相同，随后从第 3 轮开始分岔，两遍均选择第 2 轮。逐轮数值见 `repeat_validation_curves.csv`。在同一 AMD 主机、同一单体张量和同一输入上，再把 Torch 和 OMP 的线程数由 2 改为 1。两次单线程训练的所有验证轮次与逐位点预测均逐字节相同。单线程选择第 1 轮，验证 MAE 为 0.303218，测试损失 0.2588、命中率 45.43%、Δ MAE 0.3939。两种线程数的发布权重预测逐项相同。线程设置可以改变本次重训练的数值路径和选轮次；这项控制实验不能解释旧主机漂移的全部因素。结果见 `cycdelta_threads_probe.json`。代码、输入顺序与新初始化张量的指纹、逐模型重复结果见 `repeat_audit.json`。首轮与新主机分别为 Intel Xeon 8573C 与 AMD EPYC 9V74。旧主机未保存可逐项对照的首批梯度、初始化及批次随机数状态，现有证据不能定位跨主机漂移的单一原因。另在同一台 AMD 主机上仅改变 MKL 的兼容模式。Chemprop 相同初值、相同首批 48 个观测在该实验中产生 2.24×10⁻⁸ 的最大预测差和 2.98×10⁻⁸ 的最大梯度差，第一次更新后的权重指纹已发生变化。该对照只测到了第一步，对最终排名的影响未被隔离。结果见 `linear_algebra_probe.json`。

## 运行与文件

从 B_PAMPA 目录执行。先运行配对和项目模型以写出共同支持表。REF 与 CACHE 设在仓库外，供下载大权重及存放本机中间表征。上游源码压缩包、已下载的大权重及输入使用 SHA-256 校验。Linux CPU 环境使用 Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0。CycDelta 的特征化使用其作者给出的 RDKit 2026.3.4、torch-geometric 2.5.3 与 unimol-tools 0.1.3.post1。身份审计及另外两种模型使用 RDKit 2026.9.2。

```bash
export REF="$PWD/../reference_sources"
export CACHE="$PWD/../reference_features"
python scripts/refs_assets.py --out "$REF"
python scripts/refs_overlap.py --external "$REF"
python scripts/refs_ncaa.py --external "$REF" --support results/model/common_support.csv --cache "$CACHE/ncaa"
python scripts/refs_dmpnn.py --support results/model/common_support.csv
PYTHONPATH="$REF/rdkit-cyc${PYTHONPATH:+:$PYTHONPATH}" UNIMOL_WEIGHT_DIR="$REF/unimol" python scripts/refs_cycdelta.py --external "$REF" --support results/model/common_support.csv --cache "$CACHE/cyc"
python scripts/refs_report.py --predictions results/model/predictions.csv results/references/ncaa_predictions.csv results/references/cycdelta_released_predictions.csv results/references/cycdelta_refit_predictions.csv results/references/dmpnn_predictions.csv
```

仓库提供 4,455 个结构的原生 ncAA CLS 向量与结构哈希，特征矩阵为 float32、每个结构 768 维。公开的 NPZ 使用标准 ZIP-LZMA 压缩，所有内嵌 NPY 文件的字节与原归档一致。可在上述建模前以 `python scripts/refs_feature_pack.py --cache "$CACHE/ncaa" --restore` 恢复当前模型所用的冻结向量。使用空缓存可重新执行原生编码。浮点表示可随机器与批次计算路径轻微变化。

要复验同一主机的可重复性，将相同三条建模命令各再运行一次，将 `--out` 指向独立重复目录 `$CACHE/repeat`。给 CycDelta 的第二遍指定单独的 `--cache`，可同时验证重复抽取单体。然后运行下列核对。两个数值路径实验分别写出首批预测与梯度。

```bash
python scripts/refs_repeat_audit.py --external "$REF" --repeat "$CACHE/repeat"
for i in 1 2; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$REF/rdkit-cyc${PYTHONPATH:+:$PYTHONPATH}" UNIMOL_WEIGHT_DIR="$REF/unimol" python scripts/refs_cycdelta.py --threads 1 --external "$REF" --support results/model/common_support.csv --cache "$CACHE/cyc" --out "$CACHE/single_$i"
done
python scripts/refs_thread_probe.py --single-a "$CACHE/single_1" --single-b "$CACHE/single_2"
python scripts/refs_backend_probe.py --out-prefix "$CACHE/probe_default"
MKL_CBWR=COMPATIBLE python scripts/refs_backend_probe.py --out-prefix "$CACHE/probe_compatible"
```

可在独立的环境安装附加依赖。先安装上述版本的 NumPy、pandas、scikit-learn、PyYAML、SciPy、joblib、tqdm 和 CPU 版 PyTorch。以下命令中的第二份 RDKit 只在上面的 CycDelta 运行中启用。

```bash
python -m pip install rdkit==2026.9.2 torch-geometric==2.5.3 pytorch-lightning==2.4.0 lightning==2.4.0 torchmetrics==1.5.2 chemprop==2.2.1 numba==0.62.1 addict lmdb
python -m pip install --no-deps unimol-tools==0.1.3.post1
python -m pip install --no-deps --target "$REF/rdkit-cyc" rdkit==2026.3.4
```

`comparison.csv` 含主比较与剔除暴露后的比较。逐母体、逐系列、逐组结果见同名文件前缀。`comparison_series_ranges.csv` 给出独立测试系列的实测指标范围；在两个系列上，也等于逐个删去系列后所余系列的范围。各 `*_predictions.csv` 每条边一行。原始首轮运行的摘要和文件指纹保存在 `verified_first_run.json`，本轮逐分子数值由固定上游重新生成。`overlap_by_edit.csv` 提供逐端上游分组。`ncaa_scan_overlap.csv` 复核原排序问题的实际交集。核对范围只包含本实验下载、校验并读取的公开结构表。

该包提供本项目的适配程序、训练产物和派生数值。ncAA-RepDistill 仓库声明 MIT。CycDelta 固定提交未附单独 LICENSE，脚本从作者固定提交获取实现和公开权重。每份上游数据与权重按原发布方条件使用。
