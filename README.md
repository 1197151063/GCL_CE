# SGL 采样对照实验

这个 Git 项目用于检验：Yelp2018、Amazon-Book、Gowalla、iFashion 上，Full Sampling 与 Fast Sampling 在 SGL 中的差异，以及是否存在头部用户的推荐监督与对比约束失衡。

源码、四个数据集、启动脚本与说明都纳入同一个 Git 仓库，无需另一个 R-Rec 目录。使用 PyTorch 完成模型计算，无需 torch-geometric / torch-sparse，支持 CPU 和 CUDA。

## 上传服务器即可运行

项目结构：

```text
/home/lhb/cyj/GCL_CE/
├── data/yelp2018/          # 每个数据集包含 train.txt、test.txt 和校验值
├── data/amazon-book/
├── data/gowalla/
├── data/iFashion/
├── rrec_experiments/      # 模型、采样、训练、诊断与汇总
├── run.py                 # Python 统一入口，默认只跑 Yelp2018
├── run_experiments.sh     # 简单的服务器启动命令
├── run_yelp.sh            # Yelp2018 的便捷入口
├── requirements.txt       # 服务器安装入口
├── scripts/package_project.py
├── tests/                 # 自动测试与微型数据
└── runs/                  # 运行时自动生成日志、指标与汇总，不保存 .pt
```

服务器项目目录为 `/home/lhb/cyj/GCL_CE/`。先激活已有环境，再直接运行 Python：

```bash
conda activate cyj
cd /home/lhb/cyj/GCL_CE
python run.py --dry-run
python -u run.py --device cuda
```

`run_experiments.sh` 只负责进入项目目录并执行 `python -u run.py --device cuda`，使用你已激活环境中的 python，不做环境探测。`run_yelp.sh` 则额外指定 `--datasets yelp2018`。

后台运行：

```bash
conda activate cyj
cd /home/lhb/cyj/GCL_CE
mkdir -p runs
nohup python -u run.py --device cuda > runs/console.log 2>&1 &
```

兼容 Python 3.8，依赖声明为 PyTorch >=1.13、NumPy >=1.21。已有环境能运行时直接启动即可，无需重新建环境。若缺少依赖，在已激活环境中执行 `python -m pip install -r requirements.txt`。

数据读取项目内的 `data/`，结果写入 `runs/`。所有参数可直接追加到 `python run.py`，例如 `--seeds 0`、`--variants A B C D`、`--skip-completed`。

当前默认只跑 Yelp2018，先观察采样与头尾节点的现象：8 组 × 1 个种子（seed=0），共 8 次训练。每轮检查测试集 `Recall@20 + NDCG@20`，连续 10 个 epoch 没有严格提升就恢复最好模型、保存结果并进入下一组，最多 1000 轮。也可添加 `--variants A B C D` 先跑四组主实验。结果默认写到 `runs/yelp2018/`；其他三个数据集保留在项目内，显式添加 `--datasets all` 才会运行全部四个数据集，各自单独汇总。

已有进程不会自动采用新设置。终止旧进程、同步代码后，换新输出目录启动，保留此前日志：

```bash
mkdir -p runs/earlystop_seed0
nohup python -u run.py --device cuda --output-root runs/earlystop_seed0 > runs/earlystop_seed0/console.log 2>&1 &
tail -f runs/earlystop_seed0/console.log
```

日志逐轮显示 `test_score`、`best_epoch` 和 `no_improvement=当前计数/10`。例如第 20 轮最好，之后没有提升，则第 30 轮结束后停止该组。当前不支持接续旧训练的权重/优化器状态，新目录会从头训练。

只跑指定数据集：

```bash
python run.py --datasets amazon-book --device cuda
python run.py --datasets gowalla iFashion --device cuda
```

每个数据集使用显式参数配置，八组对照共享完全相同的设置：

| 数据集 | CL 权重 | 温度 | ED 丢边率 | L2 |
|---|---:|---:|---:|---:|
| yelp2018 | 0.2 | 0.2 | 0.1 | 1e-4 |
| amazon-book | 0.5 | 0.2 | 0.1 | 1e-4 |
| gowalla | 0.3 | 0.2 | 0.2 | 1e-4 |
| iFashion | 0.02 | 0.5 | 0.4 | 1e-3 |

这些数值取自原 SGL.py 的数据集配置。本项目四个数据集统一使用 ED；原代码的 Gowalla 使用 RW，因此这里不宣称复现其 RW 分数。统一的 3 层、64 维、初始化 std=0.01 不变。CLI 显式参数可覆盖预设，例如 `--temperature 0.3`；实际值写入每次运行的 config.json。直接使用底层 train/suite 模块时应显式提供所需参数，预设由统一启动入口应用。

## Git 维护和服务器同步

代码、八份 train/test 数据文件以及对应校验值均由 Git 跟踪。虚拟环境、运行结果由 .gitignore 排除。训练不写入模型权重文件。

后续修改照常 git add、git commit。没有配置远程服务时，可以重新导出 bundle，把最新提交同步到服务器：

```bash
# 本地仓库导出全部提交
mkdir -p dist
git bundle create dist/R-Rec-Sampling-Experiments.bundle --all

# 服务器收到新版 bundle 后，在服务器仓库内更新
git fetch /path/to/R-Rec-Sampling-Experiments.bundle main
git merge --ff-only FETCH_HEAD
```

如果之后配置了 Git 远程仓库，则直接使用常规 push/pull。

## 实验设计

| 名称 | BPR 采样 | CL anchor 采样 | 是否共用 batch | CL |
|---|---|---|---|---|
| shared-fast | Fast | Fast | 是 | 开启 |
| shared-full | Full | Full | 是 | 开启 |
| A | Fast | Fast | 否 | 开启 |
| B | Full | Full | 否 | 开启 |
| C | Fast | Full | 否 | 开启 |
| D | Full | Fast | 否 | 开启 |
| bpr-fast | Fast | — | — | 关闭 |
| bpr-full | Full | — | — | 关闭 |

前两组保留原实现中 BPR/CL 共享 batch 的关系。A–D 构成独立采样的 2×2 对照，同一侧相同采样在各组使用相同随机序列。最后两组在**同一个 SGL 编码器、同一初始化和层数**下关闭 CL，避免与独立 LightGCN.py 配置不同的混淆。

Fast 是每轮随机排列、无放回遍历训练边。Full 严格使用每条边 `1/(user_degree+1)` 的权重，抽取与训练边数相同的样本，**有放回**。这里的度数仅由拟合训练集计算。

CL anchor 先对用户/物品分别去重，候选仍为全体同类型节点，包含正对；key 分支参与反向传播。使用稳定的交叉熵与精确分块计算，checkpoint 重计算避免保存完整的大矩阵，不将全节点损失替换为 batch 内损失。

默认 L2 使用所有组完全相同、独立于 BPR/CL 的 Fast batch 和负采样流。这样采样干预不会同时改变 L2 系数。若需要检查原来 L2 随 BPR 采样的影响，用单独的输出目录添加 `--l2-source bpr`，不要与默认设置混合汇总。

## 跑完整 Yelp2018 实验

在仓库根目录执行：

```bash
./run_yelp.sh
```

它默认读取项目内 `data/yelp2018`，也可添加 `--dry-run` 或其他参数。等价的完整命令为：

```bash
python -m rrec_experiments.suite \
  --data-dir data/yelp2018 \
  --output-dir runs/yelp2018 \
  --seeds 0 \
  --epochs 1000
```

默认是 8 组 × 1 个随机种子，共 8 次训练，每组独立早停；这个命令不会自动在后台运行。整包迁移后不需要修改路径。

只预览命令、不开始训练：

```bash
python -m rrec_experiments.suite \
  --data-dir data/yelp2018 \
  --output-dir runs/yelp2018 --dry-run
```

只跑四组主对照：添加 `--variants A B C D`；只检查历史共享 batch 差异：添加 `--variants shared-fast shared-full`。

单次运行示例：

```bash
python -m rrec_experiments.train \
  --data-dir data/yelp2018 \
  --output-dir runs/single_C --variant C --seed 0 --device cuda
```

每次运行保存配置、数据指纹、代码提交号、训练统计、验证与测试分组指标。最好模型仅在 CPU 内存中保留一份独立副本，结束时恢复并计算最终指标与诊断，不写入 `.pt` 或其他权重文件。默认不覆盖非空输出目录。`suite --skip-completed` 仅跳过配置相同且有 `result.json` 的已完成运行；中断的单次训练没有自动断点续训功能，使用新目录重跑该次实验。

## 小数据端到端验证

```bash
python -m rrec_experiments.suite \
  --data-dir tests/fixtures/tiny --output-dir runs/smoke \
  --seeds 0 1 --epochs 2 --batch-size 16 --dim 4 --layers 1 \
  --ks 2 3 --selection-k 2 --eval-every 1 \
  --diagnostics-every 1 --diagnostic-batch-size 8 \
  --anchor-chunk-size 3 --threads 1 --device cpu
```

这会实际完成 16 次微型训练并汇总，验证所有路径能够工作。微型数据的指标不用于支持 Yelp2018 的科学结论。

如需运行自动测试，先安装测试依赖：

```bash
python -m pip install -e '.[test]'
python -m pytest -p no:capture -q
```

关闭 capture 插件用于绕过本地 Anaconda 的原生 readline 问题，不影响测试内容。服务器正常环境也可直接使用 `python -m pytest -q`。

## 重新打包

在本地仓库提交修改后执行：

```bash
python scripts/package_project.py
```

生成 `dist/R-Rec-Sampling-Experiments-server.tar.gz` 及 SHA-256 文件。包内包含已提交的代码、四个数据集、测试和说明，排除本机 `.venv`、运行输出、缓存及 Git 元数据。`SOURCE_COMMIT` 保存提交号，使服务器无需安装 Git 也能在实验配置中记录代码版本。

## 验证与测试协议

- 数据目录需包含 `train.txt`、`test.txt`；格式为每行 `user_id item_id item_id ...`，ID 非负，拒绝重复边与跨划分重叠。
- 如果存在 `valid.txt`，直接使用。否则按固定 `--split-seed 20261109` 从每个有至少两条交互的用户中抽取约 10% 训练边作为验证集，至少保留一条训练边。不同训练 seed 使用相同划分。
- **默认拟合图不包含验证边。** 这与原快照使用全部原始 train.txt 建图、直接在 test 上早停不同，绝对指标不能直接当作历史结果复现。
- 按当前实验要求，默认 `--selection-split test --patience 10`，每轮依据测试集 `Recall@20 + NDCG@20` 选择最好模型，平分也计入未提升，连续 10 轮未提升即停止。即使指定 `--eval-every 5`，启用早停时仍每轮评估，避免实际多等 50 轮。`--epochs 1000` 是最大轮数，各组每轮更新次数相同、总更新次数可不同，实际步数及停止原因写入结果。
- **测试集参与模型选择，因此报告的测试指标不是独立留出估计。** `config.json`、`result.json`、`summary.json` 都明确记录选择划分及 `test_used_for_selection`。如需验证集选择，显式添加 `--selection-split validation`；如需固定预算，添加 `--patience 0`，此时 `--eval-every` 控制评估间隔。默认仅一个种子，不估计跨种子不确定性。
- 验证排名排除拟合训练边；测试排名排除训练边与验证边。验证划分与拟合图保持原设置。训练负物品从用户未发生训练交互的物品中均匀抽取，不访问验证或测试正例。
- BPR 使用未归一化点积。评估默认使用 `--score cosine`，对应当前原快照公共评估函数中可见的归一化意图；也支持 `--score dot`。请根据历史实验的实际可运行版本确认，并在所有组使用相同设置。

## 头尾指标与梯度诊断

用户和物品分别根据**拟合训练度数**分组：head 为最高 10%（向上取整），tail 为最低 50%（向下取整），其余为 middle；零度节点单列 cold。相同度数按 ID 排序，分组不重叠。

输出包括：

- 用户组内 Recall/NDCG，每个有目标交互的用户等权。
- 物品组对整体 Recall 的贡献，分母为用户的全部目标交互数，四个组相加等于整体 Recall。
- 物品条件组内 Recall：仅在有该组目标的用户上平均，分母为该用户在该组的目标数。它不能与上一项混用，也不能跨组直接求和。
- 每轮头部 BPR 正交互系数占比、CL anchor 系数占比、去重节点数和三个损失。
- 第 0 个 epoch、每 25 个 epoch，以及选中 checkpoint 的梯度诊断；`--diagnostics-every 0` 可关闭。

梯度探针使用独立固定 Fast batch、固定负例及固定增强图，对所有组和 checkpoint 相同。对全体初始 embedding 分别求 BPR 和**加权 CL**梯度，包含 key 分支与图传播贡献，不执行参数更新。按用户/物品头中尾记录梯度范数比、逐节点比例中位数和 P90、余弦中位数、冲突比例及表示范数。

范数比不直接等于有害程度；余弦为负也不单独证明泛化受损。接近零的梯度被排除在逐节点比例/余弦统计之外，计数同时输出。`--ssl-side user` / `item` 可进一步区分用户侧和物品侧 CL，使用独立输出目录。

诊断还记录固定采样节点对的 uniformity：`log mean exp(-2 * squared_distance)`，使用归一化表示，数值更低代表更均匀；同时记录固定正交互的对齐距离。用户/物品全体与各个组分别统计，随机节点对在所有运行和 checkpoint 中固定。

## 汇总与解释

完整 suite 自动生成 `summary.csv`、`summary.json`，也可重新执行：

```bash
python -m rrec_experiments.summarize runs/yelp2018
```

汇总包括均值、样本标准差和匹配种子的成对差值，拒绝混合划分、训练配置、代码提交、PyTorch 版本等不同的结果。`n=1` 的 std=0 只表示没有重复实验，不代表没有不确定性。缺少预定结果时默认拒绝汇总，可显式用 `--allow-incomplete` 查看部分结果。

选中 checkpoint 的梯度、几何诊断也会按种子聚合到 summary 中；逐轮诊断保留在各次运行的 `diagnostics.jsonl`。

- shared-fast/shared-full：检查共享 batch 设置下的原现象。
- A/B 与 shared 组：检查共享 batch 关联性是否改变结论。
- C/B：固定 Full CL，恢复 Fast BPR 的影响。
- D/B：固定 Full BPR，恢复 Fast CL anchor 的影响。
- bpr-fast/bpr-full：同一编码器关闭 CL 后的采样差异。

只有同时观察到头部组性能下降、CL 与推荐梯度失衡/冲突，以及针对性干预恢复，才更支持“头部受过强有害约束”。整体指标上升不能自动推出头部指标上升。

## 与原快照的关系

审阅来源：[R-Rec](https://github.com/1197151063/R-Rec)，提交 `73e9633fbed2a3f2ff981dd30f2b155b5020fb98`。本仓库独立实现实验，不直接导入或复制其失配的公共模型接口。

默认保留 Yelp SGL 的 3 层、64 维、正态初始化 std=0.01、Adam lr=0.001、L2=1e-4、CL=0.2、温度 0.2、边丢弃率 0.1、初始层参与平均。图使用对称归一化，默认包含自环，匹配原代码调用 `gcn_norm` 的意图；`--no-self-loops` 用于单独测试标准无自环 LightGCN 形式。

同时修复了实验运行所必需的负采样、模型/训练/评估接口不一致，并将模型选择划分、独立 L2、独立随机流、稳定全节点损失和诊断显式化。因此本项目用于**受控验证机制**，不宣称复现原先未提供的可运行版本及其绝对分数。

所有参数：`python -m rrec_experiments.train --help`。
