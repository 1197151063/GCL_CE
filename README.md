# SGL 采样对照实验

这个独立本地仓库用于检验：Yelp2018 上，Full Sampling 为什么在 SGL 中比 Fast Sampling 更差，以及是否存在头部用户的推荐监督与对比约束失衡。

使用 PyTorch 完成模型计算，安装 NumPy 供 PyTorch 互操作使用，无需 torch-geometric / torch-sparse。支持 CPU 和 CUDA；Apple MPS 不支持本项目使用的稀疏图计算，Mac 上使用 CPU。原始 R-Rec 仓库与数据不会被修改。

## 安装

使用原生架构的 Python 3.10 或更高版本。当前电脑已经用原生 ARM Python 3.11 创建了 `.venv` 并安装依赖，可以直接使用。

新环境安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -p no:capture -q
```

如果 macOS 上 `python3` 实际是 Rosetta 的 x86_64 解释器，可能没有兼容的新 PyTorch wheel，应改用原生 ARM 解释器。CUDA 服务器可按 PyTorch 官方说明安装匹配的 CUDA wheel，再安装本项目。

这里的测试命令关闭 pytest 的终端捕获插件，用于绕过当前 Anaconda Python 的原生 readline 导入崩溃，不影响测试内容。其他正常环境也可以直接 `python -m pytest -q`。本机验证的依赖版本见 `requirements-tested.txt`。

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

它默认读取同级 `R-Rec/data/yelp2018`，也可添加 `--dry-run` 或其他参数。等价的完整命令为：

```bash
.venv/bin/python -m rrec_experiments.suite \
  --data-dir /Users/chenyijun/Desktop/R-Rec/data/yelp2018 \
  --output-dir runs/yelp2018 \
  --seeds 0 1 2 \
  --epochs 1000
```

默认是 8 组 × 3 个随机种子，共 24 次完整训练；这个命令不会自动在后台运行。Yelp2018 的全节点 SGL 计算量较大，建议在 CUDA 服务器执行。迁移时复制代码、提供相同格式的数据，并修改 `--data-dir`。

只预览命令、不开始训练：

```bash
.venv/bin/python -m rrec_experiments.suite \
  --data-dir /Users/chenyijun/Desktop/R-Rec/data/yelp2018 \
  --output-dir runs/yelp2018 --dry-run
```

只跑四组主对照：添加 `--variants A B C D`；只检查历史共享 batch 差异：添加 `--variants shared-fast shared-full`。

单次运行示例：

```bash
.venv/bin/python -m rrec_experiments.train \
  --data-dir /Users/chenyijun/Desktop/R-Rec/data/yelp2018 \
  --output-dir runs/single_C --variant C --seed 0 --device cuda
```

每次运行保存配置、数据指纹、代码提交号、训练统计、最好模型、验证与测试分组指标。默认不覆盖非空输出目录。`suite --skip-completed` 仅跳过配置相同且有 `result.json` 的已完成运行；中断的单次训练没有自动断点续训功能，使用新目录重跑该次实验。

## 小数据端到端验证

```bash
.venv/bin/python -m rrec_experiments.suite \
  --data-dir tests/fixtures/tiny --output-dir runs/smoke \
  --seeds 0 1 --epochs 2 --batch-size 16 --dim 4 --layers 1 \
  --ks 2 3 --selection-k 2 --eval-every 1 \
  --diagnostics-every 1 --diagnostic-batch-size 8 \
  --anchor-chunk-size 3 --threads 1 --device cpu
```

这会实际完成 16 次微型训练并汇总，验证所有路径能够工作。微型数据的指标不用于支持 Yelp2018 的科学结论。

## 验证与测试协议

- 数据目录需包含 `train.txt`、`test.txt`；格式为每行 `user_id item_id item_id ...`，ID 非负，拒绝重复边与跨划分重叠。
- 如果存在 `valid.txt`，直接使用。否则按固定 `--split-seed 20261109` 从每个有至少两条交互的用户中抽取约 10% 训练边作为验证集，至少保留一条训练边。不同训练 seed 使用相同划分。
- **默认拟合图不包含验证边。** 这与原快照使用全部原始 train.txt 建图、直接在 test 上早停不同，绝对指标不能直接当作历史结果复现。
- 默认固定 1000 个 epoch，所有组每 epoch 的更新次数相同；依据验证集 `Recall@20 + NDCG@20` 选择最好 checkpoint，再评估一次测试集。`--patience 0` 默认关闭提前终止，以保证相同更新预算；显式设正数会允许各组提前停止，实际步数会记录。
- 验证排名排除拟合训练边；测试排名排除训练边与验证边。不使用测试集选择 epoch 或过滤训练负例。训练负物品从用户未发生训练交互的物品中均匀抽取，不访问验证或测试正例。
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
.venv/bin/python -m rrec_experiments.summarize runs/yelp2018
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

同时修复了实验运行所必需的负采样、模型/训练/评估接口不一致，并将验证集选择、独立 L2、独立随机流、稳定全节点损失和诊断显式化。因此本项目用于**受控验证机制**，不宣称复现原先未提供的可运行版本及其绝对分数。

所有参数：`.venv/bin/python -m rrec_experiments.train --help`。
