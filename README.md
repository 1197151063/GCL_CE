# SGL 官方基线、共享 Fast 与 BPR 采样实验

## 多步反事实：CL 权重转移

独立入口，用来测量把一部分 InfoNCE 权重从某个度数组转给另一组后，哪些用户的排序改善：

```bash
python run_counterfactual.py --dry-run
python -u run_counterfactual.py --device cuda
```

默认 Yelp2018、seed=2021。固定独立 Fast BPR 和 Fast CL，L2 使用官方流。先在 fit 图上训练，在第 10 和 20 轮分别复制模型及 Adam 状态到内存中的分支。每个起点有 1 个不加权对照和 12 个转移分支：用户侧与物品侧各做头、中、尾三个组之间的六个有向转移。记录第 1 步、第 1 轮和第 5 轮的探针 Recall@20，以及同起点、同训练长度的分支差值。分支完成后，主模型沿原来的 Fast 训练轨迹继续。

每个 batch 只对一个侧的两组重复 anchor 行进行转移，默认移动两组行数较小者的 25% 对应的权重，因此反方向转移量相同。该侧的 batch 总权重不变，另一侧和全节点候选集合不变；若任一组缺席，本 batch 不转移。JSON 中记录实际转移量和缺席 batch 数。BPR/CL/L2 采样按相同 seed、轮次和步次配对，并逐轮核对正负样本指纹；增强图也使用相同的随机流。这个干预改变整条 InfoNCE 行的系数，不单独改变正对齐或 log-partition。

探针仅从原始 `train.txt` 留出，fit/探针不重叠，留出时保留每个原活跃用户和物品的训练边覆盖。分组使用 fit 图度数，所有分支使用相同探针和 fit 图。原始 `test.txt` 不读取；探针不用于早停或调权。与常规隐式反馈训练一致，BPR 排除已知 fit 正例来抽负例，探针标签不进入负采样逻辑。此结果测量干预响应，不与完整训练图的正式 test 分数拼表。

默认输出在 `runs/counterfactual_v1/yelp2018/`，包含 `influence.json`、`influence.md`、配置及日志；不保存权重。可用 `--fork-epochs 10 --horizons 1 5 --sides user` 缩小实验范围，重跑用新 `--output-dir`。第一版输出影响矩阵，不自动学习最终的节点权重。

若要测**头部权重的剂量曲线**，使用同一入口：

```bash
python run_counterfactual.py --study head-dose --dry-run
python -u run_counterfactual.py --study head-dose --device cuda
```

默认在初始化及第 10、20 轮分叉，各比较 Fast 原始权重和 0%、10%、20%、30%、40%、50%、60% 七个头部用户 CL 份额。这里的份额是 **用户侧 InfoNCE 行系数之和的占比**，不等于梯度占比。用户头部系数为目标份额除以 Fast 头部份额，非头部统一乘以对应的补足系数，因此保留头部组内以及中、尾之间的相对权重。每整轮用户侧系数总和等于原 Fast 行数；单个 batch 的总系数可以波动，日志记录其范围。物品侧、所有样本流、增强图及候选集合不变；0% 仍保留头部用户作为其他行的 key。

结果写入 `runs/head_dose_v1/yelp2018/`。比较相同起点的一步、一轮、五轮的整体和分组 Recall@20，同时报告尾部物品条件 Recall。固定探针报告同一用户跨视图的正对余弦，以及其他 anchor 的 softmax 对各组负 key 的概率质量（去掉对角线、不重新归一化）。后两者是诊断量，不是实际训练流的梯度，也不单独识别正对齐与排斥的因果贡献。可用 `--head-shares` 改剂量，用 `--horizons 1 5 20` 拉长观察；短期最优不等于收敛后最优。组间转移矩阵与剂量扫描分别回答“预算从谁那里来”和“头部应该分多少”。

## 当前归因实验（2026-11-12）

上一轮官方Full（代码名official）Recall@20=0.067432，共享Fast为0.068949；头部用户Recall@20分别为0.045276和0.046795。本轮不重复官方Full，也不跑额外R-Rec Full；固定所有组的Fast BPR正例顺序、负例和增强图，拆分收益来源。新增入口仍在本项目：

```bash
conda activate cyj
cd /home/lhb/cyj/GCL_CE
git pull --ff-only
mkdir -p runs/attribution_v4
nohup python -u run_attribution.py --device cuda > runs/attribution_v4/console.log 2>&1 &
tail -f runs/attribution_v4/console.log
```

默认只跑Yelp2018、seed=2021，完整train训练，每epoch在test评估，以Recall@20连续10轮无严格提升早停，不保存`.pt`。八组依次运行，结果位于`runs/attribution_v4/yelp2018/`，每组均有config/epochs/diagnostics/result。每完成一组就更新`attribution.json`与`attribution.md`，包含配对差值、相同epoch差值、CL×L2交互项及采样指纹检查。旧结果不覆盖。

| 组名 | CL anchors | L2三元组 | 特殊干预 |
|---|---|---|---|
| baseline | 官方均匀用户流 | 官方流 | 原BPR-only Fast基线 |
| cl-fast | 独立Fast流 | 官方流 | 仅改变CL分布 |
| l2-fast | 官方均匀用户流 | BPR batch | 仅改变L2来源 |
| cl-l2-fast | 独立Fast流 | BPR batch | 与前3组构成CL×L2对照 |
| shared-fast | BPR同一Fast batch | BPR batch | 对比cl-l2-fast，检验batch共享关系 |
| head-up | 与baseline逐条相同 | 官方流 | 仅提高用户头部CL行权重 |
| head-down | 与shared-fast逐条相同 | BPR batch | 仅降低用户头部CL行权重 |
| no-cl | 无CL | 官方流 | 与baseline比较CL的影响 |

head-up将用户头部的CL加权质量从约10%调到34.26%；head-down反向调回10%。两组都同时调整非头部权重，使实际采样分布下的用户CL期望权重仍为1，排除整体CL强度变化；组内相对权重、物品CL和全节点候选集合不变。它们重加权的是完整InfoNCE行，不是单独放大正对齐项，不能将收益直接归因为吸引项。

各组记录BPR/L2负例指纹，汇总时核对需要固定的正负样本流；CL与L2效应在对方的两个水平下分别比较，并记录交互项。独立Fast CL与共享Fast的边际分布相同，但随机梯度耦合及更新顺序不同。样本流独立不意味着改变期望目标，batch共享关系的比较需与CL分布干预区分。

归因入口默认开启`--decompose-cl`。固定Fast探针把InfoNCE精确拆为`-正对logit + logsumexp(所有候选logits)`，记录各用户/物品度数组的两项梯度范数及其对BPR梯度方向的投影，并核对两项梯度之和等于完整CL梯度。log-partition包含正例，不能称为纯负例惩罚，也不等同于几何uniformity。探针不按每组训练流采样，且不含L2/Adam，因此是模型状态诊断，不能替代训练期望梯度或因果干预。

`python run_attribution.py --dry-run --device cuda`仅查看计划。`--runs cl-fast l2-fast`可选子集，但应使用新输出目录；`--skip-completed`只跳过计划、源码与配置一致的已完成组，不续训未完成组。源码变化会终止后续调度。下载整目录后，可在本地调用`rrec_experiments.attribution.summarize_study(path)`重新汇总，路径不依赖服务器绝对位置。

本轮八组数据返回前，尚不能确认CL、L2、共享关系或头部权重哪一个是主要原因。一个seed与test选模用于观察现象，不报告统计显著性。

---

以下是原始基线入口`run.py`的说明：默认只跑Yelp2018，一个seed=2021，四组实验，不保存`.pt`。它用于对齐[SGL-Torch](https://github.com/wujcan/SGL-Torch)，与上面的八组归因入口分开。已返回服务器成绩：official Recall@20=0.06743、NDCG@20=0.05542；BPR-only Fast记录最佳为0.06781；shared-fast为0.06895，接近但未达到用户报告的0.0695。

源码、四个数据集、测试和启动脚本在同一Git项目；兼容现有Python3.8/PyTorch2.0.1环境，不依赖PyG或torch-sparse。

## 服务器启动

停止旧任务后同步代码；套件运行期间保持代码不变。源码指纹变化会阻止后续组继续启动，避免混版本。

```bash
conda activate cyj
cd /home/lhb/cyj/GCL_CE
git pull --ff-only
python run.py --dry-run
```

前台只需 `python -u run.py --device cuda`。后台运行：

```bash
mkdir -p runs/sgl_reference_v4
nohup python -u run.py --device cuda > runs/sgl_reference_v4/console.log 2>&1 &
tail -f runs/sgl_reference_v4/console.log
```

结果默认写到 `runs/sgl_reference_v4/yelp2018/{official,shared-fast,fast,full}/seed_2021/`。再次重跑请加新目录，例如 `--output-root runs/sgl_v3_repeat`。不覆盖非空单次运行目录，不自动续训；`--skip-completed` 仅跳过源码指纹、协议和配置都一致的已完成组。两个Bash入口仍只进入服务器目录后调用Python，不探测环境。

## 默认四组

| 组别 | BPR采样 | InfoNCE交互 | L2交互及负例 |
|---|---|---|---|
| official | 官方均匀用户采样 | 与BPR相同 | 与BPR相同 |
| shared-fast | 随机排列、遍历训练边 | 与BPR同一batch，保留重复anchor | 与BPR同一三元组，含同一负例 |
| fast | 随机排列、遍历训练边 | 固定官方流 | 固定官方流 |
| full | 边权重1/(用户度+1)，有放回抽M条 | 与fast逐条相同 | 与fast逐条相同 |

官方采样是均匀、有放回抽活跃用户，再均匀抽其训练正物品。本项目用边权重 `1/d_u` 实现相同联合分布，区别于R-Rec的Full权重 `1/(d_u+1)`。M为训练交互数。

同seed、同epoch，official/fast/full的CL anchor顺序、默认L2三元组一致；四组的初始化、两张增强图和每轮更新次数一致。shared-fast与fast的BPR顺序及负例也逐条一致，但shared-fast将CL和L2一起切换到BPR batch，复现直接替换官方采样器时的共享关系。fast/full只改变BPR。固定L2用于排除正则项的采样混淆，`--l2-source bpr` 可单独改变fast/full的L2来源。

每轮记录三条正交互采样指纹：official/fast/full之间的CL/L2指纹应相同；shared-fast内部BPR/CL/L2指纹应相同。三元组负例复用由训练路径保证并有一步更新测试覆盖。config和控制台记录实际采样策略，CL头部系数占比按实际重复anchor统计，另记unique数。

优先检验0.0695的共享设置可单跑：

```bash
python -u run.py --device cuda --variants shared-fast
```

shared-fast与fast的差异同时包含CL采样、L2采样及共享关系，不能仅凭两组差值归因给CL。当前Yelp训练边中头部用户占34.26%，所以shared-fast的用户CL anchor约34.26%来自头部；均匀用户流约10%。全部同类型节点作为InfoNCE候选的规则不变。

可选无CL对照：

```bash
python run.py --device cuda --variants official shared-fast fast full bpr-fast bpr-full
```

仅跑主比较可用 `--variants fast full`，但应先检查official基线。

## 官方实现对齐

核对来源：[模型与训练](https://github.com/wujcan/SGL-Torch/blob/main/model/general_recommender/SGL.py)、[采样](https://github.com/wujcan/SGL-Torch/blob/main/data/sampler.py)、[模型配置](https://github.com/wujcan/SGL-Torch/blob/main/conf/SGL.ini)、[全局配置](https://github.com/wujcan/SGL-Torch/blob/main/NeuRec.ini)。

- 完整train.txt训练，不划验证集，也不读取valid.txt；Yelp原始train/test已与官方逐对核对，ID与交互一致。
- 64维、3层、含初始层的四层表示平均；无自环、对称度归一化。用户和物品分别Xavier uniform初始化，fan大小分别对应两张表。
- ED每视图固定保留 `int(M*(1-drop_ratio))` 条交互，再双向化及归一化。每epoch两张图，各batch共用。
- InfoNCE保留重复batch anchor，不unique；全部同类型节点作候选，包含正对。单方向view1→view2，两个视图均反传，全节点key不detach。用户与物品损失相加。
- BPR、加权CL、L2按batch sum优化，Adam lr=0.001。先分别backward再执行同一次Adam更新，与官方总损失一次反传数学等价。日志损失除以当轮交互数，仅用于阅读。
- Yelp CL权重0.1、温度0.2、ED比例0.1、L2系数1e-4。
- 未归一化点积排名全部物品，排除训练正例；每个用户使用自己的NDCG分母，再平均。
- 每轮test评估，以Recall@20选择最好模型，连续10轮无严格提升即停，最大1000轮。NDCG记录但默认不参与选择；`--selection-objective recall+ndcg` 是不同协议。
- 单seed默认2021，与官方配置一致。PyTorch独立随机流和官方NumPy/Cython的随机序列不会逐位相同；对齐分布、目标和更新规则，不承诺单次分数恰好等于参考值。

最好参数只在CPU内存保留，结束时恢复并计算指标和诊断，不保存权重文件；结果明确记录test参与选择。

## 后续CL方法干预

fast/full默认不改变CL。方法变体在新目录显式指定：

```bash
python run.py --device cuda --variants fast full --cl-sampler fast --output-root runs/cl_fast_method
python run.py --device cuda --variants fast full --cl-sampler full --output-root runs/cl_full_method
python run.py --device cuda --variants fast full --cl-anchor-mode unique --output-root runs/cl_unique_method
```

`--cl-sampler` 默认user，替换后在fast/full间仍相同。它使用独立CL随机流：`--variant fast --cl-sampler fast` **不等于shared-fast**。official和shared-fast始终按各自固定共享策略决定CL与L2来源；其实际来源写入effective字段，不受这两个采样选项覆盖。unique变体按去重节点平均，再乘batch交互数，保持sum目标的量级，避免将重新加权与整体强度变化混淆。

若共享Fast收益重现，再依次拆分因素（各用新目录）：

```bash
# 相对默认fast，仅改为独立Fast CL，L2仍固定官方流。
python run.py --device cuda --variants fast --cl-sampler fast --output-root runs/cl_fast_only
# 再把L2切到BPR；与shared-fast比较时，两者CL分布相同，但batch共享关系不同。
python run.py --device cuda --variants fast --cl-sampler fast --l2-source bpr --output-root runs/cl_fast_bpr_l2
```

旧A–D/shared-full已移除。本次重新使用shared-fast名称，但其官方目标、超参和数据协议与历史版本不同；v3历史结果保留原目录与Git历史，新代码使用v4协议，不混合不同版本的结果。

## 诊断、汇总和其他数据集

输出config.json、epochs.jsonl、diagnostics.jsonl、result.json；套件完成生成summary.json/summary.csv。记录提交号、实际源码指纹、数据指纹及 `experiment_protocol=official_sgl_attribution_v4`，拒绝混合不同协议、源码及CL设置。汇总增加shared-fast相对official/fast/full的配对差值。

用户/物品分别按完整训练度数分组：head最高10%、tail最低50%、其余middle，零度为cold。输出用户组Recall/NDCG、物品对整体Recall贡献及物品条件组内Recall。

梯度探针默认第0轮、每25轮和最好模型计算，`--diagnostics-every 0` 可关闭。探针统一每交互平均损失，与重复anchor的sum训练具有相同BPR/CL比例；固定Fast探针用于比较模型状态，并非实际采样器整轮期望梯度，不含L2或Adam变换。记录范数比、余弦、冲突比例、表示范数、uniformity和对齐距离。

```bash
python -m rrec_experiments.summarize runs/sgl_reference_v4/yelp2018
```

四数据集仍保留，`--datasets all`才运行全部。Amazon-Book、iFashion保留官方README的配置；Gowalla保留此前ED探索配置，不宣称官方成绩复现。CLI可覆盖预设，实际值写入config。

## 测试与验证边界

```bash
python -m pip install -e '.[test]'
python -m pytest -p no:capture -q
```

测试覆盖官方目标损失/梯度/Adam一步更新、shared-fast训练路径与同batch总损失的一步更新对照（含负例L2）、共享与独立Fast CL区分、重复anchor、全节点key梯度、均匀用户采样、BPR-only流隔离、定数量ED、数据与排名掩码、早停、源码变更终止，以及Python3.8兼容。

详见VALIDATION.md。本次未运行shared-fast完整GPU收敛训练，不能承诺达到0.0695。一个seed不用于报告统计显著性。

所有参数：`python -m rrec_experiments.train --help`。
