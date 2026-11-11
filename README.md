# SGL 官方基线与 BPR 采样实验

当前目标是对齐 [SGL-Torch](https://github.com/wujcan/SGL-Torch)，再研究**只改变 BPR 采样**的效果。默认只跑 Yelp2018，一个seed=2021，三组实验，不保存 `.pt`。用户提供的参考 Recall@20=0.0675、NDCG@20=0.0555 是待服务器验证的目标，不是已经复现的成绩。

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
mkdir -p runs/sgl_official_v2
nohup python -u run.py --device cuda > runs/sgl_official_v2/console.log 2>&1 &
tail -f runs/sgl_official_v2/console.log
```

结果默认写到 `runs/sgl_official_v2/yelp2018/{official,fast,full}/seed_2021/`。再次重跑请加新目录，例如 `--output-root runs/sgl_v2_repeat`。不覆盖非空单次运行目录，不自动续训；`--skip-completed` 仅跳过源码指纹、协议和配置都一致的已完成组。两个Bash入口仍只进入服务器目录后调用Python，不探测环境。

## 默认三组

| 组别 | BPR采样 | InfoNCE交互 | L2交互及负例 |
|---|---|---|---|
| official | 官方均匀用户采样 | 与BPR相同 | 与BPR相同 |
| fast | 随机排列、遍历训练边 | 固定官方流 | 固定官方流 |
| full | 边权重1/(用户度+1)，有放回抽M条 | 与fast逐条相同 | 与fast逐条相同 |

官方采样是均匀、有放回抽活跃用户，再均匀抽其训练正物品。本项目用边权重 `1/d_u` 实现相同联合分布，区别于R-Rec的Full权重 `1/(d_u+1)`。M为训练交互数。

同seed、同epoch，三组的CL anchor顺序、默认L2三元组和两张增强图一致。official的BPR也复用该参考流，保留官方共享batch目标；fast/full只改变BPR。固定L2用于排除正则项的采样混淆，`--l2-source bpr` 是显式的耦合L2变体。

每轮记录三条采样指纹：主比较CL/L2指纹应相同，BPR不同。CL头部系数占比按实际重复anchor统计，另记unique数。

可选无CL对照：

```bash
python run.py --device cuda --variants official fast full bpr-fast bpr-full
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

默认不改变CL。方法变体在新目录显式指定：

```bash
python run.py --device cuda --variants fast full --cl-sampler fast --output-root runs/cl_fast_method
python run.py --device cuda --variants fast full --cl-sampler full --output-root runs/cl_full_method
python run.py --device cuda --variants fast full --cl-anchor-mode unique --output-root runs/cl_unique_method
```

`--cl-sampler` 默认user，替换后在fast/full间仍相同。unique变体按去重节点平均，再乘batch交互数，保持sum目标的量级，避免将重新加权与整体强度变化混淆。official参考的anchor采样始终user流，方法实验使用fast/full两组。

旧A–D和shared-fast/shared-full名称已移除，历史结果保留在原目录与Git历史。

## 诊断、汇总和其他数据集

输出config.json、epochs.jsonl、diagnostics.jsonl、result.json；套件完成生成summary.json/summary.csv。记录提交号、实际源码指纹、数据指纹及 `experiment_protocol=official_sgl_fixed_cl_v2`，拒绝混合不同协议、源码及CL设置。

用户/物品分别按完整训练度数分组：head最高10%、tail最低50%、其余middle，零度为cold。输出用户组Recall/NDCG、物品对整体Recall贡献及物品条件组内Recall。

梯度探针默认第0轮、每25轮和最好模型计算，`--diagnostics-every 0` 可关闭。探针统一每交互平均损失，与重复anchor的sum训练具有相同BPR/CL比例；固定Fast探针用于比较模型状态，并非实际采样器整轮期望梯度，不含L2或Adam变换。记录范数比、余弦、冲突比例、表示范数、uniformity和对齐距离。

```bash
python -m rrec_experiments.summarize runs/sgl_official_v2/yelp2018
```

四数据集仍保留，`--datasets all`才运行全部。Amazon-Book、iFashion保留官方README的配置；Gowalla保留此前ED探索配置，不宣称官方成绩复现。CLI可覆盖预设，实际值写入config。

## 测试与验证边界

```bash
python -m pip install -e '.[test]'
python -m pytest -p no:capture -q
```

测试覆盖官方目标损失/梯度/Adam一步更新、重复anchor、全节点key梯度、均匀用户采样、BPR-only流隔离、定数量ED、数据与排名掩码、早停、源码变更终止，以及Python3.8兼容。

详见VALIDATION.md。本机未完成GPU收敛训练；不能宣称已达到0.0675/0.0555。一个seed不用于报告统计显著性。

所有参数：`python -m rrec_experiments.train --help`。
