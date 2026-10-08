# ab5（DeepSeek，full day-night）逐天实验轨迹 — Chemprop 有向键 MPNN 路线

## 概览与约定

- **run 身份**：ab5 / 基座 LLM = **GLM-5.1** / **full day-night**（跨基座鲁棒性 run，验证自主科研流水线换不同基座 LLM 后能否复现/逼近同类 run 的最佳模型）/ 共 **6 个 logged day**（Day1–6，无 Day7、**无独立 optimize 目录**——Day6 的 Chemprop HPO 即头部最终模型）。mem.json 共 157 节点（113 knowledge + 11 experience + 17 integration + 16 idea）。
- **最终最强模型**：Chemprop-Polymer dense-MPNN（config=dense_d5_h1200，Day5 发现 Chemprop 有向键 MPNN 突破 → Day6 HPO 引入 DenseNet 风格密集跳跃连接 + 超参优化）。
- **主线轨迹**：数据中心（GIN+trimer 50.19 为 GIN 架构天花板）→ Day2–4 自监督预训练/对比学习/光谱特征/提示条件化**全部失败**（JEPA 塌陷、对比学习不稳定、PCR/PerioGT-Lite/BlochSpec/PCFT 均不破 50.19）→ Day5 **Chemprop 有向键级消息传递突破 43.46**（架构机制主导，机制:拓扑≈14:1）→ Day6 HPO（dense_d5_h1200 单种子 35.48 / 多种子 40.00±2.09）。
- **任务范围**：Day1–6 实验均**只预测 Tg**；Tm/Egc/EA/IP 仅出现在辅助 GAT baseline（单种子），EA/IP 完全未涉及。
- **单位约定**：Tg RMSE 单位 K（开尔文温度差），量纲同 °C；模型直接在原始 K 上回归（Chemprop 系列不做目标归一化）。
- **可信度约定**：所有数字以**代码/结果文件为权威**；mem.json 的 AI 当夜自述仅作线索，凡引用其独有、代码未佐证的数字/结论，标注"（AI 当夜自述，代码未独立验证）"。特殊缩写在首次出现处展开含义。

### day→实验目录映射

| Day | src/ 目录 | 主题 | 代表结果 |
|---|---|---|---|
| 1 | `1_01_data_centric_foundation`、`1_day1_arch_improve`、`1_2025_07_11_ka_gps`(仅代码未跑) | 数据中心 5 结构变换 + 架构改进 | trimer **50.19**；架构改进全负；KA-GPS 未执行 |
| 2 | `2_polymer_jepa_pretrain`、`2_2_polymer_jepa_pretrain`、`2_gps_spse_trimer` | JEPA/对比预训练 + GPS-SPSE | 对比 50.77（乐观单次）；JEPA/GPS-SPSE 全未破 50.19 |
| 3 | `3_day3_pcr_v2`、`3_periogt_lite` | PCR-v2 对比正则 + PerioGT-Lite 对比预训练 | 均失败（PCR 77.50；PerioGT-Lite 多种子 −0.81%） |
| 4 | `4_04_pgft`、`4_4_blochspec`、`4_4_4_blochspec` | PCFT 双编码器 + BlochSpec 光谱 | 均失败（PCFT 52.72 p=0.637；BlochSpec v2 53.25） |
| 5 | `5_6_chemprop_polymer`、`5_day5_polymer_bipartite_v2` | Chemprop 有向键 MPNN + Bipartite v2 | **Chemprop 43.46（突破，机制 14:1）**；Bipartite 53.81 失败 |
| 6 | `6_06_chemprop_hpo`、`6_06_chemprop_pa` | Chemprop HPO + 对比预训练支线 | dense_d5_h1200 单种子 **35.48** / 多种子 **40.00±2.09** |

> ab5 无 Day7、无 optimize 目录，Day6 即 terminal/头部。目录前缀数字为权威 day 归属。

---

## Day 1：数据中心基线（GIN+trimer 50.19）+ 架构改进全负

### 实验目的
Day 1 围绕 Tg 回归展开三个子实验，先固定模型筛"数据表示"，再固定数据改"模型架构"。**（A）Data Centric Foundation**（`1_01_data_centric_foundation`）：固定边感知 GIN 基线，对同一 p-SMILES 做 5 种结构变换——monomer（单体，去 * 封端）、dimer（二聚体，2 个重复单元 RU）、trimer（三聚体，3 RU）、cyclized trimer（环化三聚体，首尾加周期性边）、augmented（trimer + 3 维重复单元索引特征），判定哪种聚合物图表示对 Tg 最有利。**（B）KA-GPS**（`1_2025_07_11_ka_gps`）：把 Kolmogorov-Arnold Network（KAN，用 Fourier 级数替代 MLP 的函数逼近层）注入 GraphGPS 框架做"KAN 增强 GPS"。**（C）Architecture Improvement**（`1_day1_arch_improve`）：在筛出的最佳 trimer 与最差 cyclized 上换两种更强架构——GIN-GAT 混合卷积、带 VirtualNode（虚拟节点，与全图所有真实节点相连的共享枢纽）的 GIN，检验更强架构能否降误差或挽救 cyclized 过拟合。

### 模型架构
**（A）（C）共享基座** `EdgeAwareGINConv`：PyG MessagePassing(add 聚合)，消息=x_j+Linear(edge_attr)（边特征线性投影加到邻居特征上），更新=MLP((1+eps)·x_i+Σmsg)，eps 可学习。`GINModel`=输入线性投影→3 层 EdgeAwareGINConv(hidden 256，每层 BatchNorm+ReLU)→global_mean_pool→MLP(256→128→1)。节点特征 30 维（atom_type 10+degree 6+hybridization 4+aromatic 1+ring 1+H_count 6+chirality 2），边特征 6 维（键型 4+共轭 1+环 1）。**（C）** GIN-GAT（`EdgeAwareGINAttentionConv`）保留 GIN 的 MLP 更新但把 add 聚合换 GAT 式单头注意力（分数 LeakyReLU(W·[h_v‖h_u‖W_edge·e])）；GIN+VirtualNode 在每张图末尾追加虚拟节点连所有真实节点。参数量：GIN≈445K，GIN-GAT 481,028，GIN+VN 478,724。**（B）KA-GPS** 仅在代码层面定义（`FourierKANLayer`→`KANBlock`→`GPSKANLayer`=GIN-KAN+Transformer-KAN+门控融合，4 层 hidden 128，配 RWSE 位置编码）。

### 训练评估方式
统一：Tg 7170 条，72/8/20（train 5162/val 573/test 1435），**seed=71 单一种子**，Adam lr=1e-3，batch 32，max 500 epoch，early stop patience 30，按 val 最优 epoch 选模型后在 test 报 RMSE。（A）5 种变换全部成功处理 7170 分子（图规模随 RU 数线性增长：monomer avg_nodes 11.2→trimer 33.7→cyclized 33.7 边数+1）。（C）挑 trimer/cyclized 各加 VirtualNode 版共 4 组合，训后做诊断（注意力权重统计、虚拟节点表示与 Tg 相关性）。（B）KA-GPS 代码额外加 ReduceLROnPlateau(factor 0.5,patience 15)+weight_decay 1e-5，train.py 设计先训 MLP-GPS 再训 KA-GPS。

### 结果
**（A）5 种结构变换**（单 seed=71）：(a) Monomer **52.71**（基线）、(b) Dimer 51.29、(c) Trimer **50.19（最佳，较 monomer −2.51/−4.77%）**、(d) Cyclized trimer 53.28（最差，过拟合：train 26.5 最低但 test 最高，best_epoch 64 训 94 epoch 最久）、(e) Augmented trimer 50.52（比纯 trimer 略差 +0.33）。**（C）架构改进**（4 变体）：GIN-GAT trimer 51.66、GIN-GAT cyclized 54.01、GIN+VN trimer 53.78、GIN+VN cyclized 51.14——4 个全部差于（A）最佳 GIN+trimer=50.19；注意力诊断显示 GIN-GAT 注意力近似均匀（selectivity 低）、VirtualNode 与 Tg 仅弱相关（均值 |corr| 0.24–0.27）；唯一部分成功是 GIN+VN+cyclized（51.14 好于 GIN+cyclized 53.28，过拟合间隙 26.75→19.18）但仍未超 trimer 基线。**（B）KA-GPS：未执行**——目录只有 dataset.py/model.py/train.py，**无 results.json、无 best_model_kan.pt、无任何结果 csv/txt**；full.log 全文检索无该目录执行痕迹、无任何测试 RMSE 输出；mem.json 也无含 "KA-GPS" 的数字结论。因此旧 summary"KA-GPS 50.8/51.2"的数字既无代码也无日志支撑（疑为编造），弃用；KA-GPS 当夜仅完成代码编写。"GAT 基线 50.5 / PerioGT SOTA 30.8"出自 mem_001 的文献先验、被硬编码进 KA-GPS train.py 仅作对照展示，Day1 代码本身未复算 GAT 50.5。

### 流程
当晚实际跑通并留有结果的是两段。**（A）** `run_all.py` 顺序跑 a→e 共 5 变换（每段约 150–340 秒，cyclized 最久 339 秒/94 epoch），边跑边写 experiment.log、存模型、画学习曲线，最后汇总 results_summary.txt；结论（trimer 最佳、cyclized 过拟合）写入 mem_084。**（C）** train.py 接着在 trimer/cyclized 上顺序跑 4 架构变体，落 results.csv，再由 diagnostic_analysis.py 做诊断生成 summary_report.txt，结论写入 mem_085（"4 variants tested … ALL worse"，相对 GIN+trimer 50.19 均更差）。**（B）KA-GPS** 仅完成代码编写（dataset.py/model.py/train.py），因 full.log 无任何运行痕迹且无结果文件产出，判定当夜未实际执行，不纳入 Day1 可引用结论。Day1 最终落地：**trimer 是 Tg 最小充分表示；在 trimer 上加模型表达力（GIN-GAT/VirtualNode）不升反降，GIN 架构成为下一步要突破的瓶颈**。

---

## Day 2：JEPA 自监督预训练（塌陷）+ 对比学习（小胜但不稳）+ GPS-SPSE（欠拟合）

### 实验目的
统一目标：用 PI1M（Polymer Informatics 1M，实际取 30 万条 p-SMILES）做自监督预训练、再在 Tg 微调，验证能否把 Test RMSE 打到 <50.19（Day1 GIN+Trimer 基线）以下。**（A）JEPA v1**（`2_polymer_jepa_pretrain`）：首次尝试 Joint-Embedding Predictive Architecture（联合嵌入预测架构），学生-教师 EMA + predictor 在 PI1M trimer 图上做"上下文→目标嵌入"预测。**（B）JEPA v2–v6 + SimCLR 风格对比预训练**（`2_2_polymer_jepa_pretrain`）：针对 v1 塌陷与代码 bug 迭代，JEPA 侧逐步引入 VICReg（Variance-Invariance-Covariance Regularization，方差-不变性-协方差正则）防塌陷，另起 SimCLR 风格对比支（NT-Xent + VICReg）做对照，最终定量对比"JEPA vs 对比学习 vs 无预训练"。**（C）GPS-SPSE-Trimer**（`2_gps_spse_trimer`）：放弃预训练换架构，用 GPS（General Powerful Scalable，GIN MPNN + 全局 Transformer）+ SPSE（Shortest-Path Structural Encoding，最短路径结构/位置编码）+ 去噪辅助任务直接监督回归。

### 模型架构
**（A）JEPA v1**：GINEncoder=5 层 GINConv(hidden 256, train_eps, BN, dropout 0.5, **global_add_pool**)；JEPA predictor=2 层 MLP（context 嵌入+位置 pos_dim=3→目标嵌入）；教师=学生 EMA 副本（动量 0.996→0.999）；微调头 Linear(256,128)+Dropout+Linear(128,1)。**（B）JEPA v2–v6** 改动：dropout 0.5→0.1、add_pool→**global_mean_pool**、加节点级 VICReg 损失、EMA 动量爬坡+目标 L2 归一化；v6 用 VICReg lambda_var=50/lambda_cov=50。对比支复用同 GINEncoder(dropout 0.1, mean_pool)+ProjectionHead(→128)，两视图由节点特征 masking+边 dropping 与子图采样生成，损失 NT-Xent(τ=0.1)+VICReg。微调三策略 A（冻前 3 层）/B（端到端）/C（全冻只练头）。**（C）GPS-SPSE-Trimer**：892,073 参数，5 个 GPSBlock=GINEConv MPNN+逐图 Transformer，SPSE 作注意力偏置注入（`SinusoidalSPSE` 对全图最短路径距离矩阵正弦编码→每块可训练线性层投影为标量加到 attention scores），逐块渐进 dropout[0.10–0.30]，去噪头重建被 masking 节点特征（mask_ratio 0.1），global_mean_pool。

### 训练评估方式
**（A）** PI1M 300,000 条→trimer，预训练 15 epoch，batch 256，Adam；微调 500 epoch+早停 patience 30，策略 A/B/C 分别 lr=5e-4/1e-4/1e-3。**（B）** PI1M 同 30 万条，预训练 30 epoch；微调统一 500 epoch/patience 30/ReduceLROnPlateau；bug 修复后 run_finetune_fixed.py 显式加载 `pretrained_encoder_contrastive.pt`（对比 checkpoint），dropout 0.1/mean_pool/统一 lr 5e-4；对比学习多种子稳健性 Strategy B 端到端跑 10 种子(42–51)。**（C）** 纯监督无预训练，Adam(lr 5e-4, wd 1e-5)+CosineAnnealingLR+grad_clip 1.0，max 300 epoch/patience 30/batch 64，3 种子(71/42/123)。**通用划分**：get_data_splits(seed=71, 72%train/8%val/20%test) 索引确定性切分。

### 结果
**（A）JEPA v1 — 完全失败（表示塌陷）**：预训练 loss 0.0714→**0.000000**，monomer-trimer 余弦相似度 **1.0000**（典型塌陷）；微调 Test RMSE：baseline 54.33、Strategy A 104.49、B 104.34、C 104.00——三策略全比无预训练差约 +50，"Any pre-training beats baseline: False"。**（B）JEPA v2–v6 + 对比**：JEPA v2 loss 30.92→19.55（VICReg 部分止血）余弦 0.9952 仍近塌陷；对比预训练 loss 收敛 ~1.77，日志明确"Emb Var 1.0872 Cos Sim −0.0027 Collapse False"（VICReg 成功防塌陷）。**bug 修复后**（finetune_results_fixed.json 为准）：bug-fixed baseline 54.89；**对比学习 Strategy B=50.77（Δ−4.12，最佳单次）**、Strategy A 55.84、C 64.21；**JEPA v6 Strategy B=54.79（Δ−0.10，几乎无提升）**。对比学习 10 种子(42–51) Strategy B mean **56.95±3.43**（范围 51.48–62.37）→高方差，50.77 属乐观值。结论：对比学习>JEPA（50.77 vs 54.79），但 50.19 目标始终未稳定达成。**（C）GPS-SPSE-Trimer**：3 种子 Test RMSE 53.22/55.04/54.97，mean **54.41±0.84**→FAILS（train RMSE 28.83–38.78 偏高、best_epoch 多在 20–60 轮早停→欠拟合）。SPSE 诊断证实编码了跨 RU 结构信号（跨 RU 距离 mean 14.49 vs RU 内 5.01）但未转化为回归精度。mem_095/098 方向证实。**关键纠偏**：旧 summary 把"JEPA v1(54.79)"混为一谈不准确——代码核实 54.79 实为 **JEPA v6** Strategy B、50.77 实为**对比学习**（run_finetune_fixed.py:64 加载 `pretrained_encoder_contrastive.pt` 自证）。

### 流程
**（A）** run_all.py 串联 Phase1 PI1M+Tg 转 trimer 图→Phase2 JEPA 预训练→Phase3 baseline+三策略微调→analysis 算余弦相似度生成 results_summary.txt；塌陷信号（loss=0、余弦=1.0）在 summary 阶段才暴露。**（B）** 迭代式 debug：(1) jepa_pretrain.py 逐版加 VICReg/mean_pool/dropout 0.1（v2→v6，对应 run_all_v{2..6}.log）；(2) contrastive_pretrain.py 平行跑 SimCLR 风格支；(3) 发现并修复三类 bug（dropout 0.5 vs 0.1、mean vs add 池化不匹配、Strategy B/C 学习率）→run_finetune_fixed.py 用对比 checkpoint 重跑；(4) run_finetune_tuned.py 做超参网格；(5) run_finetune_v2.py 跑 10 种子稳健性+JEPA v6 对照。**（C）** dataset.py 离线算全图最短路径距离并缓存（spse_distances.pt 808 MB）→train.py 3 种子顺序训练（每种子约 53–101 分钟）→evaluate 出 results.json/diagnostics.json。training_iter3.log+results.json 为最终迭代（以 results.json 为准）。

---

## Day 3：PCR-v2 对比正则（失败）+ PerioGT-Lite 对比预训练（多种子翻转为负）

### 实验目的
两个子实验都在 Day1 固化的"简单 GIN+trimer+Tg"基线（≈50 K）之上用**对比学习**注入聚合物周期结构先验。**（A）PCR-v2（Polymer Contrastive Regularization v2）**（`3_day3_pcr_v2`）：在监督训练中**实时**构造"同一聚合物的 monomer↔trimer"正样本对，以 InfoNCE 对比损失作辅助项联合训练，验证 in-training 对比正则能否提升 Tg 回归（经 3 轮迭代）。**（B）PerioGT-Lite**（`3_periogt_lite`）：用 PA（Periodic-Aware，周期感知）数据增强在 PI1M（30 万子集）上做对比预训练，再微调 Tg，验证"先大规模无标注聚合物预训练、再迁移小数据 Tg"是否稳定胜过从零训练。

### 模型架构
**（A）PCR-v2** 主干 EdgeAwareGINConv，3 个迭代版本：Iter1（5 层 GIN hidden 256 dropout 0.5 mean_pool，纯 mean 基线）；Iter2 混合池化版（监督头 global_add_pool 保量级、对比投影头 global_mean_pool 消图尺寸差，投影头 256→128→64→L2 normalize，InfoNCE τ=0.07）；Iter3（放弃混合池化回 Day1 精确 3 层 mean_pool+无 dropout，投影头前加 F.normalize 消解 monomer/trimer 量级差+α warmup）。**（B）PerioGT-Lite** = Day1 精确架构（3 层 EdgeAwareGINConv hidden 256 mean_pool）；预训练模型 PerioGTPretrainModel=encoder+投影头(256→128→64→L2)；**PA 周期感知增强** 6 种正对：monomer_trimer、subgraph_crop、shifted_ru、edge_drop_feat_mask（损坏率 25/40/50/60%）、monomer_dimer、hybrid_corrupt。

### 训练评估方式
共用 Tg 数据集、同划分 seed=71、72/8/20（5162/573/1435），节点特征 30 维、边 6 维，Adam lr=1e-3，MSE，patience 30。**（A）** batch 32（PCR 模式每 batch 含 64 图：monomer+trimer 双视图），联合损失 L=L_sup+α·L_InfoNCE（Iter3 α∈{0.01,0.05,0.1,0.5}），验证/测试只走 baseline 前向(trimer 单视图)保公平。**（B）** 预训练 preprocess_pi1m.py 从 PI1M 随机采样 300,000 条(seed=42)转 monomer/trimer 图（pi1m_trimer_graphs.pt 5.2 GB），InfoNCE τ=0.07，effective batch 512（梯度累积），50 epoch；微调加载预训练 encoder 权重在 Tg 上 lr=1e-3/batch 32/max 300 epoch；最后 multiseed_verify_fixedsplit.py 在 seeds={42,71,123,456,789} 固定划分复现 50% 损坏率做统计检验。

### 结果
**（A）PCR-v2 — 三轮迭代全失败**：Iter1（mean_pool）baseline 52.68，PCR α=0.1/0.5/1.0→54.65/54.03/54.05 全劣化；**Iter2（混合池化）baseline_sum_pool=77.4978（≈77.50，代码证实）**，PCR α=0.01/0.05/0.1→76.15/76.09/74.78（最好 74.78，仍远差 Day1 50.19，gap +24.59）——77.50 的根因是 sum 池化不按节点数归一、5 层 GIN+dropout 0.5 的 trimer 节点特征累加后量级爆炸、严重过拟合（overfit_gap +30.36）；Iter3（干净 mean_pool 回 Day1）baseline 51.34，PCR α=0.01/0.05/0.1/0.5→52.49/52.24/52.99/52.27 无一优于 baseline。确认 PCR-v2 失败，与 mem_109 一致。**（B）PerioGT-Lite — 单种子边际、多种子统计翻转为负**：baseline 从零训练 51.92(seed71)/5 种子均值 52.49±1.25；6 种 PA 增强单 seed(71) 中 **edge_drop_feat_mask@50%=49.83（−4.03%，号称 sweet spot）**；但 multiseed_fixedsplit_results.json 在 seeds={42,71,123,456,789} 固定划分重跑 50% 损坏率，5 种子 baseline-pretrained 差分别为 −2.02/−1.82/+0.42/+1.09/+0.18，汇总 baseline_mean 52.49±1.25、pretrained_mean **52.92±0.29**、improvement **−0.81%**（预训练平均反而劣 0.43 K）、pretrained_wins 3/5 但赢得小输得大——**49.83 的 −4.03% 提升是统计假象**，证实 mem_112"PA-based 对比预训练不能稳定提升"。附加：损坏率-RMSE 呈非单调（25%→51.53/40%→54.06/50%→49.83/60%→51.71），cos_sim 在 0.88–0.89 一带几乎重叠但性能差 4K+，说明 cos_sim 不能预测下游性能。

### 流程
**（A）PCR-v2 线**：Iter1(mean_pool PCR 全劣化)→诊断"mean_pool 把 Day1 50.19 拉到 52.68"→Iter2 改混合池化抢救量级跑出 77.50 灾难结果→Iter3 彻底回 Day1 精确架构只留 PCR 最小改动，baseline 51.34 但 PCR 4 α 全 52.27–52.99→判定失败归档。**（B）PerioGT-Lite 线**：Iter1 monomer-trimer 对比(52.41 劣化)→Iter2 引入 subgraph_crop/shifted_ru→Iter3 扩 6 种 PA 增强+edge_drop@25%(51.53 单种子首次微弱改善)→Iter4 扫损坏率命中 50%(49.83 单种子最优)→为回应"是否统计显著"跑 multiseed_verify_fixedsplit.py，**结论翻转：均值反而劣 0.81%、胜利不复现**→判定失败归档。

---

## Day 4：PCFT 双编码器（不显著）+ BlochSpec 光谱特征（无实质帮助）

### 实验目的
**（A）PCFT（Prompt-Conditioned Fine-Tuning，提示条件化微调）**（`4_04_pgft`）：针对前序"预训练表征获得了但下游未充分利用"（representation-usage gap），把 PA 预训练编码器当作**冻结的提示通道**，其输出 h_pre 作显式条件拼到下游编码器输出上，看能否桥接 gap（子假设 H4a 提示机制够不够、H4b 双通道互补是否有用）。**（B）BlochSpec（Bloch 能带光谱特征）**（v1 `4_4_blochspec`、v2 `4_4_4_blochspec`）：在**输入空间**显式编码周期性——把单体当周期性单元胞，用 chemistry-weighted periodic Laplacian（化学权重周期拉普拉斯）算 **Bloch 能带结构**，12 维光谱摘要拼到 GIN 图表示上，看是否改进 Tg。v2 是 v1 发现池化 bug 后修复迭代。

### 模型架构
**（A）PCFT** 共用 GINEncoder（3 层 EdgeAwareGINConv hidden 256，30 维节点/6 维边，mean_pool）；**PCFTModel 双编码器**——Encoder P（冻结，从 PI1M PA 预训练得来，处理 monomer 图→h_pre 256 维，requires_grad=False）+ Encoder T（可训练，处理 trimer 图→h_task 256 维，从同 PA 权重初始化 dropout 0.5 微调）；预测头 MLP([h_task‖h_pre])=Linear(512→256)→ReLU→Dropout0.3→Linear(256→128)→ReLU→Linear(128→1)，h_pre 通过预计算 lookup 表按聚合物索引取出。对照 BaselineGINModel=单 GIN+Linear。**（B）BlochSpec**：compute_spectral.py 找单元胞边界原子→单元胞 Laplacian L_UC→化学权重键权=键型权重×(1+|Δχ|)→周期耦合矩阵 L_coupling→在 24 个 k 点上 H(k)=L_UC+e^{ik}L_coupling+e^{−ik}L_coupling^T 强制 Hermitian 后 eigvalsh→12 维光谱摘要（spectral_gap/radius/centroid/variance/low_freq_density/band_width/flatness/density_at_Fermi/skew/kurtosis 等）；模型 v1 用 global_add_pool、v2 文件头标"CRITICAL FIX: All models use global_mean_pool"修复，BlochSpecGIN=同款 GIN→池化→与 12 维光谱拼接→MLP。

### 训练评估方式
**（A）PCFT**：Tg 7170 每条同时构造 monomer+trimer 图，split 72/8/20 seed=71；Phase1 加载 PA checkpoint→Phase2 对全部 Tg 单体前向得 h_pre lookup 表→Phase3-4 跑 5 消融×3 种子[42,71,123]→Phase5 诊断。训练 Adam lr=1e-3 wd=1e-5 batch 32 max 500 epoch patience 30 余弦退火 RMSE 损失。五消融：(a) GIN+Trimer 基线随机初始化、(b) PCFT 但 EncP 随机初始化、**(c) PCFT+PA 预训练 EncP（主）**、(d) 标准微调（无 prompt 单编码器，负对照）、(e) PCFT 双编码器全冻。**（B）BlochSpec**：光谱预计算 7169/7170(99.99%) 成功；训练同 Day1 GIN 超参（lr=1e-3 余弦退火 batch 32 patience 30）seed=71 3 种子；v1 跑 5 实验×3 种子，v2 跑 7 实验×3 种子（追加 WiderMLP 参数对照、ReducedGNN 维度对照）。配对 t 检验。

### 结果
**（A）PCFT — 假设未支持**（3 种子）：(a) GIN+Trimer 基线 **52.46±0.20**、(b) PCFT+随机 EncP 53.34±0.36、**(c) PCFT+PA 预训练 EncP（主）52.72±0.76**、(d) 标准微调 53.53±0.37、(e) 全冻 54.02±1.43。配对 t：c_vs_a（PCFT vs 基线）**t=0.5511 p=0.6369 不显著**，其余亦 p>0.05。结论：PCFT(52.72) 相对基线(52.46) 反差 0.26，相对标准微调(53.53) 仅好 0.81，PA 预训练相对随机 prompt(b) 有 0.62 改进但双编码器结构本身(b vs a)无益。诊断（代码独有）：h_pre 两两余弦相似度均值 0.9633、72.8% 的对>0.95（h_pre 表征高度退化、几乎不分子）；h_task 与 h_pre 余弦稳定 0.56–0.58；预测头 ‖W_task‖=20.57 vs ‖W_pre‖=13.39（任务头权重更大，prompt 通路贡献有限）。**（B）BlochSpec — v1 池化 bug、v2 修复后无实质改进**：v1（add_pool）baseline 66.15±0.27、BlochSpec 61.52±5.99、随机特征 58.68、unweighted 58.45——看似 BlochSpec 低 4.6 但随机特征同样甚至更好→改进来自额外输入维度而非 Bloch 物理，且 baseline 66.15 远高于 Day1 ~50.19 定位为**池化不一致 bug**（add vs Day1 mean）；v2（修 mean_pool）baseline 51.71±0.33、BlochSpec 53.25±0.51（更差 +1.54，p=0.1133 不显著）、随机特征 52.11、unweighted 51.89——**Bloch 光谱特征对 GIN+Trimer 基线无实质帮助**。任务中 66.15/61.52/51.71/53.25/p=0.64 全部代码证实。mem_126/128 证实。

### 流程
**（A）PCFT**：加载 Day3 PA checkpoint→载入 Tg 生成 monomer+trimer 图缓存→compute_hpre_lookup 用冻结 EncP 对全部 Tg 单体前向得 lookup 表(7170×256)→依次跑消融 (a)→(e) 各 3 种子→Phase5 诊断 h_pre 质量/h_task-h_pre 分歧/预测头权重范数→汇总 summary_report.txt（5 消融表+5 配对 t+诊断+自动结论），H4a 未支持。**（B）BlochSpec**：v1 compute_spectral 预计算 weighted/unweighted/random 三套光谱→dataset 构造 PyG InMemoryDataset→train 跑 5 实验×3 种子+12 单特征重要性→diagnostics 做 Pearson 相关→期间发现 baseline 66.15 远高于 Day1 ~50.19 定位为 add_pool 池化不一致 bug；v2 拷贝 v1 代码统一改 mean_pool（文件头标 CRITICAL FIX）+ 新增参数/维度对照，跑 7 实验×3 种子，结论同 v1：改进来自维度而非物理。

---

## Day 5：Chemprop 有向键 MPNN 突破（43.46，机制主导 14:1）+ Bipartite v2 失败

### 实验目的
**（A）Chemprop-Polymer**（`5_6_chemprop_polymer`）：验证"有向键级消息传递（directed bond-level MPNN，Chemprop 风格）"与"周期边"是否架构-拓扑兼容——相比 Day1–4 的 GIN/原子级聚合（GIN+trimer 50.19、GIN+polymer 53.28），把消息传递从原子级下沉到**有向键级**、把周期边当普通有向键参与消息传递，能否打破 baseline；通过 6 项消融 A–F 厘清"架构机制"与"拓扑细节"对增益的相对贡献。**（B）PolymerBipartite v2**（`5_day5_polymer_bipartite_v2`）：验证 XIMP 风格的二分（bipartite）原子↔重复单元消息传递（标准 GIN 原子层之上叠加 atom→RU 聚合 + RU→atom 门控广播）能否优于参数匹配的 GIN+trimer baseline。mem_141 已预告其失败。

### 模型架构
**（A）Chemprop-PolymerMPNN**：每条无向键拆 2 条**有向键**各自维护独立消息向量，更新用**反向键上一轮消息** m_ji^{t-1}（Chemprop 标志设计）；初始化 m_ij^0=ReLU(W_init·concat(atom_i,atom_j,bond_ij))，第 t 步 m_ij^t=ReLU(W_h·concat(h_i^t,m_ji^{t-1},bond_ij))，h_i=ReLU(W_a·concat(atom_i,Σincoming))，输出 z_i=ReLU(W_o·concat(atom_i,a_i^final))，graph pooling atom 级 mean→3 层 MLP。reverse_map 预计算每条有向键的反向键索引；**depth=3**（W_h/W_a 共享）；ATOM_FDIM=123（原子序 100+度 6+电荷 5+杂化 5+芳香 1+环 1+Hs 5）、BOND_FDIM=9（键型 4+共轭 1+环 1+is_periodic 1+电负性梯度 1+边界原子度 1）。图构造：monomer（单 RU 无周期边）/trimer（3 RU 无周期边）/full polymer（trimer+一条周期边从 copy-2 右端连回 copy-0 左端闭合为环，置 is_periodic=1 携带电负性梯度与边界度）；变体 minimal_periodic（周期边特征置 0 仅留标志）、random_periodic（周期边端点替换为随机非端点原子）。**（B）PolymerBipartiteV2**：30 维原子→256 hidden，RU 初始化由 atom 嵌入逐 RU mean-pool 后线性投影；每层(5 层)三阶段——atom→atom(EdgeAwareGINConv)、atom→RU 聚合(scatter mean+线性)、inter-RU 耦合(RU0↔RU1↔RU2 线性消息+LayerNorm)、RU→atom 门控广播（g=sigmoid 标量，按 is_boundary 取 boundary/interior 两可学习标量）；预测头 global_mean_pool(h_atom)+global_mean_pool(h_ru)→MLP；参数 1,769,736（baseline GIN+Trimer 714,502）。

### 训练评估方式
**（A）** Tg 7170，图构造 seed 固定 42，**split 用 sklearn train_test_split(random_state=seed)**——每训练 seed 得不同 72/8/20 划分；Adam lr=1e-4 batch 50 epochs 100 patience 30 MSE+grad_clip 1.0**目标不归一化**；每消融 3 种子[42,71,123] mean±std（np.std 总体方差）。**（B）** split 按索引切片 72/8/20（seed 参数未实际使用，三种子共用同划分）；Adam+CosineAnnealingLR，backbone_lr 1e-3/ru_lr 5e-4（分组学习率）wd 1e-4 batch 32 epochs 300 patience 30；损失 MSE+判别损失 disc_lambda 0.1+gate 惩罚 0.01·mean(g)；3 种子[42,71,123]。

### 结果
**（A）Chemprop 6 消融**（3 种子，逐数字与 diagnostic_report.txt 核验一致）：A_monomer **45.76±0.28**、B_trimer 43.92±0.45、**C_polymer_full（trimer+正确周期边+完整特征）43.46±0.89（最佳）**、D_polymer_minimal（仅 is_periodic 标志）43.71±0.69、E_polymer_random（端点随机）43.58±1.20、F_polymer_256（hidden 256）44.48±1.23。**核心对比（机制 vs 拓扑）**：架构机制增益=GIN+trimer(50.19)−Chemprop+polymer(43.46)=**6.73 RMSE**（Chemprop+polymer 还比 GIN+polymer 53.28 低 9.82）；拓扑贡献=Chemprop-trimer(43.92)−Chemprop-polymer(43.46)=**0.46**；机制:拓扑≈**14.6:1**。**随机周期边(43.58)≈正确周期边(43.46)，差 0.12**；minimal(43.71)≈full(43.46)差 0.25→周期边的具体化学/富特征非关键，"额外连接本身"即提供全部拓扑增益。周期边消息可分性 mean cos_sim=0.7536(<0.90 具判别性)→Chemprop 确在利用其化学。hidden 300(43.46)>256(44.48)差 1.01；过拟合 gap 均为负 −2.3~−2.7 无严重过拟合。**（B）PolymerBipartite v2 iter2**：**53.81±0.94** vs GIN+Trimer(h256) 54.00±1.96；**参数匹配宽 baseline（hidden≈408, 1,758,590 参数）53.66 反而略胜 bipartite 53.81**→参数匹配后 bipartite 不及 baseline，FAILED（印证 mem_141）。诊断：RU 可分性 cos_sim 0.4623（判别弱）、随机 RU 对照 Real 54.85 vs Random 54.88 Δ=0.03（结构对应无贡献）、gate 值 boundary mean 0.2846/interior 0.0617 已饱和（iter1 用 MLP gate 失效 mean 0.9972 故 iter2 改标量）。所有数字代码证实，无需"AI 自述未验证"标注。

### 流程
**（A）** construct_polymer_graphs.py 解析 p-SMILES→三种图构造(monomer/trimer/trimer+periodic_edge)→atom/bond 特征化+reverse_map+incoming_bonds 预计算→落盘 5 个 .pkl（各 ~160–470 MB）；train_ablations.py 依序跑 A→F 六消融每项 3 种子，早停后回载 best_val 评估输出 ablation_results.csv+ablation_summary.csv；diagnostics.py 出周期边消息 cos_sim/架构对比/随机对照/minimal-full 对照/hidden 对照→结论 Chemprop 有向键 MPNN+正确周期边 43.46±0.89 较 GIN+trimer 降 6.73、架构机制主导 14:1，为 Day6 HPO 提供基础。**（B）** iter1(MLP gate)发现 gate 饱和到 ~1.0、参数匹配失败→iter2 改可学习标量 gate/LayerNorm/RU 投影初始化/分组学习率/gate 惩罚；run_all.py 顺序训练+诊断（RU 可分性/随机 RU/参数匹配宽 baseline/过拟合 gap/gate 分析）→结论即便修好 gate、参数匹配宽 baseline(53.66)仍优于 bipartite(53.81)，XIMP 二分 atom↔RU 未带来正增益、标记失败不再推进。

---

## Day 6：Chemprop HPO（dense_d5_h1200 单种子 35.48 / 多种子 40.00±2.09）+ 对比预训练支线（OOM 未完成）

### 实验目的
Day6 是本 run terminal/头部模型天（无独立 optimize 目录）。**(a) chemprop_hpo（主线）**：在 Day5 Chemprop 有向键 MPNN 之上做系统性超参优化（HPO）并用 DenseNet 风格密集跳跃连接（dense skip connections）测试容量扩展是否进一步降 Tg 误差。四阶段：Phase1 OFAT（One-Factor-At-a-Time 单因素扫描）→Phase2 组合最佳→Phase3 密集跳跃连接→Phase4 多种子验证。**(b) chemprop_pa（支线）**：构建 Chemprop-Polymer-K（K-node 即虚拟全局原子节点），验证"对比预训练+K-node"能否超过 (a) 纯监督基线，设 4 变体 A（无预训练+K-node）/B（PI1M 预训练+K-node）/C（随机图预训练+K-node）/D（PI1M 预训练无 K-node）。**(c) fix_phase4**：修补 (a) 中 Phase3 与 Phase4 batch size 不一致的协议缺陷。

### 模型架构
**（a）ChempropPolymerMPNN**（同 Day5 架构，ATOM_FDIM=123、BOND_FDIM=9，mean pool，无归一化层、目标不归一化直接 K 回归，W_h/W_a/W_o 单层共享故 depth 不增参数）；密集跳跃连接：MP 步 i>0 时把第 0..i-1 步消息向量拼接(i×hidden)经 dense_projections[step-1]: Linear(i*hidden→hidden) 投影回再聚合，depth=5 时 4 个投影层、depth=7 时 6 个。**参数量（核实）**：combined_best(d7,h1200,无 dense)=9,260,401；**dense_d5_h1200=23,665,201（≈23.7M）**（dense 增量 ≈14.4M）；dense_d7_h1200=39,507,601。**（b）ChempropPolymerKMPNN**：7 层有向键 MPNN hidden 1200，K-node 连所有真实原子携带 528 维聚合物全局特征（K_FEAT_DIM=528），预训练头投影器 1200→128→64，参数 15,213,601。**（c）** 复用 (a) 仅改 batch size 与种子协议。

### 训练评估方式
**（a）** Tg 6960 图（原始 7170，解析失败 210），72/8/20→5011/557/1392；**split 种子=init 种子未解耦**（train_model 用同一 seed 同时设 torch.manual_seed 与 split_dataset）；Adam（仅 lr 无 wd 无 scheduler）grad_clip 1.0 max 200 epoch patience 30 MSE，**RMSE 用 numpy 手算**；batch：Phase1/2 默认 50（h=1200 降到 25），Phase3 在 continue_phases.py 改 bs=16（h=1200），**Phase4 多种子沿用默认 bs=50**。**（b）** 6960 图 72/8/20→5011/556/1393，**split_seed=71 固定、FFN 种子遍历{71,42,123}解耦**（比 (a) 协议更正确），batch 64 max 300 epoch patience 30；对比预训练 InfoNCE τ=0.07，trimer-shift+键丢弃 15%+特征掩码 25% 增强，K-node 特征永不增强，batch 32（每批 64 图双视图）50 epoch lr=0.001。**（c）** 同 (a) 仅强制 bs=16 并计划补一组无 dense 多种子。

### 结果
**（a）chemprop_hpo 四阶段（全部数字来自 results_phase1/2/3/4.csv 与 summary.txt）**：Phase1 OFAT（13 配置 seed=71）按组最佳 A_depth7=40.06、B_hidden1200=38.13、**C_lr1e-3=36.57（Phase1 全局最佳，与 mem_155 一致）**、D_dropout0.1=41.51→选 depth=7/hidden=1200/lr=0.001/dropout=0.1。Phase2 combined_best(d7,h1200,lr1e-3,do0.1,无 dense,bs=25) **35.7190**（train 21.44 gap −14.28 best_epoch 130）。Phase3 密集跳跃连接(seed=71 bs=16 continue_phases.py 重跑)：dense_combined_d7_h1200 35.7840、dense_d7_h1200 35.7317、**dense_d5_h1200=35.4758（本 run 全局单种子最佳，train 20.94 gap −14.53 best_epoch 179）**。Phase4 多种子(dense_d5_h1200 seeds{42,71,123} bs=50)：40.95/37.11/41.95→**40.00±2.09**（train 28.32±5.26 gap −11.68±3.99）。**vs PerioGT SOTA(Tg 30.8)**：单种子最佳 35.48 落后 ~4.7、多种子 40.00 落后 ~9.2。**（b）chemprop_pa**：variant_A（无预训练+K-node 3 种子×3 lr）lr=1e-4 **40.23±0.28（最佳）**、lr=5e-4 41.06±1.23、lr=1e-3 44.86±2.71，均显著劣于 (a) 的 35.48；variant_B（PI1M+K-node）对比预训练跑到 epoch 16/50 日志中断、中间 4 次保存 best checkpoint、pretrain_pi1m_knode.pt **161 MB 存在**→预训练权重确有保存但预训练未跑完、微调阶段未成功启动；**variant_C/D 完全未启动**（experiment_log 在 variant_B 预训练中段截断）；CUDA OOM 实锤（variant_A_lr5e-4_final.log 含 torch.OutOfMemoryError，GPU 争用所致，另一进程占 9.68 GiB）。**（c）fix_phase4 — 未完成**：仅 seed42 跑完(bs=16) Test 41.7050、seed71 到 epoch 80 截断、seed123 未启动，results_phase4_v2.csv/results_phase4_nodense.csv 均未生成。

### 流程
**（a）主线两次运行**：首次 chemprop_hpo.py main() 跑 Phase1(13 配置)→Phase2(35.7190)→**Phase3 首配置 dense_combined_d7_h1200 崩溃**：run_log.txt `RuntimeError: mat1 and mat2 shapes cannot be multiplied (5544x2400 and 1200x1200)`，发生在 dense_projections 维度与 forward 实际拼接维度不符；修复改写 chemprop_model.py（Linear(i*hidden,hidden)）+ 写 continue_phases.py 从 Phase3 续跑(改 bs=16)；续跑 Phase3 三 dense 配置全成功(35.7840/35.7317/35.4758)→选全局最佳 dense_d5_h1200→Phase4 三种子(**bs=50**)→40.00±2.09→DONE。⚠️**协议不一致（已二次证实）**：Phase3 最佳 dense_d5_h1200 用 **bs=16, seed=71** 得 35.4758；Phase4 用同 seed=71 同配置但 **bs=50** 得 37.1104，同 seed 同配置仅 batch size 不同→1.63 RMSE 差异，是 bs 影响的直接证据；且 Phase4 用单 seed 同时控制 split 与 init（未解耦），多种子混入两类随机性。fix_phase4 试图用 bs=16 修补但未完成。⚠️**dense 增益诚实修正**：dense_d7_h1200(35.7317) 反比无 dense 的 combined_best d7h1200(35.7190) **差约 0.0127**，即 dense 在 d7 同协议对比下增益很小甚至为负；dense_d5_h1200(35.4758) 虽全局最佳但缺同协议无 dense 的 d5h1200 对照，不能证明 dense 本身贡献。**（b）支线**：先跑完 variant_A 三 lr→启动 variant_B PI1M 对比预训练(298057 图 bs=32 50 epoch)→epoch 16/50 日志中断，后续 finetune OOM、variant_C/D 未启动；预训练权重虽存盘但"预训练→微调"链路未走通。**（c）fix_phase4**：计划跑 dense_d5_h1200 三种子(bs=16)+无 dense d7h1200 三种子(bs=25)，实际仅 seed42 完成(41.7050)，协议修正未达成。

---

## 6 天轨迹总结

ab5 是 DeepSeek 基座的跨基座鲁棒性 run，Tg RMSE 演进：**Day1 GIN+trimer 50.19（GIN 天花板）→ Day2 对比学习 50.77（乐观单次，10 种子均值 56.95）→ Day5 Chemprop 43.46（突破）→ Day6 HPO dense_d5_h1200 单种子 35.48 / 多种子 40.00±2.09**。核心节点：

- **数据中心（Day1）**：5 种结构变换确立 trimer 为 Tg 最小充分表示（50.19），cyclized 过拟合最差（53.28）；4 架构变体（GIN-GAT/VirtualNode）全差于 GIN+trimer→GIN 架构成瓶颈。
- **自监督预训练/对比学习（Day2–4）全失败**：Day2 JEPA v1 表示塌陷（loss→0、余弦 1.0），对比学习 bug 修复后 50.77 但 10 种子高方差不稳定，GPS-SPSE 54.41 欠拟合；Day3 PCR-v2 三轮迭代全失败（混合池化 77.50 灾难、干净 mean_pool baseline 51.34 但 PCR 各 α 全劣），PerioGT-Lite 单种子 49.83 经 5 种子固定划分复核翻转为 −0.81%（统计假象）；Day4 PCFT 双编码器 52.72 vs baseline 52.46（p=0.637 不显著），BlochSpec v2 53.25 vs baseline 51.71（更差、随机特征同样有效）。
- **Chemprop 突破（Day5）**：有向键级消息传递 + 周期边 43.46±0.89（较 GIN+trimer 降 6.73），机制:拓扑≈14.6:1，随机周期边≈正确周期边（差 0.12）→架构机制主导，为 Day6 HPO 提供基础。
- **HPO（Day6，头部）**：4 阶段 Phase1 OFAT lr1e-3=36.57→Phase2 35.72→Phase3 dense_d5_h1200 单种子 35.48（全局最佳）→Phase4 多种子 40.00±2.09；含一次 dense 维度 bug 崩溃+continue_phases.py 修复续跑；与 PerioGT SOTA 30.8 差 +4.7（单）/ +9.2（多）。

**与旧 summary 的关键修正**：(1) **KA-GPS（Day1）从未执行**——无 results/checkpoint/full.log 痕迹/mem 声明，旧 summary"KA-GPS 50.8/51.2"无任何支撑（弃用）；(2) Day2 **50.77 实为对比学习、54.79 实为 JEPA v6**（旧 summary 把 JEPA/对比混谈，run_finetune_fixed.py:64 加载 contrastive checkpoint 自证）；(3) Day3 **PerioGT-Lite 49.83 是统计假象**（5 种子固定划分均值反而劣 0.81%）；(4) Day6 **bs=16 vs bs=50 协议不一致**（Phase3 头条 35.48 用 bs=16、Phase4 多种子 40.00±2.09 用 bs=50，非同协议复现）；(5) Day6 **dense 增益很小/不确定**（dense_d7 反比无 dense 略差 0.0127，dense_d5_h1200 缺同协议对照）；(6) Day6 **chemprop_pa variant_B 预训练权重保存（161 MB）但预训练→微调链路未完成**、variant_C/D 完全未启动（OOM），旧 summary 已正确修正但需强调。

**任务范围限定**：Day1–6 实验均只预测 Tg；Tm/Egc 仅来自辅助 GAT baseline（单种子 seed=42：Tg 50.52、Tm 79.30、Egc 0.6326），EA/IP 完全未涉及。基座=DeepSeek、模式=full day-night 来自任务描述，日志未直接自标识模型名（无法从归档 100% 独立确认）。

## 主要参考文件

- 代码：`src/1_01_data_centric_foundation/{gin_model.py,feature_utils.py,transform_utils.py,train.py,run_all.py,results_summary.txt,experiment.log}`、`src/1_day1_arch_improve/{models.py,train.py,diagnostic_analysis.py,results.csv,summary_report.txt}`、`src/1_2025_07_11_ka_gps/{model.py,train.py,dataset.py}`(仅代码无结果)、`src/2_polymer_jepa_pretrain/`+`src/2_2_polymer_jepa_pretrain/{jepa_pretrain.py,contrastive_pretrain.py,run_finetune_fixed.py,finetune_results_fixed.json,results_summary_final.txt}`、`src/2_gps_spse_trimer/{model.py,train.py,results.json,diagnostics.json}`、`src/3_day3_pcr_v2/{model.py,model_v3.py,results.json,results_v2.json,results_v3.json}`、`src/3_periogt_lite/{model.py,pretrain.py,finetune.py,augment.py,FINAL_REPORT.md,multiseed_fixedsplit_results.json,MULTISEED_VERIFICATION_REPORT.md}`、`src/4_04_pgft/{pcft_experiment.py,summary_report.txt,results_ablation_a..e.json,diagnostic_*.json}`、`src/4_4_blochspec/`+`src/4_4_4_blochspec/{compute_spectral.py,model.py,train.py,results/results_summary.txt,FINAL_REPORT.md}`、`src/5_6_chemprop_polymer/{chemprop_model.py,construct_polymer_graphs.py,train_ablations.py,ablation_summary.csv,ablation_results.csv,diagnostic_report.txt}`、`src/5_day5_polymer_bipartite_v2/{model.py,train.py,run_all.py,results_summary_iter2.txt}`、`src/6_06_chemprop_hpo/{chemprop_hpo.py,chemprop_model.py,continue_phases.py,fix_phase4.py,construct_polymer_graphs.py,summary.txt,results_phase1/2/3/4.csv,run_log.txt,run_phase3_4_log.txt,fix_phase4_log.txt}`、`src/6_06_chemprop_pa/{model.py,finetune.py,pretrain.py,data_utils.py,README.md,results/experiment_log.txt,results/pretrain_pi1m_knode.pt}`。
- 记忆：`mem.json`（157 节点；mem_001 TopicState 自述、各 day experience 节点 mem_084/085、095/098、109/112、126/128、141/144、155）、`lite.log`、`full.log`。
- baseline：`src/baseline/results/{Egc,Tg,Tm}_baseline_report.txt`。
