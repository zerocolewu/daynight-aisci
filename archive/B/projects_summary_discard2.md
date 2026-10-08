# CPA-GAT - ab6 (GLM-5, full day-night)

> 一句话定位：ab6 / 基座 GLM-5 / full day-night / 最终模型：CPA-GAT（基数保持注意力 GATv2 + is_backbone 最长路径节点特征 + SWA，唯一改动 lr 1e-3→1e-4），头部目录 `src/optimize`。

## 实验目的

该 ab6 run（autonomous，基座 GLM-5，full day-night 模式）的目标是在 7 天内逐步把聚合物性能预测模型推向 PerioGT SOTA（Tg RMSE 30.8 K，GAT baseline 50.5 K）。前期（day1-5）系统性穷举并证伪了"数据/编码/预训练/全局注意力"等改进轴：数据中心特征工程（rich features）主导、3-RU oligomer 是最稳表示，但非预训练 GAT 渐近于约 35 K；SPSE/MaxGAT、LSPE、GraphMAE、SelfGNN（BYOL）、MoMuCon（NT-Xent）、GPS-Lite 全部负向，三次 PI1M 自监督均以"越复杂越差"的方式失败。day5 读了 CardinalGraphFormer 后确立新假设：默认 GNN 是 cardinality-blind（基数盲），mean 聚合会丢弃链长/支化信号。于是 day6 提出 CPA（Cardinality-Preserving Attention）——在标准 GATv2 注意力之外加一条 sigmoid 门控的 unnormalized-sum 通道，显式保留邻居基数，并在所有层部署；这打破了前期"指数衰减"的改进轨迹（34.77 K，+1.61 K）。day7 进一步验证了 axis hierarchy：CPA 在聚合层 subsume 掉 PSPE 链几何特征与 bond-graph 基数（CPA+PSPE、PolyBG-CPA 均 null）。最终头部模型 `src/optimize` 即把项目内经验证最优的单配置 ORIG_BB（simple original CPA-GAT + is_backbone 节点特征 + SWA，唯一改动 lr 1e-3→1e-4）以解耦 split/init 的多种子协议，在全部 5 个 benchmark（Tg/Tm/Egc/EA/IP）上做一次干净、协议对齐的复跑，给出 ensemble 与 single-seed 的 mean±std。设计动机的本质是"周期性/链状先验应通过聚合函数的基数保持来注入，而非靠特征工程或全局注意力"。

## 模型架构

最终头部模型 = `src/optimize` 的 CPAGATv2，但部署的是 ORIG_BB 配置（simple 原始 CPA-GAT：use_edge=False、use_residual=False、use_norm=True、pool="mean"、predictor_type="simple"、use_input_bn=False）。输入到输出：

**输入**（PyG Data batch）：节点特征 x [N,43]、edge_index [2,E]、batch [N]、可选描述子 u [G,6]（共聚物）、目标 y。图来自 p-SMILES 的 head-to-tail 3-mer oligomer（N_MER=3）；共聚物（EA/IP）把两个单体拼成一个连通 A-B 分子（A 出口与 B 入口显式成键）。节点特征 43 维 = rich chemistry（8 维：atomic_num/100、hybridization/6、aromatic、degree/10、charge/5 clipped[-1,1]、in_ring、chirality/10、H-count/10）+ PSPE（2 维：dce=到最近链端的归一化距离、mi=沿骨架的单体索引/2，经 double-BFS 最长路径求得）+ RWPE（32 维，AddRandomWalkPE walk_length=32）+ is_backbone（1 维，double-BFS 最长路径节点标记，被认为是全项目最有效的单一特征）。边特征 4 维 [single, multi(aromatic/double), in_ring, conjugated] 由 data_utils 计算并存入 edge_attr，但 ORIG_BB 的 use_edge=False 使 edge_dim=None，头部模型并不消费边特征（仅未启用的 v2 变体才用）。共聚物额外 u=[rA, rB, type-one-hot(4)]=6 维，拼到 readout 之后。

**骨干**：input_proj = Linear(43→256)；然后 4 层 CPAGATv2Conv（hidden=256，head_dim=64，heads=4，dropout=0，edge_dim=None）。每层每个 head：(1) o_std = GATv2Conv(256→64, heads=1, concat, add_self_loops=True, fill_value='mean')；(2) CPA 通道 v=gat.lin_r(x)，s=scatter_sum(v over 邻居+self)，g=sigmoid(W_g@h+b_g)，输出 o_std+g*s。4 个 head concat → 每层 256 维。层后接 ELU → LayerNorm(256) → 不加 residual → Dropout(0.0)。

**Pooling / readout**：global_mean_pool → [G,256]；随后**无条件**经过一层 readout LayerNorm(256)（`forward()` 对每个 branch 都执行 `self.branch_norm[bi](t)` 再拼入 z，ORIG_BB 单 mean branch 同样适用；该 readout LayerNorm 没有开关可关，`use_norm` 只控制每层卷积后的 LayerNorm，不控制 readout 这层）→ 共聚物再把 u(6) concat 上 → 262 维。（代码里的 sum/att/多分支 readout 是 v2 变体路径，ORIG_BB 不走。）

**预测头**（simple）：单聚物 Linear(256→128) → ReLU → Dropout(0.1, 硬编码) → Linear(128→1)；共聚物第一层为 Linear(262→128)（其余相同），输出 squeeze。

**归一化**：目标按 split 的 train mean/std 归一化，模型在归一化空间预测、指标还原到原始单位；无 input BatchNorm，每层卷积后有 LayerNorm，readout 后还有一层 LayerNorm。

**损失**：MSELoss（在归一化目标上），梯度裁剪 1.0。

**参数量**：单聚物 838,401；共聚物 839,169；两者差 **768**（不是 256），来自预测头第一层 Linear 输入维由 256 增到 262（ro_dim = hidden_dim×1 + u_dim = 256+6），多出 6×128 = 768 个权重；input_proj、各 GAT 层、Linear(128→1) 在单/共聚物间完全一致。共聚物的 6 维描述子 u 是在 readout 后 concat 进去的，并没有单独的"u→head"投影。

## 训练评估方式

**数据集**（ablation6archive corpus）：Tg.csv 7170、Tm.csv 3651、Egc.csv 3380、ea.csv 与 ip.csv 各 42966 条。划分后 train/val/test：Tg 5162/574/1434、Tm 2628/292/731、Egc 2433/271/676、EA/IP 30935/3437/8594。

**划分协议（解耦 split/init）**：Tg、Tm 为多 split——3 个 split_seed {2,42,71}（config.SPLIT_SEEDS，72/8/20，sklearn train_test_split random_state=split_seed，先切 20% test 再在 trainval 内切 10% val），每个 split 配 3 个 init_seed {42,71,2026} = 9 runs/数据集。Egc/EA/IP 用官方给定 split（Egc_split.pkl / ea_split.pkl / ip_split.pkl，单 split）× 3 init_seed = 3 runs/数据集。合计 27 runs。init_seed 控制 random/np/torch 初始化与训练随机（set_seed），与 split_seed 完全解耦。

**预处理**：一次性 featurize 全数据集并缓存（cache/{ds}_v1_m3_rw32_d43_full.pkl），多 split 复用；RDKit 建 3-mer、算 PSPE/RWPE/is_backbone；失败分子在 split 前丢弃并记录 valid/total。

**优化器**：AdamW，lr=1e-4（唯一相对原 ORIG_BB 的改动，原 1e-3），weight_decay=1e-4。调度：warmup=5 步线性升温 → cosine 衰减到 eta_min_ratio=0.001（LambdaLR）。batch_size=64（train drop_last + shuffle；val/test=256）。epochs=300，early stopping patience=50（监控 val RMSE）。

**协议选择**：取 val-RMSE 最低的 checkpoint 在 test 上评估一次作为正式结果；SWA 模型（从 50% epochs 起对 state_dict 做常数均值，swa_start_frac=0.5）仅作 reference 另报，不参与选模。

**损失/指标**：MSELoss（归一化空间）；指标 RMSE（主）、MAE、R²，全部还原到原始单位（Tg/Tm 为 K，Egc/EA/IP 为各自原始单位）。

**Ensemble**：同一 split 内（test 集跨 init seed 相同）把 3 个 init seed 的 test 预测做 mean/median；Tg/Tm 再把 3 个 per-split ensemble 的 RMSE 取 mean±std；Egc/EA/IP 单 split 直接报该 ensemble。single-seed headline：Tg/Tm = 各 split 内 init-mean 再跨 split 取 mean±std；Egc/EA/IP = 跨 init seed 的 mean±std。无 best-seed/best-split 选择。运行由 run_all.sh 在两块 GPU 上并发（GPU0: tg,ea；GPU1: tm,egc,ip），run.py 支持断点续跑（结果 JSON 存在即跳过）。

## 结果

最终头部模型（ORIG_BB, lr=1e-4）在 `src/optimize/results/RESULTS.md` 与 `results_table.json` 的全部数字（RMSE，原始单位，mean±std；已与 results_table.json、RESULTS.md、各 per-run *_sp*_init*.json、ensemble_*.json 逐项核对一致）：

**Tg**（3 splits × 3 init = 9 runs；headline = 跨 split 的 per-split init-mean 的 mean±std）：single 34.6174 ± 0.2350 K；per-split init-ensemble ens_mean 33.6477 ± 0.2275；ens_median 33.8304 ± 0.2176。per-split：sp2 single 34.3135(init_std 0.383)/ens_mean 33.3601/swa 33.8714；sp42 single 34.8858(0.652)/33.9163/34.7927；sp71 single 34.6531(0.185)/33.6667/34.5882。最佳单 run = `tg_sp2_init71` 33.9734（9 个 Tg run 中最低）。per-split ens_mean R²≈0.9072–0.9104，MAE≈22.25–23.11。

**Tm**（9 runs）：single 55.2098 ± 1.8820 K；ens_mean 54.0924 ± 1.9605；ens_median 54.2408 ± 1.9070。per-split：sp2 single 57.2209/ens_mean 56.1398；sp42 single 52.6944/51.4496；sp71 single 55.7141/54.6877（split 间方差大，主要来自 sp2 偏高）。

**Egc**（official × 3 init）：single 0.5686 ± 0.0038；ens_mean 0.5557；ens_median 0.5583（swa_mean 0.5633）。

**EA**（official × 3 init，共聚物）：single 0.0419 ± 0.0001；ens_mean 0.0407；ens_median 0.0409（swa 0.0423）。

**IP**（official × 3 init，共聚物）：single 0.0312 ± 0.0002；ens_mean 0.0302；ens_median 0.0303（swa 0.0313）。

Ensemble 相对 single 一致带来约 0.9–1.1 K（Tg/Tm）与约 3–4%（Egc/EA/IP）的 RMSE 下降；SWA per-run reference 略劣于 ens 但通常略优于 single。

**与 PerioGT SOTA（Tg 30.8 K）对比**：头部 Tg 单模型 single 34.62（差 3.82 K）、ensemble_mean 33.65（差 2.85 K）、最佳单 run 33.97（差 3.17 K），均未达到 SOTA；但相比探索期 day6 的 CPA-GAT 34.77 K（单 split 71, 3 seeds）与 day3 PSPE 35.44 K，final optimize 的 single headline 34.62 / ensemble 33.65 是本次 run 报告的最强、最稳健 Tg 结果，并把"非预训练 GAT 渐近约 35 K"的下界进一步压低。其余 4 个性质无本 run 内的 SOTA 基准可比，仅作多性质泛化记录。全部 27 runs 完成（gpu0/gpu1 日志各 12/15 条 DONE，无 killed）。

## 流程

**探索期**（`src/1_day1_* … 7_day7_*`，多为单 split=71）：day1 (a) multimer×feature 消融确立 rich features 主导（basic 69→rich 43 K，约 28 K gap）与 3-RU 最稳（rich 41.6 K），35/36 runs 完成（1 run 因 process timeout 丢失）；(b) GRIN+ELENE 复跑确认 max/L1/ELENE 全负（GAT+mean 44.57 K 最优）。day2 (a) SPSE-MaxGAT 因聚合物树退化 + max 方差崩塌惨败（38–39 K）；(b) static RWPE k=32 = 36.29 K 为正（优于 baseline 38.51 K），但可学习 LSPE-GAT = 37.96 K、LSPE-GAT-edge = 37.85 K 均劣于 static RWPE——即 LSPE 本身负向，正收益属 static RWPE（与实验目的将 LSPE 列入'全部负向'一致；原写'LSPE-GAT 部分成功'系误把 static-RWPE 正结果挂在 LSPE 名下）。day3 (a) GraphMAE 掩码预训练 → 负（39.69 vs 40.27）；(b) PSPE 位置编码 → 正，combined 35.44 K（dce 36.43、mi 36.65；注：该三元组混跨两轮——combined=35.44 取自第二轮 `3_3_day3_pspe`，其同轮 dce=36.19/mi=35.99；dce 36.43/mi 36.65 取自第一轮 `3_day3_pspe`，详见备注）。day4 SelfGNN(BYOL) 自监督 → 负，ep10 塌陷 cos→0.999，baseline 40.55±0.75 未复现 35.44。day5 (a) GPS-Lite 全局注意力 → 负（43.14/43.43，+8 K）；(b) MoMuCon(NT-Xent) 对比预训练 → 负（38.75 vs 35.44，−3.31 K，但 cos~0.39 未塌陷）。day6 CPA-GAT（sigmoid-gated unnormalized sum，全层）→ 决定性正：34.77±0.21 K vs baseline 36.38 K（+1.61 K, 4.4%），打破指数衰减；CPA 仅末层 null（36.47），degree baseline 有害（37.48）。day7 (a) CPA+PSPE → null（35.86 vs 35.52，聚合层 subsume 链几何）；(b) PolyBG-CPA 并行 null（35.64 vs 35.80）、层级崩塌（41.75/110.25），bond-graph 范式关闭。

**多种子协议**：探索期多数实验 3 seeds（42/123/456 或 71/72/73 或 42/71/123），个别单 seed 消融；is_backbone 的最早系统验证在 `6_day7_cpa_pse` 条件 E（CPA+is_backbone, 43-dim）5 seeds {42,71,123,888,2024} = 35.03±0.73 K，击败 42-dim CPA baseline（35.39）与同组的 modular_position(36.29)/periodic_distance(35.83)/PSE-full(36.00)，确立 is_backbone 入选 ORIG_BB。

**消融**（相对 ORIG_BB / CPA）：去 CPA（baseline GAT）36.38；CPA 仅末层 36.47；degree 替代 37.48；去 is_backbone（42-dim CPA）35.39；加 PSPE/PSE 不增反降（35.86/36.00）；加 bond-graph 并行/层级 null 或崩塌；GPS-Lite 全局注意力 +8 K。结论：聚合层 CPA 是唯一稳定正贡献，与 spectral RWPE、atom chemistry 互补，与链几何/bond 特征冗余。

**头部 finalize**：`src/optimize` 用 ORIG_BB（CPA-GAT + is_backbone + SWA, lr 1e-4）跨 5 数据集做 27-run 干净复跑，全部完成，输出 RESULTS.md / results_table.json / ensemble_*.json。状态：头部模型与全部 27 runs 均成功（无 killed）；探索期仅 day1 multimer 的 1/36 run 超时丢失、个别层级实验（polybg hierarchical exp6/7/8）出现崩塌但已记录。

## 备注

- 主要参考文件：`src/optimize/{README.md, plan.md, config.py, model.py, train.py, data_utils.py, run.py, run_all.sh, ensemble.py, collect_results.py}`、`src/optimize/results/{RESULTS.md, results_table.json, ensemble_tg.json, tg_sp2_init42.json, tm_sp2_init42.json, egc_spgiven_init42.json, ea_spgiven_init42.json, ip_spgiven_init42.json}`、`src/optimize/logs/{gpu0.log, gpu1.log}`、`mem.json`、`mem_day1.json`，以及各 day 实验的 `results_summary*.json / report.md / summary.json`（1_day1_multimer_feature_gat、1_day1_followup_grin_elene、2_day2_spse_maxgat、2_day_2_lspe_gat、3_day3_pspe、3_3_day3_pspe、3_day3_graphmae_pretrain、4_4_day4_selfgnn_polymer、5_day5_gps_lite、5_day5_momucon_finetune、6_day6_cpa_gat、6_day7_cpa_pse、7_day7_cpa_pspe、7_day7_polybg_cpa）。
- 本份已经过二次校验（本轮 verdict: accurate），本轮修了：流程 day2 LSPE/static-RWPE 归属（LSPE 本身负向 37.96/37.85 K、正收益属 static RWPE 36.29 K）、删除来源不明的 36.85 K baseline、day3 PSPE 三元组加 inline 跨轮限定。上一轮已修正两处架构描述（readout LayerNorm 无条件开启、单/共聚物参数差 768）。所有 RMSE/ensemble/per-split headline 数字经核对与 results_table.json/RESULTS.md/ensemble_tg.json/per-run JSON 完全一致。
- 不确定项：
  - split seed 文档不一致：`plan.md` 正文写 split_seed∈{42,71,2026}、`run.py` docstring 写 7 splits {42,71,2026,100,2025,2002,2027}，但 `config.SPLIT_SEEDS=[2,42,71]` 且实际 results 文件只覆盖 sp{2,42,71}；实际执行的是 3 splits {2,42,71}、共 27 runs，本报告按实际文件报告。
  - day3 PSPE 的 combined=35.44 取自第二轮 `3_3_day3_pspe/report.md`（combined=35.44, dce=36.19, mi=35.99），而正文引用的 dce 36.43 / mi 36.65 取自第一轮 `3_day3_pspe/report.md`——每个数字 individually 真实，但 dce/mi/combined 三元组混跨了两轮。
  - is_backbone 最早系统验证出现在 `6_day7_cpa_pse` 条件 E（35.03±0.73, 5 seeds）；是否在 day6 cpa_gat 内部已非正式试过，文件无法确认（day6 cpa_gat 用 42-dim，未含 is_backbone）。
  - Tg SOTA 30.8（PerioGT）与 GAT baseline 50.5 来自 `mem.json` 的 knowledge 条目，archive 内未找到对应论文 PDF/可核验来源。
  - simple 预测头里的 Dropout 被硬编码为 0.1，与 ORIG_BB 全局 dropout=0.0 不一致；无法从配置项调整，视为代码常量。
  - 边特征（4 维）在 data_utils 中被计算并写入每个图的 edge_attr，但 ORIG_BB use_edge=False → edge_dim=None，头部模型实际不消费边特征；仅未启用的 v2 变体会消费。
  - 共聚物 EA/IP 的官方 split pkl 来源（ea_split.pkl/ip_split.pkl）与 rA/rB/type 列的物理含义（组成比/共聚类型）未在 `src/optimize` 内单独文档化，推断自 data_utils 代码。
  - Tm 跨 split 方差较大（±1.88 K，主要由 sp2 57.22 vs sp42 52.69 拉开），是否为数据集本身 split 敏感还是协议问题未做进一步诊断。
