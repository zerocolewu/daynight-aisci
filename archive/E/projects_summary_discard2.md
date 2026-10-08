# Chemprop-Polymer dense-MPNN (dense_d5_h1200) - ab5 (DeepSeek, full day-night, 跨基座 LLM 鲁棒性)

> 一句话定位：ab5 / 基座 DeepSeek / full day-night（昼知识吸收 + 夜反思） / 最终模型：Chemprop-Polymer dense-MPNN（有向键级消息传递 + DenseNet 风格密集跳跃连接，config=dense_d5_h1200），头部目录 `src/6_06_chemprop_hpo`。

## 实验目的

这是 Poly-Circa AI-Scientist 的第 5 个 autonomous ablation run（ab5），基座 LLM 为 DeepSeek、full day-night 连续多日自主科研模式，目的是验证该自主科研流水线在"换一个不同基座 LLM"后是否仍能复现/逼近同类 run 的最佳模型与性能（跨基座鲁棒性）。落点到任务是聚合物 Tg（玻璃化转变温度）回归：以 Tg 数据集（7170 条 one repeat-unit p-SMILES，baseline GAT RMSE=50.5，SOTA PerioGT RMSE=30.8）为 sandbox。run 早期（Day1-4）按"数据中心 → 架构改进 → 自监督预训练"三类先验系统试错（GIN/GPS+KAN、JEPA+对比学习、Bloch 光谱、PCFT 等），均未突破 50；Day5 发现 Chemprop 风格的"有向键级消息传递"把 RMSE 从约 50 拉到 43.46，且证实是"架构机制"而非"周期边拓扑"在起作用（机制贡献 6.73 vs 拓扑 0.46，约 14:1）；据此 Day6 设计最终/最佳模型——对 Chemprop-Polymer MPNN 做系统性超参优化（HPO）并引入 DenseNet 风格的密集跳跃连接（缩短梯度路径 + 特征复用），得到 dense_d5_h1200。架构设计动机：聚合物图小、消息路径短，加深 + 加宽（depth/hidden）能显著降 RMSE，而 dense skip 期望在深网下缓解过平滑并进一步降误差（但实际 dense 增益很小，见流程/备注的诚实修正）。

## 模型架构

最终模型 ChempropPolymerMPNN（config = dense_d5_h1200，定义于 `src/6_06_chemprop_hpo/chemprop_model.py`），输入到输出逐模块如下：

(1) **输入图构造**（construct_polymer_graphs.py，construct_trimer_graph，add_periodic_edge=True）：每个 p-SMILES → 解析 repeat unit + 两个 star 端点 → 复制 3 份拼成 trimer（含 RU 间连接键）→ 在两个边界（端）原子之间加一条"周期边"模拟无限链周期性。每条无向边拆成 2 条有向边，并为每条有向边预算其反向边索引 reverse_map。

(2) **原子特征** ATOM_FDIM=123：原子序数 one-hot(100) + 度 one-hot(6) + 形式电荷 one-hot(5) + 杂化 one-hot(5: SP/SP2/SP3/SP3D/SP3D2) + 芳香性(1) + 成环(1) + 氢原子数 one-hot(5)。

(3) **边特征** BOND_FDIM=9：键类型 one-hot(4: SINGLE/DOUBLE/TRIPLE/AROMATIC) + 共轭(1) + 成环(1) + is_periodic(1) + 电负性梯度 en_gradient(1, 仅周期边=|端原子电负性差|) + 边界原子度 boundary_degree(1)。RU 间键与周期边用同一 9 维模板（非周期位=0）。

(4) **嵌入/消息传递**（depth=5, hidden_dim=1200，directed bond MPNN，Chemprop 风格）：
- W_init: Linear(atom_fdim×2 + bond_fdim = 255 → 1200)，由 [源原子特征 ‖ 目标原子特征 ‖ 键特征] 初始化每条有向键的消息 m0；
- 每步：先把消息按 target 求和聚合到原子（scatter sum），W_a: Linear(atom_fdim + hidden = 1323 → 1200) 更新原子隐状态 h_atom=ReLU(W_a([atom ‖ agg]))；再用 W_h: Linear(hidden + hidden + bond_fdim = 2409 → 1200) 以 [h_source ‖ m_reverse ‖ bond_feat] 更新有向键消息 m_{k+1}=ReLU(W_h(·))；W_h/W_a 跨步共享。
- **Dense skip connections**（dense_connections=True，本配置核心）：第 i>0 步前，把第 0..i-1 步的消息向量拼接（i×1200 维）经 dense_projections[i-1]: Linear(i×1200 → 1200) 投影回 hidden 再聚合，共 depth-1=4 个 dense 投影层（Linear(1200→1200)、Linear(2400→1200)、Linear(3600→1200)、Linear(4800→1200)）。
- W_o: Linear(atom_fdim + hidden = 1323 → 1200)，最终原子表示 z_atom=ReLU(W_o([atom ‖ agg_final]))。

(5) **Pooling**：global mean pool（scatter reduce='mean'，按 atom_batch）得到图表示 z_graph(1200)。

(6) **预测头**：MLP Linear(1200→1200) → ReLU → Dropout(0.1) → Linear(1200→1200) → ReLU → Dropout(0.1) → Linear(1200→1)。

(7) **归一化**：无 BatchNorm/LayerNorm（代码中无任何归一化层）。目标也**不做归一化**，直接在 K 单位上回归。

(8) **Dropout=0.1**（仅预测头）；消息传递中无 dropout。

(9) **激活**：ReLU。

(10) **参数量** n_params=23,665,201（≈23.7M）。相对非 dense 基线（combined_best = 9,260,401 ≈9.26M）的 dense-skip 增量为 23,665,201 − 9,260,401 = 14,404,800 ≈**14.4M**，全部来自上述 4 个 dense_projections 层。（注意：9.26M 是非 dense 基线自身的参数量，**不是** dense 增量；早期表述"dense 贡献 9.26M 额外参数"有误。）

(11) **损失**：MSE（nn.functional.mse_loss(pred, target)）。

## 训练评估方式

**数据集与划分**：Tg.csv，解析后得 6960 个聚合物图（原始 7170，约 210 条 p-SMILES 解析失败被跳过），使用 Day5 预计算的 graphs_polymer.pkl。划分 72/8/20：train=5011 / val=557 / test=1392（sklearn train_test_split，先切 20% test，再在 trainval 内切 10% 作 val）。

**种子（重要约定）**：split 种子 = init 种子 = 同一个 seed 参数（chemprop_hpo.py 的 train_model 中 split_dataset(data, seed=seed) 与 torch.manual_seed(seed) 用同一值），二者**未解耦**。Phase-4 多种子用 {42, 71, 123}；Phase 1-3 单种子用 71。

**预处理**：RDKit 解析 p-SMILES → 去星原子 → 3 份 repeat unit 拼 trimer + 周期边 → 转 tensor（atom_features/bond_features/edge_index/reverse_map）。

**优化器**：Adam，lr=1e-3（best），无 weight_decay，无 lr scheduler。梯度裁剪：clip_grad_norm_ max_norm=1.0。

**batch_size（已修正）**：头条多种子验证（Phase-4，40.00±2.09）实际用 **bs=50**——产出 `results_phase4.csv`/`summary.txt` 的代码路径是 `continue_phases.py`，其 Phase-4 循环调用 `train_model(...)` 时**未传 batch_size**，故用 train_model 的默认值 50，`run_phase3_4_log.txt` 中三次 Phase-4（seed 42/71/123）均明确记录 bs=50。`chemprop_hpo.py` 内 run_phase4 helper 虽对 hidden>600 设了 bs=25，但**并非**产出最终结果的代码路径。Phase-3 的 dense 探索阶段用 bs=16。epochs：max 200；early stopping patience=30（按 val RMSE 选最优 epoch，restore best state_dict 后再评估 train/val/test）。

**评估指标**：RMSE / MAE / R²，主指标 test RMSE（K 单位）。其中 RMSE 实为 numpy 手算 `np.sqrt(np.mean((preds-targets)**2))`，仅 MAE / R² 用 sklearn（`mean_absolute_error` / `r2_score`）——sklearn 无内置 RMSE，故 RMSE 用 numpy 手算合理；并记录 overfitting_gap = train_RMSE − test_RMSE（负值=欠拟合）。硬件：单卡 NVIDIA RTX 2080 Ti（11.3 GB），CUDA。损失函数：MSE。无 ensemble（chemprop_hpo 未做模型集成，仅多种子求均值/标准差）。

## 结果

**头条模型**（Tg，K 单位，`src/6_06_chemprop_hpo/summary.txt` + results_phase3.csv + results_phase4.csv）：

最佳单种子（dense_d5_h1200, seed=71, **Phase-3, bs=16**）：Test RMSE = 35.48（35.4758），Val RMSE = 35.44，Train RMSE = 20.94，overfitting_gap = -14.53（欠拟合），best epoch = 179，R²(test) ≈ 0.895。

多种子（Phase-4, final_best_dense_d5_h1200, seeds {42,71,123}, **bs=50**）：
- Seed 42: Test 40.95, Val 31.45, best_epoch 140, R²(test)=0.860
- Seed 71: Test 37.11, Val 36.44, best_epoch 170, R²(test)=0.885
- Seed 123: Test 41.95, Val 39.82, best_epoch 46, R²(test)=0.855
- 多种子 mean±std：Test RMSE = 40.00 ± 2.09，Train RMSE = 28.32 ± 5.26，gap = -11.68 ± 3.99

⚠️ **协议不一致提示**：头条最佳单种子 35.48 是 Phase-3 用 bs=16 训练的；其 Phase-4 多种子验证 40.00±2.09 用的是 bs=50。两者 batch size 不同，**不是严格同一训练协议**——多种子数字不能视为单种子 35.48 的同协议复现，interpret 时需注意。

HPO 中间关键点（seed=71）：Phase1 OFAT 最佳 C_lr1e-03 = 36.57；Phase2 combined_best (d7,h1200,lr1e-3,do0.1) = 35.72；Phase3 dense 三配置均 ≈35.5（dense_d5_h1200 35.48 最优，dense_d7_h1200 35.73，dense_combined_d7_h1200 35.78）。

**与基线/SOTA 对比**（Tg）：Day5 Chemprop-Polymer baseline（C_polymer_full，3 种子）43.46 ± 0.89 → 头条单种子 35.48（改进 7.98）、多种子 40.00 ± 2.09（改进 3.46）；本 run 的 GAT baseline（src/baseline, seed=42）50.52；PerioGT SOTA 30.8。头条单种子 35.48 仍高于 SOTA 4.68；多种子 40.00 高于 SOTA 9.20。结论：DeepSeek 基座 run 把 Tg 从 50 量级压到 35.48（单种子）/ 40.00（多种子），显著缩小与 SOTA 的差距但未超越。

**其他性质**（本 run 仅 baseline GAT 单种子 seed=42，src/baseline/results/*_baseline_report.txt）：Egc（2433/271/676）RMSE=0.6326, MAE=0.4615, R²=0.8361；Tg（4971/553/1382）RMSE=50.5182, MAE=37.4409, R²=0.7912；Tm（2615/291/727）RMSE=79.2976, MAE=60.4233, R²=0.5026（最难预测）。EA/IP：本 run 未涉及。无 ensemble 结果。

## 流程

5 个 logged day + Day6（在 Day5 末规划，对应 src/6_06_*）的推进（lite.log 的 Day 边界 + 各目录 README/results）：

**Day1**（数据中心基础）：固定 3 层 EdgeAwareGINConv（hidden 256）测 5 种结构变换，trimer 最佳 Test 50.19（val 39.65），cyclized 过拟合；另测 KA-GPS（KAN 增强 GPS，50.8/51.2）、GIN-GAT/Virtual-Node（51.14–54.01）。结论：trimer 是最佳表示，GIN 架构是瓶颈（vs SOTA 30.8）。状态：完成。

**Day2**（自监督预训练）：JEPA v1（池化 bug，54.79）；JEPA v2+VICReg 对比（contrastive 端到端微调最佳 50.77，比无预训练 54.89 降 4.12，但未破 50.19）；GPS-SPSE（5-block GPS+最短路径编码，多种子 54.41 ± 0.84，欠拟合）。状态：完成，增益有限。

**Day3**：PerioGT-Lite（精确复现 Day1 GIN 以公平评估对比预训练）；PCR-v2 混合池化（监督 sum / 对比 mean）——baseline_sum_pool 77.50 严重退化，混合池化失败。状态：完成（PCR-v2 方向被判失败）。

**Day4**：PCFT/PGFT 双编码器 prompt 条件化微调（多种子 [42,71,123]）——baseline 52.46±0.20 最佳，PA 预训练 PCFT 52.72±0.76 无改进；BlochSpec v1（add_pool，baseline 66.15、BlochSpec 61.52 但随机特征同样有效→发现池化 bug）；BlochSpec v2（修 mean_pool，baseline 51.71±0.33、BlochSpec 53.25±0.51，光谱特征无实质帮助）。状态：完成，高级技巧均未带来改进。

**Day5**：Chemprop-Polymer 有向键 MPNN，6 项消融 A-F（每种 3 种子 [42,71,123]）：C_polymer_full 最佳 43.46 ± 0.89，full polymer 图 > trimer(43.92) > monomer(45.76)；关键发现 random periodic edge(43.58) 与 correct edge 几乎相同→"架构机制主导"（机制 6.73 vs 拓扑 0.46，14:1），为 Day6 HPO 提供依据。PolymerBipartite v2（边界/内部 GIN+门控，53.81 ± 0.94）不如参数匹配 baseline。状态：完成。

**Day6**（`6_06_chemprop_hpo` + `6_06_chemprop_pa`）：
- (a) **chemprop_hpo**：4 阶段 HPO。**原 `run_log.txt`（chemprop_hpo.py 主入口）在 Phase-3 的 dense_combined_d7_h1200 处因 dense_projections 维度不匹配（"mat1 and mat2 shapes cannot be multiplied 5544x2400 and 1200x1200"）抛 RuntimeError 崩溃**；随后用独立的 `continue_phases.py` 修复 dense 投影 bug 并续跑，`run_phase3_4_log.txt` 末尾才出现 "DONE."。即：Phase1 OFAT（depth 3/5/7、hidden 300/600/1200、lr 1e-4/5e-4/1e-3、dropout 0/0.1/0.2/0.3）→ Phase2 combined_best 35.72 → Phase3 dense skip（dense_d5_h1200 单种子 35.48 全局最优）→ Phase4 多种子验证 40.00 ± 2.09。状态：**经一次崩溃 + bug 修复续跑后完成**。
- (b) **chemprop_pa**（Chemprop-Polymer-K + 对比预训练）：仅 variant_A（无预训练 + K-node）完成 3 种子（lr1e-4 最佳 40.23±0.28，lr1e-3 44.86±2.71，lr5e-4 41.06±1.23）；variant_B/C/D（PI1M/随机图预训练）的**微调阶段因 CUDA OOM 被 killed**，`variant_A_lr5e-4_final.log` 留有 torch.OutOfMemoryError traceback。注：`checkpoints/` 目录确为空，但 `results/` 下存在 161 MB 的 `pretrain_pi1m_knode.pt`（足以容纳约 15.2M 参数的 K-node 编码器 model+optimizer state），说明 **PI1M K-node 预训练至少保存了权重**——所以更准确的说法是"预训练权重存在、但预训练→微调链路未完成（微调变体 OOM killed）"，而非"预训练完全未产出 checkpoint"。状态：chemprop_hpo 完成；chemprop_pa 部分失败。
- (c) **fix_phase4（未完成的协议修正尝试，直接关系头条 bs=16 vs bs=50 协议不一致的处置）**：`fix_phase4.py` 是一次针对上述协议不一致的修正实验——脚本注释明确动机 "Phase 3 best used bs=16, Phase 4 mistakenly used bs=50"，计划用 **bs=16 重跑 Phase-4 dense_d5_h1200 三种子 {42,71,123}**，并额外跑非 dense d7_h1200 多种子做公平对比。**该重跑未完成**：`fix_phase4_log.txt` 仅 39 行、无 "DONE"、未跑 seed123、未跑 no-dense 段，`results_phase4_v2.csv` / `results_phase4_nodense.csv` 均未生成；仅 seed42 跑完（Test RMSE=41.7050，存 `trained_models/v2_dense_d5_h1200_seed42.pt`），seed71 仅训练到中途即中断。其部分结果（41.7）与 bs=50 多种子 40.00±2.09 一致，**不改变结论**，但说明头条协议不一致问题曾被尝试修复、未果。状态：未完成（中断）。

**多种子协议**：seeds {42, 71, 123} 贯穿 2_gps_spse_trimer / 4_04_pgft / 4_4_blochspec / 4_4_4_blochspec / 5_day5_polymer_bipartite_v2 / 5_6_chemprop_polymer / 6_06_chemprop_hpo(Phase4) / 6_06_chemprop_pa。注意 chemprop 系列里 split 种子与 init 种子未解耦（同一 seed 同时控制划分与初始化）。

**主要消融**：(i) Day5 graph 构造 A-F（周期边拓扑贡献边际）；(ii) 4_4/4_4_4 光谱特征 vs 随机特征（无真实增益）；(iii) Day6 OFAT depth/hidden/lr/dropout + dense-skip on/off；(iv) Day2 JEPA vs contrastive（对比学习更优）。

**dense-skip 增益的诚实修正**：早期表述"dense 在深网下降约 0.2–0.3 RMSE"**不被日志数字支持**——dense_d7_h1200(35.7317) 反而比非 dense 的 combined_best d7_h1200(35.7190) 略差约 0.013；唯一 dense 胜出的 dense_d5_h1200(35.48) 并没有匹配的非 dense d5_h1200 单种子 run 可直接对比。因此 dense-skip 的实际增益很小/不确定，不宜夸大。

**已完成**：1_01、1_day1_arch_improve、1_2025_07_11_ka_gps、2_polymer_jepa、2_2_polymer_jepa、2_gps_spse_trimer、3_periogt_lite、3_day3_pcr_v2、4_04_pgft、4_4_blochspec、4_4_4_blochspec、5_6_chemprop_polymer、5_day5_polymer_bipartite_v2、6_06_chemprop_hpo、baseline。**Killed/失败**：6_06_chemprop_pa 的 B/C/D 预训练微调变体（OOM）、3_day3_pcr_v2 混合池化方向、chemprop_pa variant_A 的部分 lr 重跑（OOM）、chemprop_hpo 原 run 在 Phase-3 的 dense 维度 bug 崩溃（后经 continue_phases.py 修复续跑完成）。

## 备注

- 主要参考文件：`src/6_06_chemprop_hpo/{chemprop_model.py, chemprop_hpo.py, continue_phases.py, construct_polymer_graphs.py, summary.txt, results_phase1.csv, results_phase2.csv, results_phase3.csv, results_phase4.csv, run_log.txt, run_phase3_4_log.txt, fix_phase4_log.txt, trained_models/}`、`src/6_06_chemprop_pa/{README.md, results/experiment_log.txt, results/variant_A_*.pkl, results/variant_A_lr5e-4_final.log, results/pretrain_pi1m_knode.pt}`、`src/5_6_chemprop_polymer/{ablation_summary.csv, ablation_results.csv}`、`src/baseline/results/{Egc,Tg,Tm}_baseline_report.txt`、`lite.log`、`full.log`、`mem_day1.json`。
- 本份已经过二次校验。上一轮（verify verdict: has_issues）修正了 4 处实质错误：① Phase-4 多种子实际 bs=50（非 25），并补出"头条单种子 35.48 用 bs=16 vs 多种子 40.00±2.09 用 bs=50"的协议不一致；② dense-skip 参数增量是 14.4M（非 9.26M，9.26M 是非 dense 基线参数量）；③ 最佳模型 .pt **确实存在**于 trained_models/（dense_d5_h1200.pt ≈94.7 MB、final_best_dense_d5_h1200_seed{42,71,123}.pt 等，可加载）；④ chemprop_hpo 原 run 在 Phase-3 因 dense 维度 bug 崩溃、后经 continue_phases.py 续跑才 DONE。并修正了 dense-skip 增益被夸大的表述、缓和了 chemprop_pa"预训练无 checkpoint"的说法（pretrain_pi1m_knode.pt 存在）。本轮（verdict: accurate，**无 corrections**）仅补全 3 处 missing_details：① 流程 Day6 新增 fix_phase4（针对 bs=16 vs bs=50 协议不一致的未完成修正重跑，seed42 Test RMSE=41.7050，与多种子 40.00±2.09 一致、不改变结论）；② Day6 归类依据所引 full.log "Ideas for Day 6..." 逐字短语经 grep 无法证实，改为目录命名 + Day4 "Chemprop+periodic edges" 选项推断；③ RMSE 实为 numpy 手算（仅 MAE/R² 用 sklearn），表述更精确。
- 不确定项：
  - 基座 LLM = DeepSeek、模式 = full day-night 来自任务描述，未在 lite.log/full.log 中直接检索到模型名自标识（日志里出现的是 DeepSeek/agent 推理内容，不是配置声明），无法从归档日志 100% 独立确认。
  - lite.log 只记录了 Day1-Day5 的边界；6_06_chemprop_hpo 与 6_06_chemprop_pa 归为 Day6 是依据 `6_06_*` 目录命名 + full.log 中 Day4 已出现 "Chemprop+periodic edges" 选项推断（此前所述 full.log 中 "Ideas for Day 6: Chemprop hyperparameter optimization..." 逐字短语经对 full.log 全文 grep "ideas for day 6" **无法证实存在**；full.log 末尾仍在运行 06_chemprop_pa，未出现清晰的 Day6 ideas 段或 "Day 6 | start" 边界）。精确的 Day6 起止时间戳未在日志中以 "Day N | start" 格式给出。
  - chemprop 系列的 split 种子与 init 种子未解耦（同 seed 同时控制两者），因此多种子 std 同时混合了划分差异与初始化差异，无法分离两者贡献。
  - chemprop_pa variant_A 三个 .pkl 的 summary 均含 3 种子结果，但对应 .log 多数只到 epoch 10 左右（被截断/覆盖）；无法确认 .pkl 内数字来自哪一次完整跑，仅以 .pkl summary 为准报 40.23±0.28 等。
  - EA/IP 在本 run 完全缺失（仅 baseline 有 Egc/Tg/Tm），无法给出 EA/IP 数字；头条模型与所有 day 实验均为 Tg 单任务。
  - Day1-Day4 的逐项 RMSE 数字（trimer 50.19、contrastive 50.77、GPS-SPSE 54.41、PCFT 52.46/52.72、BlochSpec 66.15/61.51/51.71/53.25、PCR-v2 sum-pool 77.50 等）主要来自 `projects_summary_discard.md`，仅 Day5/Day6/baseline 的关键数字已用原始 CSV/log 交叉核实，Day1-Day4 逐项未逐一回到原始结果文件核验。
