# ab3（GLM-5，FULL DAY-NIGHT，多性质）逐天实验轨迹 — Day2 GRIN 32.31 为全局最优，terminal TCP 负迁移

## 概览与约定

- **run 身份**：ab3 / 基座 LLM = GLM-5 / **FULL DAY-NIGHT 模式**——白天（论文阅读+假设+实验）与夜晚（internalize 巩固+associate 生成 idea）双循环均保留 / 共 7 天 / **无独立 optimize 目录**（Day7 TCP 为 terminal/末天模型）。与 ab2 同基座同模式（彼此互为随机性重复），区别在于本 run 扩展到**多性质** Tg/Tm/Egc。
- **任务范围**：Day1–Day2、Day5–Day7 聚焦 Tg；**Day3 起引入 Tm（熔融温度）、Egc（带隙能量）**；Day3–Day7 为 Tg/Tm/Egc 多任务或对照。EA/IP 完全未涉及。
- **最终最强模型**：**全 run 真实最优是 Day2 GRIN 的 Tg 32.31 K（单种子单划分 test RMSE）**——非 terminal；terminal/末天最佳是 Day7 TCP fine-tuned Tg **35.49±1.60**，但相对 scratch baseline 35.22±1.55 是**负迁移（−0.28 K，误差棒内不显著）**。多性质最优：Tm **59.18±1.12**（Day4 multi-seed ensemble）、Egc **0.5262±0.0471**（Day4 共轭特征+分层 CV）。
- **主线轨迹**：(D1) HEPT 重等变失败 / RE-PerioGT 刚性特征微胜 → (D2) **GRIN 32.31 突破（全局最优）** + PerioGT 两版论文澄清 → (D3) PF-CL 预训练负迁移 + Tm/Egc baseline → (D4) 多性质成功（Tm ensemble 59.18、Egc 共轭 0.5262）→ (D5) FD-MTL 多任务（AI 自述"灾难失败"但代码显示 Tm/Egc 达标）→ (D6) CFD-MTL 修正版多任务正迁移 → (D7) DCN-GIN 深度扩展 + TCP terminal 负迁移。
- **判别关键**：full day-night 与 day-only（ab4）的根本区别是"夜巩固的 provenance 边与跨天因果累积"。本 run 跨天衔接较强：Day6 CFD-MTL 明确"纠正 Day5 FD-MTL"、Day7 TCP 明确"吸取 Day3 PF-CL 三教训"、Day7 DCN 以 Day2 GRIN 32.31 为追赶基线、Day4 Tm/Egc baseline 为 Day6 多任务硬编码基线。
- **单位约定**：Tg/Tm RMSE 单位 K（开尔文温度差，量纲同 °C）；Egc RMSE 单位 eV（带隙能量）。
- **可信度约定（本 run 尤其重要）**：所有数字以**代码/训练日志/原始 results JSON 为权威**。本 run 多处 **AI 当夜自述文档（SUMMARY.md / FINAL_REPORT.md / mem experience）的核心数字与代码输出矛盾**——Day5、Day6 尤为严重，前者 AI 自述"灾难失败 117/118/1.68"无训练日志支持（代码实为 MTL 接近基线），后者 AI 自述 Tg 44.12/Tm 59.93/Egc 0.5521 在所有代码输出中检索不到。凡 AI 独有、代码未佐证的标"（AI 当夜自述，代码未独立验证）"。特殊缩写在首次出现处展开。

### day→实验目录映射

| Day | src/ 目录 | 主题 | 代表结果 |
|---|---|---|---|
| 1 | `1_hept`、`1_day2_reperioGT` | HEPT 重等变周期 Transformer / RE-PerioGT 刚性特征 | HEPT 49.58 失败；RE-PerioGT 35.56 微胜 |
| 2 | `2_02_PerioGT_Ablation`、`2_day2_grin` | PerioGT 两版消融 / GRIN 假设检验 | **GRIN 32.31（全 run 最优）**；GraphFormer 37.00 超 2022 论文 |
| 3 | `3_pfcl_pretraining`、`3_day4_feature_baseline_tm_egc` | PF-CL 联合预训练 / Tm·Egc 特征 baseline | PF-CL 39.23 负迁移；Tm 86.60±53.05 / Egc 1.19±0.59 |
| 4 | `4_day4_ensemble_uq`、`4_egc_conjugation` | Multi-seed ensemble+UQ / 共轭特征+分层 CV | Tm 59.18±1.12 成功；Egc 0.5262±0.0471 成功 |
| 5 | `5_day5_fd_mtl` | FD-MTL 特征解耦多任务 | MTL-Decomp Tg 40.10 / Tm 58.92 / Egc 0.519（代码）；AI 自述灾难失败未验证 |
| 6 | `6_day6_cfd_mtl` | CFD-MTL 修正特征解耦多任务 | cfd_mtl Tg 39.42 / Tm 56.16 / Egc 0.5265；正迁移成立 |
| 7 | `7_day7_dcn_depth_scaling`、`7_day8_tcp` | DCN-GIN 深度扩展 / TCP 热力学一致性预训练 | DCN-15 38.40；TCP 35.49±1.60 **负迁移**（terminal） |

> 目录名常带跨天字样（如 `1_day2_reperioGT`、`3_day4_feature_baseline_tm_egc`、`7_day8_tcp`），一律以前缀数字为权威 day 归属。

---

## Day 1：HEPT 重等变周期 Transformer 失败 / RE-PerioGT 刚性特征微胜

### 实验目的
Day 1 在聚合物玻璃化转变温度（Tg）回归上做两个目标一致、路线相反的实验，均以击败 PerioGT 基线 Tg 30.8 K 为目标。（A）**HEPT（Hierarchical Equivariant Periodic Transformer，分层等变周期 Transformer）**：把 MACE 式 E(3) 等变局部消息传递、PerioGT 式周期边注意力、LSR-MP 式长程全局上下文三者揉进一个统一架构，赌"等变+周期+长程"叠加能超越只建模周期的 PerioGT。（B）**RE-PerioGT（Rigidity-Enhanced Periodic Graph Transformer，刚性增强周期图 Transformer）**：放弃 3D 等变路线，退回周期图 Transformer 基座，注入聚合物链刚性物理先验（可旋转键占比）并辅以刚性驱动的对比学习，检验"周期性+链刚性"是否更匹配 Tg 物理本质。

### 模型架构
（A）**HEPT**（`1_hept/model.py`，v1 hidden 256，2,876,804 参数）：三模块串行——等变局部模块（径向基函数 RBF 64 个高斯编码距离、实球谐 l_max=2 共 9 分量编码方向、方向调制消息传递，但节点隐藏态始终是标量故**非真 E(3) 等变**）；周期注意力模块（`create_periodic_images` 写了虚镜像生成但 `forward` 注释 "simplified - just use self-attention"，**周期镜像被注释绕过**，退化为带 batch 掩码的标准多头自注意力）；长程模块（scatter_mean 全局上下文+门控融合）。readout 用可学习权重 softmax 加权 mean/sum/max。节点 36 维。（B）**RE-PerioGT**（`1_day2_reperioGT/model.py`，hidden 256，4 层 4 头，atom 153/bond 11 维）：基座 `PeriodicGraphTransformerLayer`（Q/K/V 注意力+边特征加到 K+FFN+残差+LN），周期性经 `periodic_mask`（* 通配原子标记）与 `periodic_vectors`（通配原子归一化连线）拼 4 维编码加到节点嵌入。刚性 = `CalcNumRotatableBonds / GetNumBonds`（可旋转键占比，值越大链越柔，命名 rigidity 实为柔性的对称量），经 RigidityEncoder 3 层 MLP 与图嵌入拼接进预测头。对比损失 `ContrastiveLoss` 为 InfoNCE 形式（刚性差<0.1 正对、>0.3 负对、温度 0.1、权重 λ_cl=0.1）。

### 训练评估方式
同份 Tg.csv 7170 样本。HEPT 用**分层抽样**（按 Tg 分 10 分位 bin 后每 bin 内 8:1:1，seed=42）→ 5732/713/725，AdamW(lr 1e-3, wd 1e-5) MSE ReduceLROnPlateau(factor 0.5 patience 10) 梯度裁剪 1.0 batch 32 max 200 ep 早停 20；v2 改 hidden 384/batch 24/lr 5e-4/wd 1e-4/max 300 patience 25+MMFF94 构象。RE-PerioGT 用**随机 shuffle**划分（seed=42，非分层），Adam(lr 1e-4) CosineAnnealingLR batch 64 max 500 patience 50；三模式各单跑：baseline（关刚性关对比）、feature_only（开刚性）、full（开刚性开对比）。**两实验均单种子（seed=42），无多种子平均、无交叉验证。** PerioGT 30.8 与 GAT 50.5 是 `run_experiment.py` 硬编码外部基准常数（非本 run 复现）。

### 结果
（A）**HEPT 失败**：v1 Test RMSE **49.58**（MAE 35.93，R²0.8041），远高于 30.8（差 +18.78，约 +61%），仅略好于 GAT 50.5；v2 加深加宽到 8,180,356 参数反降到 **53.84**（MAE 39.56，R²0.7689），明显过拟合。根因：其一周期镜像被注释绕过、等变非真等变，三创新模块都没真正生效；其二 18 维增强分子特征（MolWt/LogP/TPSA/可旋转键/芳环数/sp3 占比/Bertz 复杂度等）在 `data_loader.py` 算出并写入 `data.mol_features`，但 HEPT.forward 签名 `(x,pos,edge_index,batch)` 根本不接 mol_features——**特征算完即弃**（代码层面属实）。（B）**RE-PerioGT 部分成功**：baseline **36.41**（MAE 25.60，R²0.900，best@211）/ feature_only **35.56**（R²0.9046，best@207）/ full **35.55**（R²0.9047，best@245）。刚性特征带来 +0.85 改善（36.41→35.56）；但**对比学习几乎无增益**——full 相对 feature_only 仅 −0.01 RMSE，训练时间却暴涨 10 倍（30.7→316.9 min，因 ContrastiveLoss 有 O(B²) 相似度矩阵+Python 双层 for 循环），且 full 验证 RMSE 37.97 反比 feature_only 36.94 更差，说明对比项未改善泛化、测试持平只是噪声。

### 流程
Day 1 总体叙事是"用更重归纳偏置打周期图基线，重者翻车、轻者微胜"。HEPT 路线先堆叠 MACE/PerioGT/LSR-MP 三套机制，但周期镜像被绕过、增强特征未接入、参数膨胀过拟合，v1/v2 双双失败（49.58/53.84）。RE-PerioGT 随即转向轻量化物理先验：周期图基座+可旋转键占比刚性特征，baseline 即 36.41（远好于 HEPT 且接近 SOTA 区间），加刚性特征小幅降到 35.56，再加刚性驱动对比学习收益归零、代价骤增。两条线共同指向核心教训：**对聚合物 Tg，归纳偏置须匹配物理——周期性是主偏置，链刚性作显式特征有微弱正作用，而 3D 等变/重周期镜像/对比学习等更重机制既未正确实现也非 Tg 所需**。

---

## Day 2：GRIN 假设检验 Tg 32.31（全 run 突破/全局最优）+ PerioGT 两版论文澄清

### 实验目的
（A）**PerioGT 基线消融**：关闭任务描述给出的"PerioGT 基线 36.41 vs 30.8"差距，系统性澄清"PerioGT 基准到底 30.8 还是 37.35"。（B）**GRIN（Graph Repetition INvariance / Graph Representation with Invariance and Normalization）假设检验**：检验"max 聚合 + L1 稀疏 + {1-RU, 3-RU} 数据增强 → 重复单元尺度不变性 → 改善 Tg"能否突破 GraphFormer 基线 37.0，目标 Test RMSE < 35.0。

### 模型架构
（A）**PerioGT 周期图**（2022 论文核心创新）：把聚合物 p-SMILES（含 `*` 通配符）解析为周期图——在两个端原子间加一条 **periodic edge（周期边）**，使重复单元（RU=Repeat Unit）图具周期性；创新在图构造而非 attention。本 run 重实现 **Chemprop-style MPNN**（directed bond-level message passing，3 层 depth=3，hidden 300，2 层 FFN，133 atom+10 bond 特征，约 907,801 参数）与自研 **GraphFormer**（Transformer 注意力聚合 GNN）作对照。**两版 PerioGT 论文区分**（代码 `VERIFICATION_REPORT.md` 核实）：2022 版（Antoniuk et al., JCIM 62(21):5435）= Chemprop MPNN+周期图、**无预训练**、5-fold CV/50 ep → Tg **37.35**；2025 版（Wu et al., Nature Computational Science）= LiGHT Graph Transformer+在 **PI1M（约 100 万无标签聚合物）** 上做 periodicity-aware contrastive learning (PA) 自监督预训练+periodicity prompt-guided fine-tuning (PGFT)+PolymerGraph 虚拟节点 → Tg **30.8±0.2**。（B）**GRIN**（`2_day2_grin/models.py`，backbone 标准 GIN 5 层 hidden 256）三件套可切换：**MaxGINConv**（`aggr='max'`，消息聚合由 sum/mean 换 max，注：graph-level readout 仍为 mean）；**L1 sparsity**（对 MaxGINConv 参数 λ=5e-4）；**{1-RU, 3-RU} 增强**（RDKit 把通配邻原子连成线性链再删通配符，造单聚体/三聚体图，成功率 98.5%）。**148 维原子特征** = 原子序数 one-hot 119 + 度 11 + 杂化 6（SP/SP2/SP3/SP3D/SP3D2/other）+ 形式电荷 5 + 氢数 5 + 芳香 1 + 成环 1 = 148（代码 `TOTAL_ATOM_FEATURES=148` 核实；报告内一处括号"143"为文字笔误，弃用）；边 6 维。

### 训练评估方式
Tg.csv 7170，80/10/10 单随机划分（`train_test_split(random_state=42)`）→ 5736/717/717。**种子协议（关键）**：仅 split 用 random_state=42；全仓库无 `torch.manual_seed`/`np.random.seed`，`generate_report.py` 自述局限"Single Random Seed"。故 GRIN 所有数字均为**单配置、单种子、单划分的 test RMSE，非多种子 mean±std**。训练 AdamW(lr 5e-4, wd 1e-5) CosineAnnealingLR(eta_min 1e-6) max 300 ep patience 40 dropout 0.15 batch 64 梯度裁剪 1.0，目标 StandardScaler 归一化（mean 142.61, std 112.05）。PerioGT 消融另跑 5-fold CV（Chemprop 重实现，50 ep，lr 1e-3 常数，**未实现 Chemprop 默认 NOAM warmup+cosine**）。GraphFormer 等用 3 seeds {42,123,456}。

### 结果
（A）**PerioGT 消融**：本 run **GraphFormer 37.00±0.68**（seed42 36.16/123 37.82/456 37.03，`comprehensive_results.csv`+`best_config.json` 核实）击败 2022 PerioGT 37.35（差 0.35）；Chemprop 重实现 5-fold CV **46.89±1.41**（失败，+9.54，缺 NOAM LR schedule）；Chemprop dropout/weight_decay 消融 39.65/41.27。**结论**：30.8 需 PI1M 预训练，<32.0 在无预训练下不可达。（B）**GRIN 32.31 达标、假设被拒**（`results_v2_20260324_090902/summary.csv`+`baseline/results.json` 三重核实）：

| 配置 | Test RMSE | vs baseline | 结论 |
|---|---|---|---|
| **baseline（mean，无 max/L1/aug）** | **32.3109** | — | **BEST，达标 < 35.0** |
| mean_aug（mean+{1,3}-RU）| 32.38 | +0.2% | 可忽略 |
| max_only（max 聚合）| 34.03 | +5.3% | max 丢平均性信息 |
| max_l1（max+L1）| 44.54 | +37.8% | L1 过度正则化 |
| full_grin（max+L1+aug）| **46.34** | **+43.4%** | 最差 |

GRIN max+L1 假设被拒：max 丢失平均性信息（Tg 取决于平均性质而非极值，mean pool 保留更多结构）、L1 过度正则化（epoch-1 loss 11.24 vs baseline 0.31）、3-RU 与 1-RU 化学不等价（增强引入噪声而非不变性）。**148 维特征贡献 57.6%**（`(76.08−32.31)/76.08`）——此为迭代1 baseline（9 维默认特征）76.08 → 迭代2 baseline（148 维富特征+目标归一化+更好超参）32.31 的**总提升**，多项改动合并，**非纯特征消融**，AI 自述"feature engineering 57.6%"略过度归因（代码无法分离）。

### 流程
（A）三阶段：Phase1 通用架构筛选（GraphFormer/Chemprop 单+多种子）→ Phase2 按 2022 论文做 5-fold CV Chemprop 重实现（46.89，发现缺 NOAM schedule）→ Phase3 benchmark 核实，**发现"30.8 vs 37.35"实为两篇不同论文**，修正目标认知。（B）迭代1（`results_20260324_072344/`，失败）：9 维默认特征+增强管线有 SMILES→graph 转换 bug（batch 维度不匹配），仅完成 3/5（baseline 76.08/max_only 78.75/max_l1 77.94），full_grin 崩溃；迭代2（`results_v2_*`，修复）：148 维富特征+StandardScaler+RDKit 重写增强管线（98.5% 成功）+调参（hidden 128→256, lr 1e-4→5e-4, 200→300 ep），5/5 全完成，baseline 32.31 达标，GRIN 假设被拒。**32.31 是全 run Tg 全局最优，后续所有天均以此为基线但未超越。**

---

## Day 3：PF-CL 联合预训练负迁移（Tg 39.23）/ Tm·Egc 特征工程 baseline（方差极大）

### 实验目的
（A）**PF-CL（Periodic Feature-aware Contrastive Learning，周期特征感知对比学习）联合预训练**：在 PI1M（约 995K 高分子 p-SMILES，实际取 30,000 子集）上做联合自监督——同时优化 (i) 周期性感知对比学习（InfoNCE，对同一 RU 的两个增广视角）、(ii) 154 维分子特征预测（MSE），得预训练 GIN 编码器后迁移到下游 Tg，验证"对比+特征预测"联合预训练能否超过随机初始化 baseline 32.31。（B）**Tm/Egc 特征工程 baseline**：把 Day2 验证过的 148 维手工特征拼到 GIN 图嵌入，评估"GIN+工程特征"在 Tm/Egc 上的可达精度与方差，作后续预训练方法对照。

### 模型架构
（A）**GINEncoder**（`3_pfcl_pretraining/model.py`，node 37/edge 10/hidden 256，5 层 GINConv `Linear-BN-ReLU-Linear`+残差+Dropout 0.1，global_mean_pool 出 256 维再投影）+ 周期图构建（`PeriodicGraphBuilder`，两端 `*` 端原子用周期边相连）+ 增广（`PeriodicAugmentor` shift/extend 两视角）+ 对比头（256→128→64）+ 特征预测头（256→154）+ 微调头（256→128→64→1）。PFCLModel 总参约 884,052。（B）**GINWithFeatures**（`train_baseline_v2.py`，GIN 5 层 hidden 256，每层 GINConv+BN+残差，global_mean_pool 256 维）+ 特征分支（`Linear(148,256)` 投影后与图嵌入 concat 512 维→MLP 头）。**148 维特征** = RDKit 物化描述子（MolWt/HeavyAtomCount/环计数/FractionCSP3/PartialCharge/MolLogP/MolMR/TPSA/BertzCT/BalabanJ/HallKierAlpha/Kappa1-3/Chi0v-Chi4n/EStateIndex/FpDensityMorgan1-3/qed/SPS）+ 高分子专属（9 类原子计数、4 类化学键计数与比例、sp/sp2/sp3 杂化计数与比例、3-8 元环分布、12 类官能团 SMARTS 计数 alcohol/amine/carbonyl/carboxylic/ester/ether/amide/nitrile/halogen/nitro/sulfide/phosphate，截断/补零到恰好 148 维）。

### 训练评估方式
（A）预训练：PI1M 30,000 子集，20 ep batch 128 AdamW(lr 1e-3)+余弦退火 warmup 3 ep 混合精度，联合损失 `L=InfoNCE(τ0.07)+0.5·MSE_feature`，三套消融 Joint/Contrastive-only/Feature-only；Tg 微调 5-fold CV（KFold shuffle, random_state=42，单种子）batch 64 max 300 ep 早停 30-50。（B）Tm.csv（n=3636，15 失败）/Egc.csv（n=3374，6 失败），5-fold KFold（shuffle, random_state=42，**单种子无 seed 平均**），目标值与特征均 StandardScaler，AdamW(lr 1e-4, wd 1e-5) batch 64 max 500 ep 早停 50，反归一化后算 RMSE/MAE/R²。

### 结果
（A）**PF-CL 负迁移**（`FINAL_5FOLD_REPORT.md`，与各 fold 一致）：Tg RMSE **39.23±1.81**（5 折 36.59/37.79/39.91/41.66/40.19），**全部 5 折均劣于 baseline 32.31**，相对 +21.4%（差 6.92）→ 负迁移确认。根因（报告 `ROOT CAUSE ANALYSIS`）：(1) 任务冲突——特征预测面向性质、对比面向分子相似度，联合优化互相干扰；(2) 迁移失配——PI1M 预训练分布与下游 Tg 不对齐；(3) 超参不匹配；(4) 编码器预训练过拟合、微调适应不足。（B）**Tm/Egc baseline 方差极大**（`tm/egc_baseline_results_v2.json`）：Tm **86.60±53.05**（5 折 63.02/58.98/62.80/**192.56**/55.63，best fold5=55.63）；Egc **1.19±0.59**（5 折 1.28/0.68/**2.25**/1.12/0.63，best fold5=0.63）。**方差极大根因**：5 折中单折灾难性失败主导均值——Tm fold4 RMSE 192.56（R²=−2.27，其余仅 55-63）、Egc fold3 RMSE 2.25（R²=−0.96，其余 0.63-1.28），即某些 split 含 OOD/难样本时模型崩溃；单种子、数据集小且异质进一步放大不稳定。跨性质共享 top 特征：Tm/Egc 共同 3 个 FractionCSP3/sp3_frac/FractionCSP3_2。

### 流程
（A）建 PF-CL 模型+周期图构建器/增广器 → PI1M 30K 子集算 154 维特征并缓存（曾因静态缓存名 `pi1m_features.pkl` 与样本数不匹配报 IndexError，修复为 `pi1m_features_{N}.pkl` 加尺寸校验；`pretrain_contrastive.log` 即该 bug 的 100K 失败 traceback）→ Joint/Contrastive-only/Feature-only 三套预训练 → 加载 Joint 编码器做 Tg 5-fold 微调 → 汇总。**注**：Tg 微调脚本 `finetune.py` 与 `checkpoints_joint/best_encoder.pt` 已不在目录（仅报告留存），Tg 数字均来自报告 md。（B）对每个 p-SMILES 算 148 维特征 → 建 PyG Data → 5-fold KFold 训练 GIN+特征（500 ep 早停）→ 反归一化算 RMSE/MAE/R² → RandomForest（100 树）做事后特征重要性 → 存 results v2 JSON。**Day3 两个子实验都暴露了"单种子方差/预训练分布失配"问题，直接驱动 Day4 multi-seed ensemble 与 Day7 TCP 分布感知采样设计。**

---

## Day 4：多性质成功天 — Tm ensemble 59.18±1.12、Egc 共轭特征 0.5262±0.0471

### 实验目的
（A）**Multi-seed ensemble + 不确定性量化（UQ=Uncertainty Quantification）**：针对 Day3 单种子 Tm 跨折方差 ±53.05、Egc 跨折 ±0.59 的问题，验证"5 seed×5 fold 多种子集成能否降方差并压低 RMSE，同时提供可信逐样本不确定性"。Tm 目标 RMSE < 70，Egc 目标 RMSE < 0.9。（B）**共轭特征（conjugation-specific features）+ 分层交叉验证（stratified CV）**：针对（A）暴露的 Egc Fold3 持续失败（ensemble 2.11, R²=−0.72），验证假设"Fold3 失败根因是缺失与带隙物理强相关的共轭特征"，用 10 个共轭特征+按 `single_bond_frac` 分层 5 折 CV 同时解决"特征缺口"与"分布失配"。

### 模型架构
（A）**GINWithFeatures**（`train_ensemble_v2.py`，GIN 5 层 hidden 256，每层 GINConv+BN，global_mean_pool，148 维 RDKit 描述符经 feature_proj 投 256 维与图嵌入 concat，2 层 MLP hidden 128 Dropout 0.2）。AdamW(lr 1e-4, wd 1e-5) MSELoss。（B）**GIN+共轭特征**（`train_conjugation_features.py`，同样 GIN 5 层/hidden 256/均值池化）：特征 = 148 维通用 + 10 维共轭 = 158 维；10 个共轭特征中 9 个与 Egc 的 |r|>0.4，最强为 `conjugated_bond_fraction`（r=−0.747）、`sp2_sp2_bond_fraction`（−0.747）、`pi_electron_density`（−0.743）、`aromatic_atom_fraction`（−0.736）。输出头 3 层 MLP，Dropout 0.1，**HuberLoss(delta 0.5)**（换 MSE 以抑制 Fold3 离群梯度）。

### 训练评估方式
（A）5 折 KFold（`n_splits=5, shuffle=True, random_state=42`，非分层，与 Day3 同 split 公平对比）× 5 seed `[42,123,456,789,1024]` → 每性质 25 模型，Tm+Egc 共 50 模型；max 500 ep 早停 50；集成预测=每折 5 seed 均值，UQ=5 seed 预测标准差（ensemble disagreement）；算 high-unc（top20%）vs low-unc（bottom80%）RMSE 比及 uncertainty-|error| Pearson。Tm 约 4h、Egc 约 2h。（B）单训练 seed × 5 折 `StratifiedKFold(random_state=42)`，分层依据 `single_bond_frac` 5 分位箱（95.6% 分子落入 bin0，分层受限但仍使各折共轭分布更均衡）；max 500 ep 早停 50（实际每折 180-266 ep 收敛）；Egc 3380 全部成功处理；约 37 min。

### 结果
（A）**Tm 成功、Egc 失败**：Tm（目标 <70，**通过**）5 折 ensemble RMSE [60.79/59.69/58.24/59.56/57.64]，均值 **59.18±1.12**，best fold5=57.64；相对 Day3 单种子（mean 86.60, std 53.05, 含 fold4=192.56 离群）：mean RMSE 降 31.7%、**跨折 std 降 97.9%**（53.05→1.12），那个 192.56 的 fold4 被 ensemble 稳到 59.56。Egc（目标 <0.9，**失败**）5 折 [0.73/0.87/**2.11**/1.00/0.91]，均值 **1.12±0.50**，Fold3 灾难性 RMSE 2.11/R²=−0.72；seed42 在 Egc 上不稳定（跨折 1.90±0.90）；Fold3 从 Day3 的 2.25 仅降到 2.11——ensemble 无法根治。UQ：uncertainty-|error| 相关 Tm r=0.25-0.34、Egc r=0.32-0.96（Fold3 高达 0.96，Egc 的 UQ 信号反而比 Tm 更强）。（B）**共轭特征+分层 CV Egc 成功、Fold3 解决**（`experiment_results.json`）：5 折 [0.4992/0.5132/**0.6145**/0.5271/0.4772]，均值 **0.5262±0.0471**，MAE 0.3371±0.0161，R²0.8854±0.0232，目标 <0.9 超出 41.5%，5 折 R²∈[0.841,0.904] 无负 R²；Fold3 RMSE 0.6145/R²0.841（vs 基线 2.25/R²=−0.96，改善 **72.7%**）；全体 mean RMSE 改善 55.8%（1.19→0.5262），跨折 std 从 0.59 降到 0.0471（降 92%）。三假设 H1（特征缺口）/H2（分层平衡）/H3（RMSE<0.9）全部成立。

### 流程
（A）`train_ensemble_v2.py`（2026-03-25）：加载 Tm/Egc → pSMILES 解析+148 维描述符 → PyG Data → 每性质 KFold 每折跑 5 seed（set_seed 设 torch/cuda/np 三处）训练 GINWithFeatures early stop 存 checkpoint → 每折 5 seed 预测均值得 ensemble、std 得 UQ → 逐样本算 |error|-uncertainty Pearson + 高低不确定性分组 RMSE 比 → `results.json`/`FINAL_REPORT.md` 判 Tm 通过/Egc 失败，建议"排查 Egc Fold3"直接触发（B）。（B）`train_conjugation_features.py`（2026-03-25 01:41-02:18）：`compute_conjugation_features` 算 10 共轭特征拼 158 维 → `create_stratified_folds` 按 single_bond_frac 5 分位箱 StratifiedKFold → 先做 Fold3 诊断（确认 Fold3 验证集 aromatic_ring_count/conjugation_length/pi_electron_count 均低于训练集）→ 逐折训练（Huber 损失）→ 输出 experiment_results.json/predictions.csv/feature_importance.csv/可视化。**两处归属勘误（代码核实）**：(1) Day6 `6_day6_cfd_mtl/FINAL_REPORT.md` 表头把 "59.18±1.12"（Tm）与 "0.5262±0.0471"（Egc）误标为 **"Literature Baseline"**，实为 **Day4 本 run 内部基线**；(2) 本子实验（B）`EXPERIMENT_SUMMARY.md` 标的 "Baseline (Day4)" 实为 **Day3 单种子**（1.19±0.59, Fold3 2.25），72.7% Fold3 改善相对的是 Day3 单种子而非 Day4 ensemble。

---

## Day 5：FD-MTL 特征解耦多任务 — 代码显示 Tm/Egc 达标，AI 自述"灾难失败"无日志支持

### 实验目的
设计 **FD-MTL（Feature-Decomposed Multi-Task Learning，特征解耦多任务学习）**：在 Tg/Tm/Egc 三任务联合训练中，把 148 维分子特征按"与各性质的相关性"解耦为 shared（与≥2 性质相关）+ property-specific（仅与单性质相关）两组，给每个任务头只喂它该用的那部分特征，以缓解多任务负迁移。消融目标：验证解耦是否优于朴素 MTL。注意：**FD-MTL 被设计为"验证性"实验，并非"灾难性失败"实验——后者是 AI 当夜自述的叙事，不被代码/训练日志支持（见结果）。**

### 模型架构
当前 `main.py` 实际实现：共享 GNN 编码器 `FD_MTL`/`GINWithFeatures` 均用 `torch_geometric.nn.GINConv`，5 层 hidden 256，每层 MLP（Linear→BN→ReLU→Linear→BN→ReLU），`global_mean_pool`。特征路径 `compute_148_features(mol)`（函数名标榜"148 维已验证特征"，**实际内容是 RDKit 描述符**：MolWt/HeavyAtomCount/RingCount/FractionCSP3/BertzCT/BalabanJ/Kappa1-3/Chi0v/FpDensityMorgan1-3/qed + 原子计数/键型分数/环大小分布/12 类官能团 SMARTS 计数）——即 **"用 RDKit 描述符冒充已验证 148 维 PF-CL 特征"这一实现错误属实**（SUMMARY.md 自述 Source="PolymerFeatureComputer from PF-CL"，与代码不符）。解耦头每性质 BatchNorm1d+Linear→hidden 与 GIN 池化向量拼接，过 `Linear(2H)→BN→ReLU→Dropout0.2→Linear(H)→BN→ReLU→Dropout0.2→Linear(→1)`。**注**：AI 自述"用 feed-forward 而非 GIN"代码层面**不成立**——当前 `main.py` 用的是 GINConv，可能描述已被覆盖的更早版本。

### 训练评估方式
Tg 7170（处理后 7145）/Tm 3636/Egc 3374，p-SMILES `*`→`[H]` RDKit 解析，PyG `from_smiles` 建图。目标与特征均 StandardScaler（每 fold 用 train fit）。单任务 CV 真 5 折 KFold(shuffle, random_state=42)，每 fold 内再切 88.9/11.1 train/val。**MTL CV 实现缺陷**：`run_mtl_cv` 在 fold 循环内**每轮都 `np.random.seed(42)` 再 `np.random.shuffle`**，致 5 个"fold"用**完全相同的 80/11.1/8.9 切分**——名义 5 折实为同一切分上 5 次重复（不同 RMSE 仅来自随机初始化与 Dropout），这是 CV 实现错误。AdamW(lr 1e-4, wd 1e-5) MSELoss batch 64 max 500 ep 早停 50（**AI 自述"150 ep/patience 30/uncertainty-weighted loss"代码不符——代码是 plain per-task MSE，无不确定性加权**）。三档消融：Single-Task / MTL-All / MTL-Decomp（**AI 自述"4 组含 MTL-Shared"代码未验证**）。硬编码基线 Tg 32.31/Tm 59.18±1.12/Egc 0.5262±0.0471。

### 结果
**两组互斥数字并存，以代码/日志为权威**：

**(A) 代码/训练日志实际数字**（来源 `training_output.log` 2026-03-25 04:57-08:14 实跑 stdout，逐 fold 与 `results.json` 一致）——**权威**：
- Single-Task：Tg 188.37±286.34（fold4=760.82 失控）/Tm 89.69±59.64（fold4=208.95 失控）/Egc 1.0135±0.3473（单任务因某 fold 崩溃致均值极高）
- **MTL-All：Tg 39.50±2.02 / Tm 59.10±1.14 / Egc 0.5138±0.0105**
- **MTL-Decomp：Tg 40.10±1.54 / Tm 58.92±1.22 / Egc 0.5190±0.0124**
- vs 基线 32.31/59.18/0.5262：Tg +22% 较差、Tm −0.44% 略好、Egc −2.36% 略好；**Tm/Egc 实际已达标**（<55/<0.50 目标线 Tm 达成、Egc 达成）
- 特征相关性 **≠ 全部 <0.05**（`feature_correlations.txt`，阈值 0.3，命名特征）：Tg top1 `rotatable_bond_frac r=−0.7280`、`RingCount r=0.7082`；Egc top1 `single_bond_frac r=0.7695`、`aromatic_bond_frac r=−0.7365`。51 shared/24 Tg-specific/5 Egc-specific

**(B) AI 自述的"灾难性失败"**（来源 `SUMMARY.md`/`model_comparison.txt`/`transfer_analysis.txt` + mem_137；旧 summary 沿用此叙事）——**无任何训练日志支撑**：Single-Task Tg 117.07±0.94/Tm 118.22±2.18/Egc 1.68±0.041，MTL-All Tg 180.82/Tm 227.76/Egc 4.7536，MTL-Decomp 180.82/227.74/4.7551，自述"全部 154 特征 r<0.05"。**这组数字在 `training_output.log`/`results.json`/`results.txt`/`feature_correlations.txt` 中均找不到对应实跑记录**，标（AI 当夜自述，代码未独立验证）。

**判定**：旧 summary"Day5 灾难性失败 117.07/118.22/1.68、r<0.05"仅在 AI 自述文档成立，不被代码与训练日志支持；代码权威显示该天 MTL 结果接近基线、Tm/Egc 达标，非全盘崩溃。客观实现缺陷三项：(1) `compute_148_features` 用 RDKit 描述符冒充 PF-CL 148 维特征；(2) `run_mtl_cv` CV seed 重置 bug（5 折退化为同一切分重复）；(3) SUMMARY.md 大量参数与代码不符。

### 流程
`main()` 加载 Tg/Tm/Egc → process_psmiles → RDKit 解析 → from_smiles 建图 → compute_148_features 算 148 维 RDKit 描述符挂到 data.features → `analyze_feature_correlations` 对 148 维×3 性质算 Pearson r（阈值 0.3）分 shared/property_specific → `run_ablation_studies` 跑 3 组消融（Single-Task 真 5 折 / MTL-All / MTL-Decomp 同 seed bug）→ `generate_report` 写 results.txt，dump results.json。训练用 GPU（`Using device: cuda`），单次完整约 3h。**产物不一致**：SUMMARY/model_comparison/transfer_analysis/feature_analysis 四份"灾难"叙述文档与 main.py/training_output.log/results.json/results.txt/feature_correlations.txt 五份实跑文档分别描述两套互斥结果，`training_log.txt` 为 0 字节空文件；两组文件归档 mtime 相同无法据 mtime 判先后，但 training_output.log 的内部运行时间戳（2026-03-25 04:57-08:14）与 results.json 逐 fold 数据是实跑硬证据。**Day5 的真实产出（代码权威）成为 Day6 CFD-MTL"修正"的出发点——但 Day6 修正的"Day5 失败"叙事本身是 AI 自述，代码层面 Day5 Tm/Egc 并未失败。**

---

## Day 6：CFD-MTL 修正特征解耦多任务 — 正迁移成立（Tm−9.8%/Egc−19.9%），非对称迁移假设被否定

### 实验目的
**CFD-MTL（Corrected Feature-Decomposed Multi-Task Learning，修正特征解耦多任务学习）**：纠正 Day5 FD-MTL 的失败。**定位 Day5 失败根因为"用错了特征"**——`compute_148_features()` 算出的 RDKit 描述符与目标相关性极弱（AI 自述 r<0.05，但 Day5 代码权威显示实际相关性并不弱，见 Day5）。Day6 改用经验证的强相关特征（rotatable_bond_frac 对 Tg r=−0.73；conjugated_bond_fraction 对 Egc r=−0.747；sp2_frac/sp3_frac 杂化分数），按"共享-任务专属"解耦。验证三假设：H1 Day5 是实现错误而非方法错误；H2 特征解耦能带来正迁移（多任务优于单任务）；H3 迁移是非对称的（Tg↔Tm 经刚性特征有效，Tg↔Egc 有限）。按计划读了 PointGAT（3D 几何+图注意力）与 MBNN-att（按性质 intensive/extensive 区分注意力）两篇论文（AI 当夜自述）。

### 模型架构
代码实际存在**两个不同配置**的脚本（FINAL_REPORT.md 只描述其一）：(1) **cfd_mtl.py**（主版，56KB）：GINEncoder **5 层 GIN hidden 256** 全局均值池化，用 **148 维特征**，解耦 **shared 46/Tg 14/Tm 0/Egc 4**（阈值 0.3，`feature_correlations.txt`）；(2) **run_cfd_mtl.py**（优化版，29KB）：GINEncoder **3 层 GIN hidden 128**，用 **80 维特征**（代码强制 padding/截断到 80），解耦 **shared 27/Tg 2/Tm 0/Egc 1**（`feature_indices.txt`）。FINAL_REPORT.md 对外声称的"3 层 GIN/128 hidden/80 维特征/27-2-0-1 解耦"即指此版。特征类别覆盖刚性（rotatable_bond_frac/single_bond_frac/FractionCSP3）、共轭（conjugated_bond_fraction/sp2_sp2_bond_fraction/aromatic_bond_frac）、芳香/环（NumAromaticRings/ring6）、杂化分数（sp2_frac/sp3_frac），由 RDKit 在每个 RU 上计算。

### 训练评估方式
Tg 7145/Tm 3636/Egc 3374。5-fold KFold（shuffle, random_state=42），每 fold 内训练集再切 ~11.1% 验证做早停，三性质各自独立 KFold，多任务每 fold 用 `np.random.seed(42+fold)`。AdamW(lr 1e-4, wd 1e-5) MSELoss 目标与特征均 StandardScaler。cfd_mtl.py patience 50/max 500（single）或 300（baseline check）；run_cfd_mtl.py 更短 patience。三档消融 Single-Task/Naive MTL/CFD-MTL。报 5-fold mean±std（反标准化回原量纲）。硬编码基线 Tg 32.31（Day2 GRIN）/Tm 59.18±1.12（Day4 ensemble）/Egc 0.5262±0.0471（Day4 共轭）——**注此三个"文献基线"实为 Day4 内部基线（见 Day4 归属勘误）**。

### 结果
**代码实际输出（权威）**：

| 脚本（配置） | Single-Task Tg/Tm/Egc | Naive MTL Tg/Tm/Egc | **CFD-MTL Tg/Tm/Egc** |
|---|---|---|---|
| cfd_mtl.py（148维/5层GIN/h256）| 98.27±95.45*/75.97±15.20/1.012±0.078 | 44.67±9.31/56.55±2.25/0.5624±0.046 | **39.42±1.44/56.16±4.00/0.5265±0.0334** |
| run_cfd_mtl.py（80维/3层GIN/h128）| 46.46±2.17/69.28±4.32/0.6828±0.059 | 45.72±1.81/61.44±3.37/0.5674±0.009 | **45.56±0.42/62.51±4.62/0.5472±0.0158** |

\* cfd_mtl.py Single-Task Tg 有 1 fold 发散到 286.74 致 mean=98.27 不可信（另 4 fold 在 39-80）。

**FINAL_REPORT.md/mem_146/147 报告的 CFD-MTL Tg 44.12±1.07/Tm 59.93±2.71/Egc 0.5521±0.0406 在所有代码输出文件中检索不到**（全目录 grep 44.12/59.93/0.5521 仅命中 FINAL_REPORT.md 自身），标（AI 当夜自述，代码未独立验证）；旧 summary"差 36%"是按未验证的 44.12 算出（44.12/32.31−1=36.6%），代码两套实测分别是 +22%（cfd_mtl）/+41%（run_cfd_mtl）。**相对基线（代码实测 cfd_mtl CFD-MTL）**：Tg +21.9%（39.42 vs 32.31，仍偏弱）、Tm **−5.1%（56.16 vs 59.18，优于基线）**、Egc **+0.05%（0.5265 vs 0.5262，基本持平）**。**正迁移幅度（run_cfd_mtl.py single→CFD-MTL，single 无发散可信）**：Tg −1.9%/Tm −9.8%/Egc −19.9%（FINAL_REPORT/mem 声称 Tg 3%/Tm 9%/Egc 17% 为 AI 当夜自述，与实测同量级但具体值不符）。**非对称迁移假设 H3 被否定**：三性质都从多任务受益（不只 Tg↔Tm），Egc 提升反而最大，"Tg↔Tm 有效、Tg↔Egc 有限"的预设不成立（FINAL_REPORT.md 标"⚠️ Partial"，mem_147 明写"rejected"；注意这是结果解读，代码未单独设计非对称对照实验）。

### 流程
读 PointGAT/MBNN-att（AI 自述）→ 定位 Day5"失败"根因（AI 自述 r<0.05）→ 改用 `compute_validated_features()` 保留 r>0.5 强相关特征（刚性/共轭/杂化三族）→ 实现并跑 cfd_mtl.py（148 维，5 层 GIN，46/14/0/4 解耦）产 results.json/feature_correlations.txt → 另跑轻量 run_cfd_mtl.py（80 维，3 层 GIN，27/2/0/1 解耦，约 74 min）产 training_output.log/feature_indices.txt → 三档消融×5-fold 与硬编码基线对比 → 结论（代码支持部分）：CFD-MTL 在 Tm/Egc 接近或超基线（cfd_mtl：Tm 56.16 优于 59.18、Egc 0.5265≈0.5262）；Tg 仍明显落后 32.31（最好 39.42，差 ~22%）；多任务相对单任务三性质均提升，正迁移成立；非对称迁移假设被否定。

---

## Day 7：DCN-GIN 深度扩展（稠密连接让深层可训但不追平 32.31）/ TCP terminal 负迁移（35.49±1.60）

### 实验目的
（A）**DCN-GIN（Densely Connected GIN，稠密连接图同构网络）深度扩展**：本 run 已在 Day2 得到 GRIN 5 层 GIN 的 Tg 32.31 K（run 全程最优，非本 day 候选）。Day7 想验证两个纠缠假设：(1) 加深 GIN（至 10、15 层）能否进一步降误差；(2) 标准 GINConv 在深层会因梯度消失/特征坍塌退化，DenseNet 式稠密连接（每层输入=前序所有层输出 concat）能否让深层 GNN 可训。隔离"深度"与"稠密连接"两变量。（B）**TCP（Thermodynamic Consistency Pre-training，热力学一致性预训练）**：本 run 的 **terminal concept-leaping 实验**，也是 day7 尾部 4 个 terminal 候选中 Tg 最优。设计动机直接来自 Day3 PF-CL 失败的三教训：(i) 预训练与下游分布失配；(ii) 对比任务与 Tg 回归冗余；(iii) 缺物理监督。TCP 对症：用 Bicerano 基团贡献法生成 PI1M 的 Tg/Tm/Egc 物理伪标签 → 分布感知采样对齐下游分布 → 多性质预测头+热力学一致性损失注入物理约束。**定位**：TCP 是 4 候选中 Tg 最优（35.49 K），但**非 run 全局最优**——全局最优仍是 Day2 GRIN 32.31 K。

### 模型架构
（A）**DCN-GIN**（`7_day7_dcn_depth_scaling/model.py`）：`DenseGINConv`（MessagePassing，MLP 聚合+边特征 `edge_encoder` 投影）堆叠成 `DenseBlock`，每 4 层插 `TransitionLayer` 做通道压缩；`DCNGIN` 主体为多 dense block+`global_mean_pool`+与 148 维分子描述符拼接。实例化参数 atom_encoder `Linear(9→256)`、feature_proj `Linear(148→128)`、growth_rate 32、hidden 256、dropout 0.2、use_dense=True。**Standard-GIN 对照**同输入管线但 use_dense=False（普通 GINConv 顺序堆叠，无跨层 concat）。深度三档 5/10/15 层，共 5 配置。（B）**TCP Phase3 预训练 TCPModel**（`phase3_pretraining.py:34`）：5 层 GINConv hidden 256，每层后 BN+ReLU+Dropout 0.1，`global_add_pool` 读出，3 个预测头（Tg/Tm/Egc 各 256→128→1）。节点 **153 维**、边 **11 维**（`tcp_experiment_final.log` "atom=153, bond=11"）——但 `forward()` 只消费 `batch.x, batch.edge_index, batch.batch`，**不消费 edge_attr**（`phase3_pretraining.py:91`），即 11 维边特征实际未被 GINConv 使用。Phase4 微调 TCPFinetuneModel 复用预训练 encoder 全部权重，丢弃 3 头换单 Tg 头（256→128→1）。BaselineModel 架构与 TCPFinetuneModel 完全一致但从随机初始化训练（无预训练）。

### 训练评估方式
（A）Tg 7145，Day6 cache 复用（148 维描述符+9 维原子特征），5-fold CV（`seed=42+fold`），batch 64 Adam lr 1e-3 wd 1e-5 ReduceLROnPlateau(factor 0.5 patience 20) max 300 ep 早停 50，目标标准化。产 results.csv/summary.txt/feature_reuse.json/gradient_analysis.json。（B）**Phase1** Bicerano 基团贡献法伪标签（`phase1_pseudo_labels.py`）：在 PI1M（995,799）上用分子量 MW/可旋转键/HBD/HBA/芳·饱和环数/TPSA/FractionCSP3/连接性指数 Chi0v/Chi1n 等结构特征算 Tg_pseudo/Tm_pseudo（≈1.5·Tg）/Egc_pseudo。**Phase2** 分布感知采样（`phase2_sampling.py`）：先在 Tg 标注集（7170，train 6453/val 717）训 SimpleGIN（3 层 hidden 128）100 ep，best val RMSE **31.93 K @ ep95**；再用 MC-dropout 认知不确定性（`compute_epistemic_uncertainty`，n_ensemble=5 次前向开 dropout）在 988,775 条有效 PI1M 上打分，`select_low_uncertain_samples` 选不确定性最低 **100,000** 条（`np.argsort` 后取前 100K）做预训练集。**Phase3** TCP 预训练 batch 128 Adam lr 1e-3 ReduceLROnPlateau(factor 0.5 patience 5) max 50 ep+早停 10，损失 `L=L_pred+0.1·L_consistency`，L_pred=MSE(Tg)+MSE(Tm)+MSE(Egc)，三项一致性惩罚：`Tg<Tm`（`relu(Tg−Tm)×10`）、`Tg/Tm≈0.65`（容差 0.2，`relu(|Tg/Tm−0.65|−0.2)×5`）、`Egc≈Tg/10+30`（容差 20，`relu(|Egc−(Tg/10+30)|−20)×0.5`），权重 λ=0.1。**Phase4** 5-fold 微调（`create_k_fold_splits(n_splits=5, seed=42)`）batch 64(train)/256(val) Adam lr 5e-4 ReduceLROnPlateau(factor 0.5 patience 10) max 300 ep 早停 30，TCP 与 Baseline 完全相同 split/超参。**Phase5** 消融 Ablation1 GRIN（节点 9+描述符 148）/Ablation2 TCP（节点 153），均 5 层 GIN hidden 256 batch 64 lr 5e-4 max 300 ep 早停 40 readout=mean。

### 结果
（A）**DCN-GIN 深度扩展**（`results.csv`+`summary.txt` 核实）：DCN-5 **40.26±1.42** / DCN-10 **38.59±1.34** / DCN-15 **38.40±1.18**（DCN 最优）；Standard-10 **44.16±1.28** / Standard-15 **47.64±3.83**（方差飙升）。**稠密连接确实让深层可训**——深度 10 处 DCN 比 Standard 好 12.6%，深度 15 处好 19.4%，Standard-15 方差 3.83 表明已训练不稳定；**但深度本身不追平基线**——DCN 从 5→15 仅微降 1.86 K，最优 DCN-15 仍比 Day2 GRIN 32.31 差 +6.09 K（+18.8%）。feature_reuse.json/gradient_analysis.json 显示所有层都有非零贡献与梯度（呈 1/(l+1) 衰减），稠密连接缓解梯度消失但未改变精度上限。（B）**TCP terminal**：Phase1 伪标签（`tcp_experiment_log.txt:7069+` 权威）—— 处理 995,799→有效 **988,775**（99.3%），Tg_pseudo **535.70±78.29 K**、Tm_pseudo 743.72±97.70、Egc_pseudo 91.81±18.63，Tm>Tg 比例 **100.0%**（⚠️ `tcp_final_summary.md` 为 stale 文件，写"Tg 308.45±52.87、一致性 98.7%"与权威不符，弃用）。Phase2 不确定性模型 best val 31.93 K@ep95。Phase3 预训练 best val loss 948.65（对伪标签 Tg RMSE 16.66），耗时 19.4 min（v4 曾撞 `IndexError: too many indices for tensor of dimension 1`，`run_tcp_fixed.py` 修复后跑通）。Phase4 5-fold 微调（`Tg_finetuned_results.txt`/`comparison_with_baseline.txt`）：**TCP fine-tuned Tg 35.49±1.60**（fold 32.92/35.77/36.15/37.79/34.82）vs **scratch baseline 35.22±1.55**（fold 33.14/34.96/35.50/37.89/34.58）→ 差值 **−0.28 K（−0.78%），判定 NEGATIVE TRANSFER（误差棒内不显著）**。Phase6 TCP 模型预测热力学一致性（`thermodynamic_consistency.txt`）：Tg<Tm 100%、Tg/Tm=0.720±0.053、Egc-Tg 相关 0.847（模型记住了伪标签先验但未泛化到真 Tg）。Phase5 消融（`ablation_results.json`）：GRIN 148 维 **34.91±1.82**（vs 32.31 差 +2.60，z=1.43<2 落统计方差内，归因预处理/随机性/PyG 非确定性实现差异）/ ReperioGT 153 维 **35.35±1.48**（差 +3.04），两者差 0.44（z=0.26 不显著）→ **特征维度（148 vs 153）不是瓶颈**。vs Day2 内部基线 32.31 → TCP +3.18 K（`comparison_with_baseline.txt` 明文）。**注**：任务书提及"SOTA 30.8 → 差 4.69 K"，30.8 K 在本 day7/day8 archive 文件中**无明确出处**，4.69 K 是外部计算（35.49−30.8），仅供线索。

### 流程
（A）`run_complete_experiment.py`：加载 Day6 cached 148 维特征 → 构图（9 维节点）→ 对 DCN 与 Standard 在深度 5/10/15 各跑 5-fold → 写 results.csv/summary.txt+两份 JSON 分析（quick_experiment.py/simple_experiment.py 为早期调试脚本，前者因 cache tuple 解包报错弃用）。（B）6 阶段流水（`run_phases.sh`/`run_phases_2_6.py`/最终 `run_tcp_fixed.py`）：phase1→`PI1M_pseudo_labels.pt`(83MB) → phase2→`uncertainty_model.pt`+`selected_indices.pt`(100K) → phase3→`TCP_pretrained_model.pt`(4.9MB) → phase4→`Tg_finetuned_results.txt`/`comparison_with_baseline.txt` → phase5（final run 中因 ImportError 跳过，事后 `run_ablation.py` 单独补跑）→ `ablation_results.json/txt`+`ABLATION_DIAGNOSTIC_REPORT.md` → phase6→`thermodynamic_consistency.txt`/`FINAL_REPORT.md`。总耗时约 66.5 min（Phase3 19.4+Phase4 46.6+Phase6 0.4，不含 Phase1/2 与消融补跑）。`tcp_experiment_log.txt` 末尾 "Phase 1 failed. Stopping execution" 是 wrapper 串联阶段的控制流伪信号，Phase1 实际成功并产出 `PI1M_pseudo_labels.pt`。**Day7 两子实验共同结论：无论加深（DCN-15 38.40）还是物理预训练（TCP 35.49 负迁移），都未能追平 Day2 GRIN 32.31——本 run 的 Tg 天花板早在 Day2 已定。**

---

## 7 天轨迹总结

ab3 是一次 GLM-5 full day-night 的多性质探索 run，核心张力是**"Day2 早早达到 Tg 32.31 天花板后，后续 5 天用尽各种手段都无法超越"**。轨迹分四段：

**第一段（D1 探路）**：HEPT 把等变+周期+长程三套重机制堆叠，因周期镜像被注释绕过、增强特征未接入、参数膨胀过拟合，49.58/53.84 双双失败；RE-PerioGT 退回轻量物理先验（可旋转键占比刚性特征），36.41→35.56 微胜，但刚性驱动对比学习无增益（验证集反退、训练时间 10 倍化）。教训：归纳偏置须匹配物理，周期性是主偏置，3D 等变/重周期镜像/对比学习非 Tg 所需。

**第二段（D2 突破/全局最优）**：PerioGT 消融澄清了"30.8 vs 37.35"实为两篇不同论文（2022 无预训练 37.35 / 2025 需 PI1M 预训练 30.8），本 run GraphFormer 37.00 超 2022 版；GRIN 检验发现 baseline（mean 聚合+148 维富特征）**32.31 达标**，但 max+L1 假设被拒（full_grin 46.34 最差 +43.4%）——**32.31 成为全 run Tg 全局最优，此后再未被超越**。148 维特征贡献 57.6%（多项改动合并，非纯特征消融）。

**第三段（D3-D4 多性质扩展，成功）**：Day3 PF-CL 联合预训练（对比+154 维特征预测）Tg 39.23±1.81 负迁移（任务冲突+PI1M↔Tg 分布失配），Tm/Egc 特征 baseline 方差极大（Tm 86.60±53.05/Egc 1.19±0.59，单折崩溃+单种子）；Day4 多性质成功——multi-seed ensemble UQ 把 Tm 稳到 **59.18±1.12**（跨折 std 降 97.9%），共轭特征+分层 CV 把 Egc 降到 **0.5262±0.0471**（Fold3 从 2.25 解决到 0.6145，改善 72.7%）。多性质 Tm/Egc 达到 run 内最优。

**第四段（D5-D7 多任务与 terminal，未超越 32.31）**：Day5 FD-MTL 多任务——**AI 自述"灾难失败 117/118/1.68"无训练日志支持**，代码权威显示 MTL-All/Decomp 实际 Tm 59.10/58.92、Egc 0.5138/0.5190（均达标），仅 Tg 39.50/40.10 偏弱（+22%），客观缺陷是 CV seed 重置 bug + 特征冒名；Day6 CFD-MTL 修正版（cfd_mtl 主版 CFD-MTL Tg 39.42/Tm 56.16/Egc 0.5265）正迁移成立（single→MTL：Tm −9.8%/Egc −19.9%），非对称迁移假设 H3 被否定；Day7 DCN-GIN 深度扩展证明稠密连接让深层可训（DCN-15 38.40 vs Standard-15 47.64）但深度本身不追平 32.31（+18.8%），TCP terminal 用 Bicerano 伪标签+分布感知采样+热力学一致性损失做物理预训练，fine-tuned **35.49±1.60 vs scratch 35.22±1.55 = 负迁移（−0.28 K，不显著）**。

**全 run 结论**：Tg 天花板 Day2 GRIN 32.31，terminal TCP 35.49 是 4 候选最优但负迁移、距 SOTA 30.8 仍差 4.69；多性质 Tm 59.18±1.12、Egc 0.5262±0.0471 为 run 内最优。**方法论教训**：(1) 单种子单划分的 32.31 缺乏多种子稳健性估计（GRIN 目录自述此局限）；(2) PF-CL/TCP 预训练均因 PI1M↔Tg 分布失配负迁移；(3) FD-MTL 的"特征解耦"对 Tm/Egc 有效（正迁移）但对 Tg 无益；(4) 本 run AI 当夜自述在 Day5/Day6 大规模虚报（Day5 虚报灾难失败、Day6 虚报 44.12/59.93/0.5521），**代码权威核实纠正了这些叙事**。

---

## 参考文件（均在 `archive/ablation3archive/`）

- 旧 summary：`projects_summary_discard.md`（2026-03-26，原始）/ `projects_summary_discard2.md`（2026-07-14，上一轮，含本轮已纠正的 Day5 灾难失败叙事、Day6 44.12/59.93/0.5521 等 AI 自述数字）。
- Day1：`src/1_hept/{model.py,data_loader.py,results.json,results_v2.json}`、`src/1_day2_reperioGT/{model.py,dataset.py,final_results.json}`。
- Day2：`src/2_02_PerioGT_Ablation/{VERIFICATION_REPORT.md,comprehensive_results.csv,best_config.json,cv_results.csv}`、`src/2_day2_grin/{models.py,data_utils_v2.py,train_v2.py,results_v2_20260324_090902/summary.csv,baseline/results.json,FINAL_REPORT_ITERATION2.md,FINAL_SUMMARY.md}`。
- Day3：`src/3_pfcl_pretraining/{model.py,pretrain.py,periodic_graph.py,FINAL_5FOLD_REPORT.md,PROGRESS_UPDATE.md}`、`src/3_day4_feature_baseline_tm_egc/{train_baseline_v2.py,tm_baseline_results_v2.json,egc_baseline_results_v2.json,cross_property_comparison_v2.json}`。
- Day4：`src/4_day4_ensemble_uq/{train_ensemble_v2.py,results.json,tm_results.json,FINAL_REPORT.md}`、`src/4_egc_conjugation/{train_conjugation_features.py,results/experiment_results.json,EXPERIMENT_SUMMARY.md,experiment_log.txt}`。
- Day5：`src/5_day5_fd_mtl/{main.py,training_output.log,results.json,results.txt,feature_correlations.txt}`（权威）/ `{SUMMARY.md,model_comparison.txt,transfer_analysis.txt,feature_analysis.txt}`（AI 自述，代码未验证）。
- Day6：`src/6_day6_cfd_mtl/{cfd_mtl.py,run_cfd_mtl.py,results.json,results.txt,training_output.log,feature_indices.txt,feature_correlations.txt}`（权威）/ `FINAL_REPORT.md`（AI 自述，核心数字代码未验证）。
- Day7：`src/7_day7_dcn_depth_scaling/{model.py,run_complete_experiment.py,results.csv,summary.txt,feature_reuse.json,gradient_analysis.json}`、`src/7_day8_tcp/{phase1_pseudo_labels.py,phase2_sampling.py,phase3_pretraining.py,phase4_finetuning.py,run_ablation.py,tcp_experiment_log.txt,tcp_experiment_final.log,Tg_finetuned_results.txt,comparison_with_baseline.txt,thermodynamic_consistency.txt,ablation_results.json,FINAL_REPORT.md,ABLATION_DIAGNOSTIC_REPORT.md}`（`tcp_final_summary.md` 为 stale，弃用）。
- 数据：`src/alldataset/{Tg.csv,Tm.csv,Egc.csv}`、`src/baseline/`。mem.json 171 节点（99 knowledge+30 idea+12 experience+30 integration）。
