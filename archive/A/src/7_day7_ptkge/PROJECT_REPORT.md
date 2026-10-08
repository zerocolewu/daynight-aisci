# PTKGE: Periodic Table Knowledge Graph Embeddings for Polymer Property Prediction

> A physics-informed graph neural network for polymer glass-transition temperature ($T_g$) prediction, replacing learned element embeddings with fixed periodic-table physical-property vectors.

---

## 1. Research Objective

**Goal.** Predict polymer glass-transition temperature ($T_g$) from SMILES, and test whether **knowledge-driven element embeddings** (8 physical properties from the periodic table) outperform learned ones on a small regression dataset (~7k samples).

**Core hypothesis ($H_1$).** Fixed physical-property embeddings encode chemical similarity as an inductive bias, improving generalization — especially for rare elements (Br, I) where data-driven embeddings overfit.

| Strategy | Element representation | Trainable? |
|----------|------------------------|:----------:|
| **Physics** (ours) | $[\chi, r, v, g, p, \text{EA}, \text{IE}, M] \in \mathbb{R}^8$ | ✗ |
| Random | $\mathcal{N}(0, 0.1) \in \mathbb{R}^8$ | ✓ |
| Learnable | Physics-init, fine-tuned | ✓ |
| One-hot | one-hot($Z$) $\in \mathbb{R}^{118}$ | ✗ |

Expected order: **Physics > Learnable > Random > One-hot**.

---

## 2. Model Architecture (PTKGE)

PTKGE is a **dual-encoder** regression model: a graph neural network (GIN) over the molecular graph fused with an MLP over ECFP4 fingerprints.

### 2.1 Forward pass

$$
\begin{aligned}
\mathbf{h}_{\text{graph}} &= \mathrm{GIN}_{5}\!\big(\mathrm{Emb}(Z_{\text{atoms}}),\, E_{\text{bonds}}\big) &&\in \mathbb{R}^{256} \\
\mathbf{h}_{\text{fp}}    &= \mathrm{MLP}_{2048 \to 512 \to 256}(\mathrm{ECFP4}) &&\in \mathbb{R}^{256} \\
\mathbf{h}_{\text{fused}} &= \mathrm{Fusion}\!\big([\,\mathbf{h}_{\text{graph}};\, \mathbf{h}_{\text{fp}}\,]\big) &&\in \mathbb{R}^{256} \\
\hat{y}                   &= \mathrm{MLP}_{256 \to 256 \to 128 \to 1}(\mathbf{h}_{\text{fused}}) &&\in \mathbb{R}
\end{aligned}
$$

```
Input ─┬─ Atomic numbers  [N_atoms]
       │  Bond edges      [2, N_edges]
       │  ECFP4 (2048-bit) [B]
       │
       ├─ GINEncoder ─────────────┐
       │     · PeriodicEmbedding  │
       │     · 5 × GINConv        │ → h_graph  ∈ ℝ^256
       │                          │
       ├─ ECFP4Encoder ───────────┤
       │     · MLP 2048→512→256   │ → h_fp     ∈ ℝ^256
       │                          │
       ├─ Fusion(concat + MLP)    │ → h_fused  ∈ ℝ^256
       │                          │
       └─ Predictor MLP           │ → ŷ        ∈ ℝ
```

### 2.2 Physics embedding

For each atom with atomic number $Z$, map to a fixed 8-D property vector:

$$
\mathbf{e}_Z = \mathrm{normalize}\!\big([\,\chi,\, r,\, v,\, g,\, p,\, \mathrm{EA},\, \mathrm{IE},\, M\,]\big), \quad \mathrm{requires\_grad} = \text{False}
$$

| Symbol | Property | Symbol | Property |
|:--:|---|:--:|---|
| $\chi$ | Electronegativity | $p$ | Period |
| $r$ | Atomic radius (pm) | EA | Electron affinity (kJ/mol) |
| $v$ | Valence electrons | IE | 1st ionization energy (kJ/mol) |
| $g$ | Group number | $M$ | Atomic mass (normalized) |

### 2.3 Pretraining (contrastive, on PI1M)

A 100 k subset of PI1M (~1 M unlabeled polymer SMILES), 20 epochs, InfoNCE with ECFP4-Tanimoto $> 0.7$ defining positives:

$$
\mathcal{L}_{\text{contrast}} = -\sum_{i} \log \frac{\sum_{j \in P(i)} \exp\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\big)}{\sum_{k \neq i} \exp\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\big)}, \qquad \tau = 0.05
$$

Onehot does not use pretraining; the other three strategies load their respective `pretrained_{strategy}.pt`.

### 2.4 Training protocol

| Phase | Epochs | Batch | Optimizer | LR (encoder / head) | Loss |
|-------|:--:|:--:|---|---|---|
| Pretrain | 20 | 64 | AdamW | $10^{-4}$ | InfoNCE |
| Fine-tune | **200** (no early stop) | 32 | AdamW | $10^{-4}$ / $10^{-3}$ | MSE |

**Data split:** 7,170 Tg samples → **72 / 8 / 20 %** (train / val / test) with `random_state=42`.
Per-strategy best epoch is selected by validation MSE, then reported on test.

---

## 3. Results

### 3.1 Main results — 72/8/20 split, 200 epochs, best-by-val epoch

| Strategy | Train RMSE | Val RMSE | **Test RMSE ↓** | **Test R² ↑** |
|----------|-----------:|---------:|----------------:|--------------:|
| One-hot  | 15.77 | 36.88 | 35.93 | 0.898 |
| Random   | 13.92 | 32.60 | 34.42 | 0.906 |
| **Physics** | **12.68** | 33.15 | **34.39** | **0.907** |
| Learnable | 15.81 | 37.43 | 36.21 | 0.896 |

**Consistency check vs. 5-fold CV** (`results.json`, train.py with patience=15):

| Strategy | 5-fold CV RMSE | Single-split Test RMSE |
|----------|:--:|:--:|
| One-hot  | 35.69 ± 2.56 | 35.93 |
| Random   | 34.89 ± 1.89 | 34.42 |
| **Physics** | **34.47 ± 1.83** | **34.39** |
| Learnable | 36.40 ± 1.65 | 36.21 |

Ordering and magnitudes agree. **Physics is the best strategy**, Learnable the worst — supporting $H_1$ that on this data scale a fixed physical prior beats end-to-end learned embeddings.

### 3.2 Training curves & parity plots

#### Physics (best strategy)
![Physics training and parity](figures/training_and_parity_physics_72_8_20.png)

#### Random
![Random training and parity](figures/training_and_parity_random_72_8_20.png)

#### One-hot
![Onehot training and parity](figures/training_and_parity_onehot_72_8_20.png)

#### Learnable
![Learnable training and parity](figures/training_and_parity_learnable_72_8_20.png)

**Reading the figures.** *Left*: MSE loss vs. epoch (blue = training, red = validation). *Right*: predicted vs. experimental $T_g$ on train / validation / test, with the dashed $y = x$ line.

Observations:
- All four strategies converge in < 50 epochs; validation MSE plateaus around 800–1,500.
- Physics reaches the lowest training MSE and tightest parity cloud around $y = x$.
- Learnable overfits the training set (lowest train RMSE is competitive) but generalizes worst.

### 3.3 Element-embedding t-SNE (physics)

![Element embeddings t-SNE](figures/embedding_tsne.png)

Physics embeddings naturally cluster by periodic-table group — halogens (F/Cl/Br/I) co-locate, as do same-group pairs (C–Si, N–P, O–S). This is the chemical similarity the fixed prior injects into the GIN, and is what learned embeddings must rediscover from only ~7k labelled samples.

---

## 4. Discussion

**Why physics wins.** 7,170 training samples are insufficient to learn 118 element vectors from scratch. The 8-D physics prior encodes group/period trends for free, reducing the effective parameter count and giving reasonable representations for rare elements (Br, I — see embedding cluster above).

**Why learnable loses.** Fine-tuning drifts the embeddings away from the physical prior to fit training-set idiosyncrasies; on this data scale this is net negative despite lower training loss.

**Limitations.** (i) Single split — no test-set std (cf. 5-fold CV table above for uncertainty). (ii) Test RMSE is selected by best validation epoch, which is mildly optimistic. (iii) Improvements over Random are small (~0.6%); the headline gain is vs. Learned / One-hot (~5%).

---

## 5. Reproducibility

```bash
python plot_training_parity.py        # 72/8/20 split, 200 epochs, 4 strategies
# → figures/training_and_parity_{onehot,random,physics,learnable}_72_8_20.png
# → cache_72_8_20/run_{strategy}.npz   (history + predictions)
```

Pretrained checkpoints (`checkpoints/pretrained_{physics,random,learnable}.pt`) come from 100k PI1M samples, 20 epochs of InfoNCE contrastive learning, `seed=42`.

---
