"""
Training engine for the CPA-GAT optimization campaign (ORIG_BB recipe).

ORIG_BB is the project's best single configuration: the simple original CPA-GAT
(mean pool, simple predictor, no edge features / no residuals) + the is_backbone
node feature + SWA. The only change from the original run is the learning rate
(1e-3 -> 1e-4). Training randomness (init_seed) is decoupled from the data split
(split_seed); targets are normalized with the per-split train mean/std and the
model predicts in normalized space (metrics in original units). Best-by-val-RMSE
checkpoint is evaluated once on test; the SWA model is reported as a reference.
"""
import os
import sys
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import data_utils as DU
from model import CPAGATv2

RESULTS_DIR = C.RESULTS_DIR


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_scheduler(optimizer, warmup, total, eta_min_ratio=0.001):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


def _metrics(preds, targets):
    preds = np.asarray(preds, dtype=np.float64); targets = np.asarray(targets, dtype=np.float64)
    return dict(rmse=float(np.sqrt(mean_squared_error(targets, preds))),
                mae=float(mean_absolute_error(targets, preds)),
                r2=float(r2_score(targets, preds)))


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            raw = model(batch)
            pred = (raw * y_std + y_mean).cpu().numpy()
            tgt = batch.y.view(-1).cpu().numpy()
            preds.append(pred); targets.append(tgt)
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return _metrics(preds, targets), preds.tolist(), targets.tolist()


def train_one(dataset, split_seed, init_seed, quick=False, overrides=None):
    overrides = overrides or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(init_seed)

    sp = DU.get_split(dataset, split_seed)
    train_g, val_g, test_g = sp["train"], sp["val"], sp["test"]
    y_mean, y_std = sp["y_mean"], sp["y_std"]

    cfg = dict(C.ORIG_BB)
    if quick:
        cfg.update(epochs=3, patience=3, batch_size=32)
    cfg.update(overrides)

    in_dim = train_g[0].x.size(1)                  # 70 (35 atom + 2 PSPE + 32 RWPE + 1 is_backbone)
    u_dim = sp["u_dim"]                            # 0 single, 6 copolymer
    edge_dim = None if not cfg.get("use_edge", False) else \
        (train_g[0].edge_attr.size(1) if hasattr(train_g[0], "edge_attr") and
         train_g[0].edge_attr is not None else None)

    model = CPAGATv2(
        in_dim=in_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        heads=cfg["heads"], dropout=cfg["dropout"], edge_dim=edge_dim, u_dim=u_dim,
        pool=cfg["pool"], use_residual=cfg["use_residual"], use_norm=cfg["use_norm"],
        use_input_bn=cfg["use_input_bn"], predictor_type=cfg["predictor_type"],
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bs = cfg["batch_size"]
    train_loader = DataLoader(train_g, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_g, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_g, batch_size=256, shuffle=False)

    epochs = cfg["epochs"]
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = cosine_warmup_scheduler(optimizer, cfg["warmup"], epochs)
    criterion = nn.MSELoss()

    use_swa = cfg.get("use_swa", True)
    swa_start = int(epochs * cfg.get("swa_start_frac", 0.5))
    swa_n = 0
    swa_state = {k: torch.zeros_like(v.detach().cpu()) for k, v in model.state_dict().items()}

    best_val, best_epoch, best_state, patience_counter = float("inf"), 0, None, 0
    tag = f"{dataset}_sp{sp['split_tag']}_init{init_seed}"
    if quick:
        tag = f"quick_{tag}"   # isolate 3-epoch smoke runs from real results / ensemble
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            raw = model(batch)
            y_norm = (batch.y.view(-1) - y_mean) / y_std
            loss = criterion(raw, y_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_m, _, _ = evaluate(model, val_loader, device, y_mean, y_std)
        if use_swa and epoch >= swa_start:
            swa_n += 1
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if not v.is_floating_point():
                        swa_state[k] = v.detach().cpu().clone(); continue
                    swa_state[k] += (v.cpu() - swa_state[k]) / swa_n

        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"[{tag}] ep{epoch:3d} val_rmse={val_m['rmse']:.3f} best={best_val:.3f}"
                  f"{' swa' if (use_swa and epoch >= swa_start) else ''}", flush=True)
        if patience_counter >= cfg["patience"]:
            print(f"[{tag}] early stop @ ep{epoch}", flush=True)
            break

    # protocol selection: lowest-val-RMSE checkpoint -> test
    model.load_state_dict(best_state); model.to(device)
    test_m, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std)

    # SWA model as an extra reference (not the protocol selection)
    swa_test_rmse = None
    if use_swa and swa_n > 1:
        model.load_state_dict({k: v.clone() for k, v in swa_state.items()}); model.to(device)
        model.train()  # refresh any running stats (LayerNorm has none, harmless)
        with torch.no_grad():
            for batch in train_loader:
                _ = model(batch.to(device))
        swa_m, _, _ = evaluate(model, test_loader, device, y_mean, y_std)
        swa_test_rmse = swa_m["rmse"]

    train_time = time.time() - t0
    result = dict(
        tag=tag, dataset=dataset, kind=sp["kind"],
        split_tag=sp["split_tag"], split_seed=split_seed, init_seed=init_seed,
        counts=dict(train=sp["counts"][0], val=sp["counts"][1], test=sp["counts"][2]),
        y_mean=y_mean, y_std=y_std,
        best_epoch=best_epoch, best_val_rmse=best_val,
        test_rmse=test_m["rmse"], test_mae=test_m["mae"], test_r2=test_m["r2"],
        swa_test_rmse=swa_test_rmse, num_params=num_params, train_time=train_time,
        config={k: v for k, v in cfg.items()}, in_dim=in_dim, u_dim=u_dim,
        test_preds=test_preds, test_targets=test_targets,
    )
    with open(os.path.join(RESULTS_DIR, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] DONE test_rmse={test_m['rmse']:.4f} swa={swa_test_rmse} "
          f"val={best_val:.4f}(ep{best_epoch}) params={num_params} time={train_time:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    _ds = sys.argv[1] if len(sys.argv) > 1 else "tg"
    _sp = None if _ds in C.GIVEN_SPLIT_DATASETS else (int(sys.argv[2]) if len(sys.argv) > 2 else 71)
    _init = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    train_one(_ds, _sp, _init, quick=("--quick" in sys.argv))
