"""
Training engine for the PTKGE optimization campaign.

Protocol per run (split_seed, init_seed are decoupled):
    * build the split via data_utils.get_split (deterministic by split_seed)
    * normalize targets with train mean/std (model predicts in normalized space;
      metrics are computed in original units)
    * single-polymer (tg/tm): PTKGEModel initialized from the pretrained
      checkpoint (strict load), fine-tuned with a differential encoder/head LR
    * copolymer (ea/ip): PTKGECopolymer trained from scratch on the connected
      A-B graph
    * early-stop on validation RMSE (original units); evaluate the best checkpoint
      once on test; store original-unit predictions for ensembling
"""
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config as C
import data_utils as DU
from model import PTKGEModel
from model_copolymer import PTKGECopolymer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(preds, targets):
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return dict(
        rmse=float(np.sqrt(mean_squared_error(targets, preds))),
        mae=float(mean_absolute_error(targets, preds)),
        r2=float(r2_score(targets, preds)),
    )


def evaluate(model, loader, device, y_mean, y_std, forward_fn):
    """Original-unit RMSE/MAE/R2 + per-sample predictions/targets."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            raw = forward_fn(model, batch)                 # normalized
            pred = (raw * y_std + y_mean).cpu().numpy()    # original units
            tgt = batch.y.view(-1).cpu().numpy()
            preds.append(pred); targets.append(tgt)
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    m = _metrics(preds, targets)
    return m, preds.tolist(), targets.tolist()


def train_one(dataset, split_seed, init_seed, quick=False, overrides=None):
    overrides = overrides or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(init_seed)

    sp = DU.get_split(dataset, split_seed)
    kind = sp["kind"]
    train_g, val_g, test_g = sp["train"], sp["val"], sp["test"]
    y_mean, y_std = sp["y_mean"], sp["y_std"]

    md = dict(C.MODEL_DEFAULTS)
    md.update(overrides.get("model", {}))

    if kind == "single":
        model = PTKGEModel(**md)
        # Load pretrained encoder (strict: architecture must match exactly).
        if os.path.exists(C.PRETRAINED_PATH):
            ck = torch.load(C.PRETRAINED_PATH, map_location=device)
            model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck,
                                  strict=True)
            pretrained = True
        else:
            print(f"[WARN] pretrained checkpoint missing at {C.PRETRAINED_PATH}; "
                  f"training {dataset} from scratch", flush=True)
            pretrained = False

        def forward_fn(m, b):
            # PTKGEModel.forward returns [B,1]; flatten to [B] so the MSE target
            # ([B]) does not broadcast into a [B,B] outer product.
            return m(b.z, b.edge_index, b.batch, b.fp).reshape(-1)
    else:
        extras_dim = sp["extras_dim"]
        model = PTKGECopolymer(extras_dim=extras_dim, **md)
        pretrained = False

        def forward_fn(m, b):
            return m(b).reshape(-1)

    model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fcfg = dict(C.FINETUNE_SINGLE if kind == "single" else C.FINETUNE_COPOLYMER)
    train_bs = fcfg["batch_size"]
    if quick:
        fcfg.update(epochs=3, patience=3, batch_size=32)
        train_bs = 32
    fcfg.update(overrides.get("train", {}))

    train_loader = DataLoader(train_g, batch_size=train_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_g, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_g, batch_size=256, shuffle=False)

    # Optimizer: differential LR for the pretrained single-polymer model.
    if kind == "single":
        optimizer = AdamW([
            {"params": model.get_encoder_params(), "lr": fcfg["encoder_lr"]},
            {"params": model.get_predictor_params(), "lr": fcfg["head_lr"]},
        ], weight_decay=fcfg["weight_decay"])
    else:
        optimizer = AdamW(model.parameters(), lr=fcfg["lr"],
                          weight_decay=fcfg["weight_decay"])
    criterion = nn.MSELoss()

    best_val, best_epoch, best_state, patience_counter = float("inf"), 0, None, 0
    tag = f"{dataset}_sp{sp['split_tag']}_init{init_seed}"
    if quick:
        tag = f"quick_{tag}"   # isolate 3-epoch smoke runs from real results / ensemble
    t0 = time.time()

    for epoch in range(1, fcfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            raw = forward_fn(model, batch)
            y_norm = (batch.y.view(-1) - y_mean) / y_std
            loss = criterion(raw, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_m, _, _ = evaluate(model, val_loader, device, y_mean, y_std, forward_fn)
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{tag}] ep{epoch:3d} val_rmse={val_m['rmse']:.3f} "
                  f"best={best_val:.3f}", flush=True)
        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= fcfg["patience"]:
                print(f"[{tag}] early stop @ ep{epoch}", flush=True)
                break

    model.load_state_dict(best_state); model.to(device)
    test_m, test_preds, test_targets = evaluate(model, test_loader, device,
                                                y_mean, y_std, forward_fn)
    train_time = time.time() - t0

    result = dict(
        tag=tag, dataset=dataset, kind=kind,
        split_tag=sp["split_tag"], split_seed=split_seed, init_seed=init_seed,
        counts=dict(train=sp["counts"][0], val=sp["counts"][1], test=sp["counts"][2]),
        y_mean=y_mean, y_std=y_std, pretrained=pretrained,
        best_epoch=best_epoch, best_val_rmse=best_val,
        test_rmse=test_m["rmse"], test_mae=test_m["mae"], test_r2=test_m["r2"],
        num_params=num_params, train_time=train_time,
        model=md, train_cfg={k: v for k, v in fcfg.items()},
        test_preds=test_preds, test_targets=test_targets,
    )
    with open(os.path.join(C.RUN_DIR, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] DONE test_rmse={test_m['rmse']:.4f} val={best_val:.4f}(ep{best_epoch}) "
          f"params={num_params} pretrained={pretrained} time={train_time:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    import sys
    _ds = sys.argv[1] if len(sys.argv) > 1 else "tg"
    _sp_seed = None if _ds in C.GIVEN_SPLIT_DATASETS else int(sys.argv[2]) if len(sys.argv) > 2 else 71
    _init = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    train_one(_ds, _sp_seed, _init, quick=("--quick" in sys.argv))
