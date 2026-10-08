"""
Training loop for standard baselines.
- Adam + ReduceLROnPlateau, MSE loss on standardized targets.
- Early stopping on val RMSE (in original target units), patience=50.
- Best checkpoint (by val RMSE) restored, then evaluated once on test.
- Test RMSE / MAE reported in original (inverse-transformed) units.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GLoader


def _rmse_mae_orig(preds_n, targs_n, scaler):
    preds_o = scaler.inverse_transform(preds_n.reshape(-1, 1)).ravel()
    targs_o = scaler.inverse_transform(targs_n.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(np.mean((preds_o - targs_o) ** 2)))
    mae = float(np.mean(np.abs(preds_o - targs_o)))
    return rmse, mae


def _eval_graph(model, loader, device, scaler):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            preds.append(out.cpu().numpy())
            targs.append(batch.y.cpu().numpy().ravel())
    preds = np.concatenate(preds); targs = np.concatenate(targs)
    return _rmse_mae_orig(preds, targs, scaler)


def _eval_ecfp(model, loader, device, scaler):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).squeeze(-1)
            preds.append(out.cpu().numpy())
            targs.append(yb.numpy().ravel())
    preds = np.concatenate(preds); targs = np.concatenate(targs)
    return _rmse_mae_orig(preds, targs, scaler)


def train_graph(model, train_g, val_g, test_g, scaler, device,
                epochs=300, patience=50, lr=1e-3, wd=1e-4, batch_size=64, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader = GLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader = GLoader(val_g, batch_size=batch_size)
    test_loader = GLoader(test_g, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=15, min_lr=1e-5)
    crit = nn.MSELoss()
    best_val = float('inf'); best_state = None; best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch).squeeze(-1)
            loss = crit(out, batch.y.view(-1))
            loss.backward(); opt.step()
        val_rmse, _ = _eval_graph(model, val_loader, device, scaler)
        sched.step(val_rmse)
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    test_rmse, test_mae = _eval_graph(model, test_loader, device, scaler)
    return {
        'test_rmse': test_rmse, 'test_mae': test_mae,
        'best_val_rmse': best_val, 'best_epoch': best_epoch,
    }


def train_ecfp(model, train_x, train_y, val_x, val_y, test_x, test_y, scaler, device,
               epochs=300, patience=50, lr=1e-3, wd=1e-4, batch_size=128, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tx = torch.tensor(train_x, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.float32)
    vx = torch.tensor(val_x, dtype=torch.float32); vy = torch.tensor(val_y, dtype=torch.float32)
    ex = torch.tensor(test_x, dtype=torch.float32); ey = torch.tensor(test_y, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(ex, ey), batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                       patience=15, min_lr=1e-5)
    crit = nn.MSELoss()
    best_val = float('inf'); best_state = None; best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            out = model(xb).squeeze(-1)
            loss = crit(out, yb); loss.backward(); opt.step()
        val_rmse, _ = _eval_ecfp(model, val_loader, device, scaler)
        sched.step(val_rmse)
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    test_rmse, test_mae = _eval_ecfp(model, test_loader, device, scaler)
    return {
        'test_rmse': test_rmse, 'test_mae': test_mae,
        'best_val_rmse': best_val, 'best_epoch': best_epoch,
    }
