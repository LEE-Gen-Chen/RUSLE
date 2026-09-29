# ============================================================
# GTNNWR_FULL_PIPELINE.py
# NumPy chunk -> grid aggregation -> sampling -> training
# ============================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import datetime

# ====================== USER CONFIG ======================
BASE_DIR = r"./data\Data\GTNNWR\GTNNWR_Run"
NPZ_PATH = r"./data\Data\GTNNWR\Result_data_all.npz"

SPACE_RES = 1000
CHUNK_SIZE = 2_000_000
SAMPLE_PER_YEAR = 2000

BATCH_SIZE = 1024        # 4GB GPU safe
EPOCHS = 80
LR = 5e-4
XR_SIZE = 8000
GT_CHUNK = 2000

YEARS_TRAIN = range(1990, 2010)
YEARS_TEST  = range(2010, 2020)
YEARS_VAL   = range(2020, 2025)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)
torch.manual_seed(42)

# ====================== PATH SETUP ======================
SAMPLE_DIR = os.path.join(BASE_DIR, "sampled_numpy")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_PATH = os.path.join(BASE_DIR, "run.log")

for d in [BASE_DIR, SAMPLE_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ====================== LOGGING ======================
def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] {msg}\n")

log("===== GTNNWR FULL PIPELINE START =====")
log(f"Device: {DEVICE}")

# ============================================================
# STEP 1: NumPy chunk + vectorized grid aggregation + sampling
# ============================================================

if len(os.listdir(SAMPLE_DIR)) == 0:
    log("Sampling not found, start NumPy chunk aggregation...")

    data = np.load(NPZ_PATH, mmap_mode="r")
    x, y, t = data["x"], data["y"], data["t"]
    SOC, R, K, LS, C, P = data["SOC_loss"], data["R"], data["K"], data["LS"], data["C"], data["P"]

    def t_to_year(tv):
        return np.round(tv * (2024 - 1990) + 1990).astype(np.int16)

    year_blocks = {}
    N = len(x)

    for start in tqdm(range(0, N, CHUNK_SIZE), desc="Chunk scanning"):
        end = min(start + CHUNK_SIZE, N)

        year = t_to_year(t[start:end])
        gx = (x[start:end] // SPACE_RES).astype(np.int32)
        gy = (y[start:end] // SPACE_RES).astype(np.int32)

        feats = np.column_stack([
            SOC[start:end], R[start:end], K[start:end],
            LS[start:end], C[start:end], P[start:end]
        ])

        keys = np.column_stack([year, gx, gy])
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)

        sums = np.zeros((len(uniq), feats.shape[1]), dtype=np.float64)
        cnts = np.zeros(len(uniq), dtype=np.int32)

        np.add.at(sums, inv, feats)
        np.add.at(cnts, inv, 1)

        for i, (yr, gxi, gyi) in enumerate(uniq):
            year_blocks.setdefault(yr, []).append((gxi, gyi, sums[i], cnts[i]))

    for yr, blocks in year_blocks.items():
        agg = {}
        for gxi, gyi, s, c in blocks:
            key = (gxi, gyi)
            if key not in agg:
                agg[key] = [s.copy(), c]
            else:
                agg[key][0] += s
                agg[key][1] += c

        rec = []
        for (gxi, gyi), (s, c) in agg.items():
            rec.append([
                gxi * SPACE_RES + SPACE_RES / 2,
                gyi * SPACE_RES + SPACE_RES / 2,
                yr,
                *(s / c)
            ])

        rec = np.array(rec, dtype=np.float32)
        if len(rec) > SAMPLE_PER_YEAR:
            rec = rec[np.random.choice(len(rec), SAMPLE_PER_YEAR, replace=False)]

        np.save(os.path.join(SAMPLE_DIR, f"sample_{yr}.npy"), rec)
        log(f"Year {yr}: {len(rec)} samples saved")

    log("Sampling finished.")
else:
    log("Sampling cache found, skip sampling.")

# ============================================================
# STEP 2: Load sampled data
# ============================================================

def load_years(years):
    data = []
    for y in years:
        p = os.path.join(SAMPLE_DIR, f"sample_{y}.npy")
        if os.path.exists(p):
            data.append(np.load(p))
    return np.vstack(data)

train = load_years(YEARS_TRAIN)
test  = load_years(YEARS_TEST)
val   = load_years(YEARS_VAL)

X_tr, Y_tr, C_tr = train[:, 4:], train[:, 3:4], train[:, :3]
X_te, Y_te, C_te = test[:, 4:], test[:, 3:4], test[:, :3]

# ============================================================
# STEP 3: Scaling
# ============================================================

sx, sy = MinMaxScaler(), MinMaxScaler()
X_tr = sx.fit_transform(X_tr)
Y_tr = sy.fit_transform(Y_tr)
X_te = sx.transform(X_te)
Y_te = sy.transform(Y_te)

C_tr[:, :2] /= 500_000
C_tr[:, 2] /= 35
C_te[:, :2] /= 500_000
C_te[:, 2] /= 35

# ============================================================
# STEP 4: GTNNWR model
# ============================================================

class STWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class GTNNWR(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.w = STWeightNet()
        self.beta = nn.Parameter(torch.randn(p, 1))
    def forward(self, Xi, Ci, Xr, Cr):
        y_r = Xr @ self.beta
        num = torch.zeros((Xi.size(0), 1), device=Xi.device)
        den = torch.zeros_like(num)
        for i in range(0, len(Xr), GT_CHUNK):
            d = torch.abs(Ci.unsqueeze(1) - Cr[i:i+GT_CHUNK].unsqueeze(0))
            w = self.w(d).squeeze(-1)
            num += w @ y_r[i:i+GT_CHUNK]
            den += w.sum(dim=1, keepdim=True)
        return num / (den + 1e-6)

# ============================================================
# STEP 5: Training
# ============================================================

class DS(Dataset):
    def __init__(self, X, Y, C):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.C = torch.from_numpy(C).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.C[i]

loader = DataLoader(DS(X_tr, Y_tr, C_tr), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = GTNNWR(X_tr.shape[1]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

ref_idx = np.random.choice(len(X_tr), XR_SIZE, replace=False)
Xr = torch.from_numpy(X_tr[ref_idx]).to(DEVICE)
Cr = torch.from_numpy(C_tr[ref_idx]).to(DEVICE)

best_r2 = -1e9

for ep in range(EPOCHS):
    model.train()
    for Xb, Yb, Cb in loader:
        Xb, Yb, Cb = Xb.to(DEVICE), Yb.to(DEVICE), Cb.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = ((model(Xb, Cb, Xr, Cr) - Yb) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_te).float().to(DEVICE)
        Ct = torch.from_numpy(C_te).float().to(DEVICE)
        pred = model(Xt, Ct, Xr, Cr).cpu().numpy()

    r2 = r2_score(sy.inverse_transform(Y_te), sy.inverse_transform(pred))
    log(f"Epoch {ep+1:03d} | Test R2 = {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gtnnwr_best.pth"))

# ============================================================
# STEP 6: Save results
# ============================================================

np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), pred)
with open(os.path.join(RESULT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Best Test R2: {best_r2:.4f}\n")

log("===== PIPELINE FINISHED SUCCESSFULLY =====")
