import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shap
import zipfile
import gc

import torch
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ==================== 配置参数（4GB显存安全推荐） ====================
SAMPLES_PER_YEAR   = 2500      # 每年的抽样数，建议2000~3000（先用2500测试）
REF_POINTS         = 3000      # 参考点数量，6000~8000（显存紧张用6000）
CHUNK_SIZE         = 3000      # predict分块大小
BATCH_SIZE         = 4096     # 训练batch（4GB显存建议12288或更小）
EPOCHS             = 200
PATIENCE           = 15
SEED               = 42
BETA_L2            = 1e-4
LEARNING_RATE      = 5e-4

# 美观风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
NATURE_COLORS = {
    'red': '#e64b35', 'blue': '#4dbbd5', 'green': '#00a087',
    'dark_blue': '#3c5488', 'orange': '#f39b7f'
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== 数据集 ====================
class SimpleDataset(Dataset):
    def __init__(self, X, Y, C):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.C = torch.from_numpy(C).float()

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], self.C[idx]

# ==================== 模型定义 ====================
class STWeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, delta):
        return self.net(delta)

class GTNNWR(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.weight_net = STWeightNet()
        self.beta = nn.Parameter(torch.randn(p, 1))

    def predict(self, Xi, Ci, Xr, Cr, chunk_size=CHUNK_SIZE):
        B = Xi.size(0)
        if B == 0:
            return torch.empty((0, 1), device=Xi.device)
        y_r = Xr @ self.beta
        num = torch.zeros((B, 1), device=Xi.device)
        den = torch.zeros((B, 1), device=Xi.device)
        R = Xr.size(0)
        for start in range(0, R, chunk_size):
            end = min(start + chunk_size, R)
            Cr_chunk = Cr[start:end]
            yr_chunk = y_r[start:end]
            delta = torch.abs(Ci.unsqueeze(1) - Cr_chunk.unsqueeze(0))
            w_chunk = self.weight_net(delta).squeeze(-1)
            num += torch.matmul(w_chunk, yr_chunk)
            den += w_chunk.sum(dim=1, keepdim=True)
        return num / (den + 1e-6)

    def forward(self, Xi, Ci, Xr, Cr, chunk_size=CHUNK_SIZE):
        return self.predict(Xi, Ci, Xr, Cr, chunk_size)

# ==================== 主程序 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 路径设置
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
best_model_path = r"./data\Data\GTNNWR\gtnnwr_best.pth"
output_dir = r"./data\Data\GTNNWR\figures"
zip_path = r"./data\Data\GTNNWR\GTNNWR_results_pack.zip"

# 1. 加载数据
loaded = np.load(npz_path)
print("npz文件中的键名:", list(loaded.keys()))

data = pd.DataFrame({
    'x': loaded['x'], 'y': loaded['y'], 't': loaded['t'],
    'SOC_loss': loaded['SOC_loss'],
    'R': loaded['R'], 'K': loaded['K'], 'LS': loaded['LS'],
    'C': loaded['C'], 'P': loaded['P']
})

years = np.round(data['t'] * (2024 - 1990) + 1990).astype(int)
data['year'] = years
print(f"原始数据行数: {len(data):,}")

# 2. 分年重采样 + 抽样
sampled_dfs = []
for year in range(1990, 2025):
    df_year = data[data['year'] == year].copy()
    if len(df_year) == 0:
        continue

    print(f"处理 {year} 年，原始 {len(df_year):,} 行")

    # 1000m 重采样
    df_year['grid_x'] = np.floor(df_year['x'] / 1000).astype(int)
    df_year['grid_y'] = np.floor(df_year['y'] / 1000).astype(int)
    df_year['x_new'] = df_year['grid_x'] * 1000 + 500
    df_year['y_new'] = df_year['grid_y'] * 1000 + 500

    agg_dict = {
        'SOC_loss': 'mean', 'R': 'mean', 'K': 'mean', 'LS': 'mean',
        'C': 'mean', 'P': 'mean', 't': 'mean',
        'x_new': 'mean', 'y_new': 'mean'
    }
    df_resampled = df_year.groupby(['grid_x', 'grid_y']).agg(agg_dict).reset_index()
    df_resampled = df_resampled.rename(columns={'x_new': 'x', 'y_new': 'y'})
    df_resampled = df_resampled.drop(columns=['grid_x', 'grid_y'])
    df_resampled['year'] = year

    # 抽样
    if len(df_resampled) > SAMPLES_PER_YEAR:
        df_resampled = df_resampled.sample(n=SAMPLES_PER_YEAR, random_state=SEED)
    else:
        print(f"  {year} 年网格不足，保留全部 {len(df_resampled)} 行")

    sampled_dfs.append(df_resampled)
    print(f"  {year} 年处理后: {len(df_resampled)} 行")

data_sampled = pd.concat(sampled_dfs, ignore_index=True)
print(f"\n总抽样数据量: {len(data_sampled):,} 行")

# 立即释放原始大数组内存
del data, loaded
gc.collect()

# 3. 数据集划分
train_df = data_sampled[(data_sampled.year >= 1990) & (data_sampled.year <= 2009)]
test_df  = data_sampled[(data_sampled.year >= 2010) & (data_sampled.year <= 2019)]
val_df   = data_sampled[(data_sampled.year >= 2020) & (data_sampled.year <= 2024)]

print(f"训练集: {len(train_df):,} | 测试集: {len(test_df):,} | 验证集: {len(val_df):,}")

# 4. 特征工程
features = ['R', 'K', 'LS', 'C', 'P']
target = 'SOC_loss'

def build_arrays(df):
    X = df[features].values
    Y = df[target].values.reshape(-1, 1)
    C = df[['x', 'y', 't']].values
    return X, Y, C

X_tr, Y_tr, C_tr = build_arrays(train_df)
X_te, Y_te, C_te = build_arrays(test_df)
X_va, Y_va, C_va = build_arrays(val_df)

# 归一化
scaler_X = MinMaxScaler().fit(X_tr)
scaler_Y = MinMaxScaler().fit(Y_tr)
X_tr = scaler_X.transform(X_tr)
X_te = scaler_X.transform(X_te)
X_va = scaler_X.transform(X_va)
Y_tr = scaler_Y.transform(Y_tr)
Y_te = scaler_Y.transform(Y_te)
Y_va = scaler_Y.transform(Y_va)

# 时空坐标归一化
SPACE_SCALE = 500000.0
TIME_SCALE = 35.0

def scale_C(C):
    C_scaled = C.copy()
    C_scaled[:, 0:2] /= SPACE_SCALE
    C_scaled[:, 2] /= TIME_SCALE
    return C_scaled

C_tr = scale_C(C_tr)
C_te = scale_C(C_te)
C_va = scale_C(C_va)

# 5. DataLoader（关键：num_workers=0 避免多进程内存爆炸）
train_dataset = SimpleDataset(X_tr, Y_tr, C_tr)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,          # 重要！防止内存爆炸
    pin_memory=True
)

# 6. 模型与训练准备
model = GTNNWR(p=len(features)).to(device)

# torch.compile（安全调用）
# if torch.__version__ >= "2.0" and torch.cuda.is_available():
#     print(f"启用 torch.compile (PyTorch {torch.__version__})")
#     try:
#         model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=True)
#     except Exception as e:
#         print("torch.compile 失败，使用普通模式:", e)
print("使用普通模式训练（已禁用 torch.compile，避免 Triton 问题）")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler_amp = torch.amp.GradScaler('cuda')  # 推荐新写法

# 参考点
g = torch.Generator().manual_seed(SEED)
ref_idx = torch.randperm(len(X_tr), generator=g)[:REF_POINTS].numpy()
Xr_fixed = torch.from_numpy(X_tr[ref_idx]).float().to(device)
Cr_fixed = torch.from_numpy(C_tr[ref_idx]).float().to(device)

# 训练循环
best_val_loss = float('inf')
early_stop_counter = 0
train_losses, val_losses = [], []

print("\n开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for Xi, Yi, Ci in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        Xi, Yi, Ci = Xi.to(device), Yi.to(device), Ci.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            y_pred = model(Xi, Ci, Xr_fixed, Cr_fixed)
            loss = ((y_pred - Yi) ** 2).mean() + BETA_L2 * torch.sum(model.beta ** 2)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()
        num_batches += 1

    train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    train_losses.append(train_loss)

    # 验证
    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(X_va), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(X_va))
            Xi_val = torch.from_numpy(X_va[i:end]).float().to(device)
            Ci_val = torch.from_numpy(C_va[i:end]).float().to(device)
            with torch.amp.autocast(device_type='cuda'):
                pred_val = model(Xi_val, Ci_val, Xr_fixed, Cr_fixed)
            val_preds.append(pred_val.cpu().numpy())

    val_pred = np.vstack(val_preds) if val_preds else np.array([])
    val_loss = ((val_pred - Y_va) ** 2).mean() if len(val_pred) > 0 else float('inf')
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("  → 保存新最佳模型")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f"早停触发（连续 {PATIENCE} 个epoch未改善）")
            break

# 7. 加载最佳模型并评估（以下部分与您原代码相同）
model.load_state_dict(torch.load(best_model_path))

def full_evaluate(X, Y, C, name, return_preds=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(X))
            Xi = torch.from_numpy(X[i:end]).float().to(device)
            Ci = torch.from_numpy(C[i:end]).float().to(device)
            with torch.amp.autocast(device_type='cuda'):
                pred = model(Xi, Ci, Xr_fixed, Cr_fixed)
            preds.append(pred.cpu().numpy())
    y_pred_scaled = np.vstack(preds)
    y_true = scaler_Y.inverse_transform(Y)
    y_pred = scaler_Y.inverse_transform(y_pred_scaled)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    if return_preds:
        return r2, rmse, mae, y_true.flatten(), y_pred.flatten()
    return r2, rmse, mae

print("\n=== GTNNWR 最终评估 ===")
gt_te_r2, gt_te_rmse, gt_te_mae, y_te_true, y_te_pred = full_evaluate(X_te, Y_te, C_te, "2010–2019 测试集", True)
gt_va_r2, gt_va_rmse, gt_va_mae, y_va_true, y_va_pred = full_evaluate(X_va, Y_va, C_va, "2020–2024 验证集", True)

# 线性回归基准
print("\n=== 线性回归基准 ===")
lr = LinearRegression().fit(X_tr, Y_tr)
lr_te_pred = scaler_Y.inverse_transform(lr.predict(X_te).reshape(-1, 1)).flatten()
lr_va_pred = scaler_Y.inverse_transform(lr.predict(X_va).reshape(-1, 1)).flatten()
lr_te_true = scaler_Y.inverse_transform(Y_te).flatten()
lr_va_true = scaler_Y.inverse_transform(Y_va).flatten()

print(f"2010–2019 测试集 (LR) | R²: {r2_score(lr_te_true, lr_te_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(lr_te_true, lr_te_pred)):.4f} | MAE: {mean_absolute_error(lr_te_true, lr_te_pred):.4f}")
print(f"2020–2024 验证集 (LR)  | R²: {r2_score(lr_va_true, lr_va_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(lr_va_true, lr_va_pred)):.4f} | MAE: {mean_absolute_error(lr_va_true, lr_va_pred):.4f}")

# 8. 出图
os.makedirs(output_dir, exist_ok=True)

# 损失曲线
plt.figure(figsize=(9, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color=NATURE_COLORS['red'], linewidth=2)
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', color=NATURE_COLORS['blue'], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'loss_curves.tif'), dpi=300, bbox_inches='tight')
plt.close()

# 测试集散点图
plt.figure(figsize=(9, 6))
plt.scatter(y_te_true, y_te_pred, alpha=0.4, s=15, color=NATURE_COLORS['green'], edgecolor='none')
plt.plot([y_te_true.min(), y_te_true.max()], [y_te_true.min(), y_te_true.max()], '--', color=NATURE_COLORS['dark_blue'], lw=2)
plt.xlabel('True SOC Loss')
plt.ylabel('Predicted SOC Loss')
plt.title('True vs Predicted - 2010–2019 Test Set')
plt.savefig(os.path.join(output_dir, 'scatter_test.tif'), dpi=300, bbox_inches='tight')
plt.close()

# 验证集散点图
plt.figure(figsize=(9, 6))
plt.scatter(y_va_true, y_va_pred, alpha=0.4, s=15, color=NATURE_COLORS['green'], edgecolor='none')
plt.plot([y_va_true.min(), y_va_true.max()], [y_va_true.min(), y_va_true.max()], '--', color=NATURE_COLORS['dark_blue'], lw=2)
plt.xlabel('True SOC Loss')
plt.ylabel('Predicted SOC Loss')
plt.title('True vs Predicted - 2020–2024 Validation Set')
plt.savefig(os.path.join(output_dir, 'scatter_val.tif'), dpi=300, bbox_inches='tight')
plt.close()

# 残差图
residuals_te = y_te_true - y_te_pred
plt.figure(figsize=(9, 6))
plt.scatter(y_te_pred, residuals_te, alpha=0.4, s=15, color=NATURE_COLORS['orange'], edgecolor='none')
plt.axhline(0, color='black', linestyle='--', lw=1.5)
plt.xlabel('Predicted SOC Loss')
plt.ylabel('Residuals')
plt.title('Residual Plot - 2010–2019 Test Set')
plt.savefig(os.path.join(output_dir, 'residual_test.tif'), dpi=300, bbox_inches='tight')
plt.close()

# 指标表
metrics_df = pd.DataFrame({
    'Dataset': ['Test (GTNNWR)', 'Val (GTNNWR)', 'Test (LR)', 'Val (LR)'],
    'R²': [gt_te_r2, gt_va_r2, r2_score(lr_te_true, lr_te_pred), r2_score(lr_va_true, lr_va_pred)],
    'RMSE': [gt_te_rmse, gt_va_rmse, np.sqrt(mean_squared_error(lr_te_true, lr_te_pred)), np.sqrt(mean_squared_error(lr_va_true, lr_va_pred))],
    'MAE': [gt_te_mae, gt_va_mae, mean_absolute_error(lr_te_true, lr_te_pred), mean_absolute_error(lr_va_true, lr_va_pred)]
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics_table.csv'), index=False)

# 9. SHAP（小样本）
print("\n=== SHAP分析（小样本） ===")
num_shap = 120
bg_samples = 600
shap_idx = np.random.choice(len(X_te), num_shap, replace=False)
X_shap = torch.from_numpy(X_te[shap_idx]).float()
C_shap = torch.from_numpy(C_te[shap_idx]).float().to(device)

bg_idx = np.random.choice(len(X_tr), bg_samples, replace=False)
bg = torch.from_numpy(X_tr[bg_idx]).float()

class ShapWrapper(nn.Module):
    def __init__(self, model, Ci, Xr, Cr):
        super().__init__()
        self.model = model
        self.Ci = Ci
        self.Xr = Xr
        self.Cr = Cr
    def forward(self, Xi):
        return self.model(Xi.to(self.Ci.device), self.Ci, self.Xr, self.Cr)

wrapper = ShapWrapper(model, C_shap, Xr_fixed, Cr_fixed)
explainer = shap.DeepExplainer(wrapper, bg.to(device))
shap_values = explainer.shap_values(X_shap.to(device))

shap.summary_plot(shap_values, X_shap.numpy(), feature_names=features,
                  plot_type="bar", color=NATURE_COLORS['blue'])
plt.savefig(os.path.join(output_dir, 'shap_importance.tif'), dpi=300, bbox_inches='tight')
plt.close()

# 10. 打包结果
os.makedirs(output_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            fp = os.path.join(root, file)
            arc = os.path.relpath(fp, output_dir)
            zf.write(fp, arc)
    zf.write(best_model_path, 'gtnnwr_best.pth')

print(f"\n全部完成！结果已打包至：\n{zip_path}")