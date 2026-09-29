import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import gc
import random
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
GRID_SIZE_NEW = 1500                  # 使用 1500m 网格（如 XGBoost 版）
USE_LOG_TRANSFORM = True              # 与 XGBoost 版一致，使用 log1p 变换
REMOVE_OUTLIERS = True
OUTLIER_PERCENTILE = 99
SEED = 42
N_ESTIMATORS = 12000                  # 最佳折中值（可根据需要调整为 10000~15000）

random.seed(SEED)
np.random.seed(SEED)

# ==================== 路径 ====================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
output_dir = r"./data\Data\GTNNWR\Result\xgboost_gt nnwr"
os.makedirs(output_dir, exist_ok=True)
model_save_path = os.path.join(output_dir, "xgboost_gt nnwr_final.json")

# 图片路径
feature_importance_path = os.path.join(output_dir, "feature_importance_gt nnwr.png")

# ==================== 数据加载 ====================
print("正在加载原始数据...")
loaded = np.load(npz_path)
data = pd.DataFrame({
    'x': loaded['x'],
    'y': loaded['y'],
    't': loaded['t'],
    'SOC_loss': loaded['SOC_loss'],
    'erosion': loaded['erosion'],
    'R': loaded['R'],
    'K': loaded['K'],
    'LS': loaded['LS'],
    'C': loaded['C'],
    'P': loaded['P']
})
data = data.dropna()

# ==================== 计算年份 + 1500m 网格聚合（XGBoost 抽样思路） ====================
print("计算年份与 1500m 网格...")
years = np.round(data['t'] * (2024 - 1990) + 1990).astype(int)
data['year'] = years

data['grid_x_new'] = np.floor(data['x'] / GRID_SIZE_NEW).astype(int)
data['grid_y_new'] = np.floor(data['y'] / GRID_SIZE_NEW).astype(int)
data['x_new'] = data['grid_x_new'] * GRID_SIZE_NEW + GRID_SIZE_NEW / 2
data['y_new'] = data['grid_y_new'] * GRID_SIZE_NEW + GRID_SIZE_NEW / 2

# 分组聚合（每年 + 网格）
agg_dict = {
    'SOC_loss': 'mean', 'erosion': 'mean', 'R': 'mean', 'K': 'mean',
    'LS': 'mean', 'C': 'mean', 'P': 'mean', 't': 'mean',
    'x_new': 'mean', 'y_new': 'mean'
}
data_agg = data.groupby(['year', 'grid_x_new', 'grid_y_new']).agg(agg_dict).reset_index()
data_agg = data_agg.rename(columns={'x_new': 'x', 'y_new': 'y'})

print(f"1500m 网格聚合后总行数: {len(data_agg):,}")

if REMOVE_OUTLIERS:
    print("去除 SOC_loss 极端异常值...")
    threshold = data_agg['SOC_loss'].quantile(OUTLIER_PERCENTILE / 100)
    before = len(data_agg)
    data_agg = data_agg[data_agg['SOC_loss'] <= threshold].copy()
    print(f"去除后剩余行数: {len(data_agg):,} (原 {before:,})")

# ==================== 特征工程（XGBoost 强项：交互 + 时空统计） ====================
print("进行特征工程...")
data_final = data_agg.copy()

data_final['R_K'] = data_final['R'] * data_final['K']
data_final['LS_C'] = data_final['LS'] * data_final['C']
data_final['R_LS'] = data_final['R'] * data_final['LS']
data_final['R_P'] = data_final['R'] * data_final['P']
data_final['year_sq'] = data_final['year'] ** 2
data_final['R_K_LS'] = data_final['R'] * data_final['K'] * data_final['LS']
data_final['R_C_P'] = data_final['R'] * data_final['C'] * data_final['P']
data_final['LS_P'] = data_final['LS'] * data_final['P']
data_final['year_trend'] = (data_final['year'] - 1990) / 34.0

# 年份均值统计
year_stats = data_final.groupby('year')[['R', 'K', 'LS', 'C', 'P']].mean().add_prefix('year_mean_')
data_final = data_final.merge(year_stats, on='year', how='left')

# 空间均值统计
spatial_stats = data_final.groupby(['grid_x_new', 'grid_y_new'])[['R', 'K', 'LS', 'C', 'P', 'SOC_loss']].mean().add_prefix('grid_mean_')
data_final = data_final.merge(spatial_stats, on=['grid_x_new', 'grid_y_new'], how='left')

if USE_LOG_TRANSFORM:
    data_final['SOC_loss_log'] = np.log1p(data_final['SOC_loss'])
    target = 'SOC_loss_log'
else:
    target = 'SOC_loss'

features = [
    'R', 'K', 'LS', 'C', 'P', 'year',
    'R_K', 'LS_C', 'R_LS', 'R_P',
    'year_sq', 'year_trend',
    'R_K_LS', 'R_C_P', 'LS_P',
    'year_mean_R', 'year_mean_K', 'year_mean_LS', 'year_mean_C', 'year_mean_P',
    'grid_mean_R', 'grid_mean_K', 'grid_mean_LS', 'grid_mean_C', 'grid_mean_P',
    'grid_mean_SOC_loss'
]

# ==================== 每年份内随机抽样划分（XGBoost 抽样方法） ====================
print("\n=== 每年份内随机抽样划分（train 70% / val 15% / test 15%） ===")
train_dfs, val_dfs, test_dfs = [], [], []
for yr, group in data_final.groupby('year'):
    group = group.sample(frac=1.0, random_state=SEED + int(yr)).reset_index(drop=True)  # 打乱
    n = len(group)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_dfs.append(group.iloc[:n_train])
    val_dfs.append(group.iloc[n_train:n_train + n_val])
    test_dfs.append(group.iloc[n_train + n_val:])

train_df = pd.concat(train_dfs, ignore_index=True)
val_df = pd.concat(val_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

X_train = train_df[features].values
y_train = train_df[target].values
X_val   = val_df[features].values
y_val   = val_df[target].values
X_test  = test_df[features].values
y_test  = test_df[target].values

print(f"训练集: {len(X_train):,} ({len(X_train)/len(data_final)*100:.1f}%)")
print(f"验证集: {len(X_val):,} ({len(X_val)/len(data_final)*100:.1f}%)")
print(f"测试集: {len(X_test):,} ({len(X_test)/len(data_final)*100:.1f}%)")

# ==================== XGBoost 模型训练（核心） ====================
print("\n=== XGBoost版 GTNNWR 训练（最佳参数） ===")
best_params = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.09052034737785881,
    "max_depth": 11,
    "min_child_weight": 7,
    "subsample": 0.9982243819097307,
    "colsample_bytree": 0.7862410638365948,
    "reg_lambda": 1.179748132268195,
    "reg_alpha": 0.003330536873817549,
    "tree_method": "hist",
    "device": "cpu",  # 如有 GPU 可改为 "cuda"
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "rmse",
    "early_stopping_rounds": 300,
}

# 合并 train + val 训练最终模型
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

model = XGBRegressor(**best_params)
model.fit(X_full, y_full, eval_set=[(X_val, y_val)], verbose=100)

# ==================== 评估（原始尺度） ====================
def evaluate(y_true_log, y_pred_log, name):
    # 逆变换到原始尺度
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    print(f"\n--- {name} XGBoost-GTNNWR (原始尺度) ---")
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = float('nan')
    print(f"SOC_loss: R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}%")

evaluate(y_val, model.predict(X_val), "验证集")
evaluate(y_test, model.predict(X_test), "测试集")

# ==================== 特征重要性图 ====================
plt.figure(figsize=(10, 8))
from xgboost import plot_importance
plot_importance(model, importance_type='gain', max_num_features=20)
plt.title('Top 20 Feature Importance (Gain) - XGBoost GTNNWR')
plt.tight_layout()
plt.savefig(feature_importance_path, dpi=300)
plt.close()
print(f"特征重要性图保存: {feature_importance_path}")

# ==================== 保存模型 ====================
model.save_model(model_save_path)
print(f"XGBoost版 GTNNWR 模型保存: {model_save_path}")

print("\nXGBoost思路改进版 GTNNWR（1500m网格 + 每年份内抽样）完成！")