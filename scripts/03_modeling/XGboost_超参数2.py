import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor
import gc
import random
import optuna
from optuna.integration import XGBoostPruningCallback

# ==================== 配置参数 ====================
GRID_SIZE_NEW = 1500
USE_LOG_TRANSFORM = True
REMOVE_OUTLIERS = True
OUTLIER_PERCENTILE = 99
SEED = 42

# GPU 设置
DEVICE = "cuda"  # 或 "cuda:0"
FALLBACK_TO_CPU = True

random.seed(SEED)
np.random.seed(SEED)

# ==================== 路径 ====================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
output_dir = r"./data\Data\GTNNWR\Result\xgboost_optimized"
os.makedirs(output_dir, exist_ok=True)


def safe_load_data():
    """安全加载数据，带异常处理"""
    try:
        print("正在加载原始数据...")
        loaded = np.load(npz_path)
        columns = ['x', 'y', 't', 'SOC_loss', 'erosion', 'R', 'K', 'LS', 'C', 'P']
        data = np.column_stack([loaded[c] for c in columns])
        del loaded
        gc.collect()
        return data, columns
    except Exception as e:
        print(f"数据加载失败: {e}")
        raise


# ===================== 数据处理 =====================
data, columns = safe_load_data()
col_idx = {name: i for i, name in enumerate(columns)}
print(f"原始数据行数: {len(data):,}")
print(f"内存占用约: {data.nbytes / 1024 ** 3:.2f} GB")

print("清洗 NaN...")
valid_mask = np.all(np.isfinite(data), axis=1)
data = data[valid_mask]
print(f"清洗 NaN 后剩余行数: {len(data):,}")

print("计算年份...")
year = np.round(data[:, col_idx['t']] * (2024 - 1990) + 1990).astype(np.int32)
data = np.column_stack([data, year])
col_idx['year'] = data.shape[1] - 1

print("计算 1500m 网格索引...")
grid_x_new = np.floor(data[:, col_idx['x']] / GRID_SIZE_NEW).astype(np.int64)
grid_y_new = np.floor(data[:, col_idx['y']] / GRID_SIZE_NEW).astype(np.int64)
x_new = grid_x_new * GRID_SIZE_NEW + GRID_SIZE_NEW / 2
y_new = grid_y_new * GRID_SIZE_NEW + GRID_SIZE_NEW / 2
data = np.column_stack([data, grid_x_new, grid_y_new, x_new, y_new])
col_idx.update({
    'grid_x_new': data.shape[1] - 4,
    'grid_y_new': data.shape[1] - 3,
    'x_new': data.shape[1] - 2,
    'y_new': data.shape[1] - 1
})

print("转为 pandas 并进行分组聚合...")
df_temp = pd.DataFrame(data, columns=list(col_idx.keys()))
data_1500m = df_temp.groupby(['year', 'grid_x_new', 'grid_y_new'], as_index=False).agg({
    'SOC_loss': 'mean',
    'erosion': 'mean',
    'R': 'mean',
    'K': 'mean',
    'LS': 'mean',
    'C': 'mean',
    'P': 'mean',
    't': 'mean',
    'x_new': 'first',
    'y_new': 'first'
})
print(f"1500m 分辨率聚合后总行数: {len(data_1500m):,}")

del data, df_temp
gc.collect()

# 去噪
data_final = data_1500m.copy()
if REMOVE_OUTLIERS:
    print("去除 SOC_loss 极端异常值...")
    threshold = data_final['SOC_loss'].quantile(OUTLIER_PERCENTILE / 100)
    before = len(data_final)
    data_final = data_final[data_final['SOC_loss'] <= threshold].copy()
    print(f"去除后剩余行数: {len(data_final):,} (原 {before:,})")

print(f"最终用于建模的数据量: {len(data_final):,} 行")

# ==================== 特征工程 ====================
print("进行特征工程...")
data_final['R_K'] = data_final['R'] * data_final['K']
data_final['LS_C'] = data_final['LS'] * data_final['C']
data_final['R_LS'] = data_final['R'] * data_final['LS']
data_final['R_P'] = data_final['R'] * data_final['P']
data_final['year_sq'] = data_final['year'] ** 2
data_final['R_K_LS'] = data_final['R'] * data_final['K'] * data_final['LS']
data_final['R_C_P'] = data_final['R'] * data_final['C'] * data_final['P']
data_final['LS_P'] = data_final['LS'] * data_final['P']
data_final['year_trend'] = (data_final['year'] - 1990) / 34.0

# 新增推荐特征
data_final['RKLS'] = data_final['R'] * data_final['K'] * data_final['LS']
data_final['RKLSCP'] = data_final['R'] * data_final['K'] * data_final['LS'] * data_final['C'] * data_final['P']
data_final['C_P'] = data_final['C'] * data_final['P']
data_final['K_LS'] = data_final['K'] * data_final['LS']
data_final['R_C'] = data_final['R'] * data_final['C']
data_final['year_cubic'] = data_final['year'] ** 3
data_final['year_R'] = data_final['year'] * data_final['R']
data_final['x_y_inter'] = data_final['x_new'] * data_final['y_new']
data_final['x_sq'] = data_final['x_new'] ** 2
data_final['y_sq'] = data_final['y_new'] ** 2

# 年份均值统计
year_stats = data_final.groupby('year')[['R', 'K', 'LS', 'C', 'P']].mean().add_prefix('year_mean_')
data_final = data_final.merge(year_stats, on='year', how='left')

# 空间均值统计（关键修改：移除 SOC_loss，防止数据泄漏）
spatial_stats = data_final.groupby(['grid_x_new', 'grid_y_new'])[['R', 'K', 'LS', 'C', 'P']].mean().add_prefix(
    'grid_mean_')
data_final = data_final.merge(spatial_stats, on=['grid_x_new', 'grid_y_new'], how='left')

# 目标变量
if USE_LOG_TRANSFORM:
    data_final['SOC_loss_log'] = np.log1p(data_final['SOC_loss'])
    target = 'SOC_loss_log'
else:
    target = 'SOC_loss'

# 特征列表（移除 grid_mean_SOC_loss，新增推荐特征）
features = [
    'R', 'K', 'LS', 'C', 'P', 'year',
    'R_K', 'LS_C', 'R_LS', 'R_P', 'year_sq', 'year_trend',
    'R_K_LS', 'R_C_P', 'LS_P',
    'RKLS', 'RKLSCP', 'C_P', 'K_LS', 'R_C',
    'year_cubic', 'year_R',
    'x_y_inter', 'x_sq', 'y_sq',
    'year_mean_R', 'year_mean_K', 'year_mean_LS', 'year_mean_C', 'year_mean_P',
    'grid_mean_R', 'grid_mean_K', 'grid_mean_LS', 'grid_mean_C', 'grid_mean_P'
]

# ==================== 数据集划分 ====================
print("\n=== 每年份内随机抽样划分（train 70% / val 15% / test 15%） ===")
train_dfs = []
val_dfs = []
test_dfs = []

for yr, group in data_final.groupby('year'):
    group = group.sample(frac=1.0, random_state=SEED + int(yr)).reset_index(drop=True)
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
X_val = val_df[features].values
y_val = val_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

print(f"训练集: {len(X_train):,} ({len(X_train) / len(data_final) * 100:.1f}%)")
print(f"验证集: {len(X_val):,} ({len(X_val) / len(data_final) * 100:.1f}%)")
print(f"测试集: {len(X_test):,} ({len(X_test) / len(data_final) * 100:.1f}%)")

# ==================== 阶段1：子样本粗调（增加到200 trials） ====================
print("\n=== 阶段1：子样本粗调 ===")
n_samples_coarse = int(len(X_train) * 0.6)
indices = np.random.choice(len(X_train), n_samples_coarse, replace=False)
X_train_sub = X_train[indices]
y_train_sub = y_train[indices]

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_train_sub, y_train_sub, test_size=0.2, random_state=SEED)

print(f"粗调子样本训练集: {len(X_train_sub):,} 条 | 验证集: {len(X_val_sub):,} 条")


def objective_coarse(trial):
    params = {
        "n_estimators": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0, log=True),
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "device": DEVICE,
        "random_state": SEED,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
    try:
        model.fit(
            X_train_sub, y_train_sub,
            eval_set=[(X_val_sub, y_val_sub)],
            early_stopping_rounds=30,
            eval_metric="rmse",
            verbose=False,
            callbacks=[pruning_callback]
        )
        preds = model.predict(X_val_sub)
        return root_mean_squared_error(y_val_sub, preds)
    except Exception as e:
        if "out of memory" in str(e).lower() and FALLBACK_TO_CPU:
            print("GPU OOM，自动回退到 CPU...")
            params["device"] = "cpu"
            params.pop("predictor", None)
            model = XGBRegressor(**params)
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                early_stopping_rounds=30,
                eval_metric="rmse",
                verbose=False
            )
            preds = model.predict(X_val_sub)
            return root_mean_squared_error(y_val_sub, preds)
        raise


study_coarse = optuna.create_study(direction="minimize")
study_coarse.optimize(objective_coarse, n_trials=50, show_progress_bar=True)

print("\n阶段1 最佳参数:", study_coarse.best_params)
print("阶段1 最佳 RMSE:", study_coarse.best_value)

# ==================== 阶段2：全量细调（增加到50 trials） ====================
print("\n=== 阶段2：全量细调 ===")


def objective_fine(trial):
    best = study_coarse.best_params
    params = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", best['learning_rate'] * 0.6, best['learning_rate'] * 1.8,
                                             log=True),
        "max_depth": trial.suggest_int("max_depth", max(5, best['max_depth'] - 2), best['max_depth'] + 2),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, min(10, best['min_child_weight'] + 4)),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0, log=True),
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "device": DEVICE,
        "random_state": SEED,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            eval_metric="rmse",
            verbose=False,
            callbacks=[pruning_callback]
        )
        preds = model.predict(X_val)
        return root_mean_squared_error(y_val, preds)
    except Exception as e:
        print(f"细调 Trial 失败: {e}")
        raise


study_fine = optuna.create_study(direction="minimize")
study_fine.optimize(objective_fine, n_trials=20, show_progress_bar=True)

print("\n阶段2 最佳参数:", study_fine.best_params)
print("阶段2 最佳 RMSE:", study_fine.best_value)

# 选择最佳参数
best_params = study_fine.best_params if study_fine.best_value < study_coarse.best_value else study_coarse.best_params
print("最终选择:", "阶段2" if study_fine.best_value < study_coarse.best_value else "阶段1")

best_params.update({
    "n_estimators": 8000,  # 增加树数量，配合更小学习率
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "device": DEVICE,
    "random_state": SEED,
    "n_jobs": -1,
})

# ==================== 最终模型训练 ====================
print("\n=== 最终模型训练（合并 train+val） ===")
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

final_model = XGBRegressor(**best_params)
final_model.fit(
    X_full, y_full,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=300,
    eval_metric="rmse",
    verbose=100
)

# ==================== 评估（包含原尺度 R² 和 RMSE） ====================
print("\n=== 模型评估 ===")
train_pred = final_model.predict(X_train)
val_pred = final_model.predict(X_val)
test_pred = final_model.predict(X_test)

# Log 尺度指标
train_rmse_log = root_mean_squared_error(y_train, train_pred)
val_rmse_log = root_mean_squared_error(y_val, val_pred)
test_rmse_log = root_mean_squared_error(y_test, test_pred)

train_r2_log = r2_score(y_train, train_pred)
val_r2_log = r2_score(y_val, val_pred)
test_r2_log = r2_score(y_test, test_pred)

print(f"[Log尺度] Train RMSE: {train_rmse_log:.6f} | R²: {train_r2_log:.4f}")
print(f"[Log尺度] Val   RMSE: {val_rmse_log:.6f} | R²: {val_r2_log:.4f}")
print(f"[Log尺度] Test  RMSE: {test_rmse_log:.6f} | R²: {test_r2_log:.4f}")

# 原尺度指标（关键新增）
if USE_LOG_TRANSFORM:
    y_train_orig = np.expm1(y_train)
    y_val_orig = np.expm1(y_val)
    y_test_orig = np.expm1(y_test)

    train_pred_orig = np.expm1(train_pred)
    val_pred_orig = np.expm1(val_pred)
    test_pred_orig = np.expm1(test_pred)
else:
    y_train_orig = y_train
    y_val_orig = y_val
    y_test_orig = y_test
    train_pred_orig = train_pred
    val_pred_orig = val_pred
    test_pred_orig = test_pred

train_rmse_orig = root_mean_squared_error(y_train_orig, train_pred_orig)
val_rmse_orig = root_mean_squared_error(y_val_orig, val_pred_orig)
test_rmse_orig = root_mean_squared_error(y_test_orig, test_pred_orig)

train_r2_orig = r2_score(y_train_orig, train_pred_orig)
val_r2_orig = r2_score(y_val_orig, val_pred_orig)
test_r2_orig = r2_score(y_test_orig, test_pred_orig)

print(f"\n[原尺度] Train RMSE: {train_rmse_orig:.6f} | R²: {train_r2_orig:.4f}")
print(f"[原尺度] Val   RMSE: {val_rmse_orig:.6f} | R²: {val_r2_orig:.4f}")
print(f"[原尺度] Test  RMSE: {test_rmse_orig:.6f} | R²: {test_r2_orig:.4f}")

if train_rmse_log < val_rmse_log - 0.01:
    print("\n警告：Train RMSE 显著低于 Val RMSE，可能过拟合！")
else:
    print("\nTrain-Val 差距合理，模型泛化良好。")

print("\n优化版模型训练完成！")