import os
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor, early_stopping, plot_importance
from sklearn.neighbors import NearestNeighbors
import random
from sklearn.utils import shuffle

# ==================== 配置参数（可调） ====================
GRID_SIZE_NEW       = 1500
USE_LOG_TRANSFORM   = True
REMOVE_OUTLIERS     = True
OUTLIER_PERCENTILE  = 99.5
SEED                = 42

random.seed(SEED)
np.random.seed(SEED)

USE_NEIGHBOR_FEATURES = True
K_NEIGHBORS           = 8

SPATIAL_SPLIT         = True

ADD_LAG_FEATURES      = True
LAG_YEARS             = 1

SEGMENTED_MODEL       = True
SEGMENT_QUANTILE      = 0.70

lgb_params_base = {
    "n_estimators"     : 20000,
    "learning_rate"    : 0.03,
    "num_leaves"       : 512,
    "max_depth"        : -1,
    "min_child_samples": 20,
    "subsample"        : 0.9,
    "colsample_bytree" : 0.9,
    "reg_lambda"       : 5.0,
    "reg_alpha"        : 0.5,
    "random_state"     : SEED,
    "n_jobs"           : -1,
    "verbosity"        : 1,
}

# ==================== 路径 ====================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"

output_dir = r"./data\Data\GTNNWR\Result\LightGBM_GTNNWR"
os.makedirs(output_dir, exist_ok=True)

model_save_path_base_txt  = os.path.join(output_dir, "lightgbm_gt_nnwr_base.txt")
model_save_path_base_pkl  = os.path.join(output_dir, "lightgbm_gt_nnwr_base.pkl")
model_save_path_low_txt   = os.path.join(output_dir, "lightgbm_gt_nnwr_low.txt")
model_save_path_low_pkl   = os.path.join(output_dir, "lightgbm_gt_nnwr_low.pkl")
model_save_path_high_txt  = os.path.join(output_dir, "lightgbm_gt_nnwr_high.txt")
model_save_path_high_pkl  = os.path.join(output_dir, "lightgbm_gt_nnwr_high.pkl")

features_path           = os.path.join(output_dir, "features.pkl")
predictions_csv         = os.path.join(output_dir, "predictions_with_residuals.csv")
perf_table_csv          = os.path.join(output_dir, "performance_table.csv")
frozen_parquet_path     = r"./data\Data\GTNNWR\prediction_frozen.parquet"

# ==================== 加载数据 ====================
print("正在加载原始数据...")
loaded = np.load(npz_path)
data = pd.DataFrame({
    'x'       : loaded['x'],
    'y'       : loaded['y'],
    't'       : loaded['t'],
    'SOC_loss': loaded['SOC_loss'],
    'erosion' : loaded['erosion'],
    'R'       : loaded['R'],
    'K'       : loaded['K'],
    'LS'      : loaded['LS'],
    'C'       : loaded['C'],
    'P'       : loaded['P']
})
data = data.dropna()

# ==================== 计算年份 + 1500m 网格聚合 ====================
print("计算年份与 1500m 网格...")
years = np.round(data['t'] * (2024 - 1990) + 1990).astype(int)
data['year'] = years
data['grid_x_new'] = np.floor(data['x'] / GRID_SIZE_NEW).astype(int)
data['grid_y_new'] = np.floor(data['y'] / GRID_SIZE_NEW).astype(int)
data['x_new'] = data['grid_x_new'] * GRID_SIZE_NEW + GRID_SIZE_NEW / 2
data['y_new'] = data['grid_y_new'] * GRID_SIZE_NEW + GRID_SIZE_NEW / 2

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

print("\n=== SOC_loss 分布统计（聚合后） ===")
print(data_agg['SOC_loss'].describe())

# ==================== 特征工程 ====================
print("进行特征工程...")
data_final = data_agg.copy()

# 交互项
data_final['R_K']     = data_final['R'] * data_final['K']
data_final['LS_C']    = data_final['LS'] * data_final['C']
data_final['R_LS']    = data_final['R'] * data_final['LS']
data_final['R_P']     = data_final['R'] * data_final['P']
data_final['R_K_LS']  = data_final['R'] * data_final['K'] * data_final['LS']
data_final['R_C_P']   = data_final['R'] * data_final['C'] * data_final['P']
data_final['LS_P']    = data_final['LS'] * data_final['P']
data_final['year_sq'] = data_final['year'] ** 2
data_final['year_trend'] = (data_final['year'] - 1990) / (2024 - 1990)

# 年份均值
year_stats = data_final.groupby('year')[['R','K','LS','C','P']].mean().add_prefix('year_mean_')
data_final = data_final.merge(year_stats, on='year', how='left')

# 网格均值
spatial_stats = data_final.groupby(['grid_x_new','grid_y_new'])[['R','K','LS','C','P']].mean().add_prefix('grid_mean_')
data_final = data_final.merge(spatial_stats, on=['grid_x_new','grid_y_new'], how='left')

# 归一化坐标
print("添加归一化空间坐标特征...")
data_final['norm_x'] = (data_final['x'] - data_final['x'].min()) / (data_final['x'].max() - data_final['x'].min() + 1e-8)
data_final['norm_y'] = (data_final['y'] - data_final['y'].min()) / (data_final['y'].max() - data_final['y'].min() + 1e-8)

# 邻域特征
if USE_NEIGHBOR_FEATURES:
    print(f"计算空间邻域均值特征（K={K_NEIGHBORS}）...")
    unique_grids = data_final[['grid_x_new', 'grid_y_new']].drop_duplicates().reset_index(drop=True)
    coords = (unique_grids[['grid_x_new', 'grid_y_new']].values * GRID_SIZE_NEW) + GRID_SIZE_NEW / 2
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]
    factors = ['R', 'K', 'LS', 'C', 'P']
    grid_values = data_final.groupby(['grid_x_new', 'grid_y_new'])[factors].mean().values
    neighbor_values = grid_values[indices]
    neighbor_means = np.nanmean(neighbor_values, axis=1)
    neighbor_df = pd.DataFrame(neighbor_means, columns=[f'nb_mean_{f}' for f in factors])
    neighbor_df[['grid_x_new', 'grid_y_new']] = unique_grids.values
    data_final = data_final.merge(neighbor_df, on=['grid_x_new', 'grid_y_new'], how='left')

# 滞后特征
if ADD_LAG_FEATURES:
    print(f"添加滞后特征（lag={LAG_YEARS}）...")
    data_final = data_final.sort_values(['grid_x_new', 'grid_y_new', 'year']).reset_index(drop=True)
    data_final['SOC_loss_lag1'] = data_final.groupby(['grid_x_new', 'grid_y_new'])['SOC_loss'].shift(LAG_YEARS)
    data_final['erosion_lag1'] = data_final.groupby(['grid_x_new', 'grid_y_new'])['erosion'].shift(LAG_YEARS)
    data_final['R_lag1']      = data_final.groupby(['grid_x_new', 'grid_y_new'])['R'].shift(LAG_YEARS)
    data_final['SOC_loss_lag1'] = data_final['SOC_loss_lag1'].fillna(data_final.groupby(['grid_x_new','grid_y_new'])['SOC_loss'].transform('mean'))
    data_final['erosion_lag1']  = data_final['erosion_lag1'].fillna(data_final.groupby(['grid_x_new','grid_y_new'])['erosion'].transform('mean'))
    data_final['R_lag1']        = data_final['R_lag1'].fillna(data_final.groupby(['grid_x_new','grid_y_new'])['R'].transform('mean'))

# 距离中心特征
centroid_x = data_final['x'].mean()
centroid_y = data_final['y'].mean()
data_final['dist_to_centroid']      = np.sqrt((data_final['x'] - centroid_x)**2 + (data_final['y'] - centroid_y)**2)
data_final['dist_to_centroid_norm'] = (data_final['dist_to_centroid'] - data_final['dist_to_centroid'].min()) / (data_final['dist_to_centroid'].max() - data_final['dist_to_centroid'].min() + 1e-8)

# ==================== 特征列表 ====================
features = [
    'R', 'K', 'LS', 'C', 'P', 'year',
    'R_K', 'LS_C', 'R_LS', 'R_P', 'year_sq', 'year_trend',
    'R_K_LS', 'R_C_P', 'LS_P',
    'year_mean_R', 'year_mean_K', 'year_mean_LS', 'year_mean_C', 'year_mean_P',
    'grid_mean_R', 'grid_mean_K', 'grid_mean_LS', 'grid_mean_C', 'grid_mean_P',
    'norm_x', 'norm_y',
    'dist_to_centroid', 'dist_to_centroid_norm'
]
if USE_NEIGHBOR_FEATURES:
    features += [f'nb_mean_{f}' for f in ['R', 'K', 'LS', 'C', 'P']]
if ADD_LAG_FEATURES:
    features += ['SOC_loss_lag1', 'erosion_lag1', 'R_lag1']

features = list(dict.fromkeys(features))

# ==================== 划分训练/验证/测试集 ====================
print("\n=== 划分训练/验证/测试集 ===")
if SPATIAL_SPLIT:
    unique_grids = data_final[['grid_x_new', 'grid_y_new']].drop_duplicates().reset_index(drop=True)
    unique_grids['uid'] = unique_grids.index
    u = shuffle(unique_grids, random_state=SEED)
    n = len(u)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_u = u.iloc[:n_train]
    val_u   = u.iloc[n_train:n_train + n_val]
    test_u  = u.iloc[n_train + n_val:]
    train_idx = pd.merge(data_final, train_u[['grid_x_new','grid_y_new']], on=['grid_x_new','grid_y_new']).index
    val_idx   = pd.merge(data_final, val_u[['grid_x_new','grid_y_new']], on=['grid_x_new','grid_y_new']).index
    test_idx  = pd.merge(data_final, test_u[['grid_x_new','grid_y_new']], on=['grid_x_new','grid_y_new']).index
    train_df = data_final.loc[train_idx].reset_index(drop=True)
    val_df   = data_final.loc[val_idx].reset_index(drop=True)
    test_df  = data_final.loc[test_idx].reset_index(drop=True)
    print("采用空间块拆分")
else:
    train_dfs, val_dfs, test_dfs = [], [], []
    for yr, group in data_final.groupby('year'):
        group = group.sample(frac=1.0, random_state=SEED + int(yr)).reset_index(drop=True)
        n = len(group)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df   = pd.concat(val_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs, ignore_index=True)
    print("采用每年内随机抽样拆分")

X_train_df = train_df[features]
y_train    = train_df['SOC_loss'].values
X_val_df   = val_df[features]
y_val      = val_df['SOC_loss'].values
X_test_df  = test_df[features]
y_test     = test_df['SOC_loss'].values

if USE_LOG_TRANSFORM:
    print("应用 log1p 变换...")
    y_train_log = np.log1p(y_train)
    y_val_log   = np.log1p(y_val)
    y_test_log  = np.log1p(y_test)
else:
    y_train_log = y_train
    y_val_log   = y_val
    y_test_log  = y_test

# ==================== 分段建模 ====================
if SEGMENTED_MODEL:
    print("训练 base model（用于分段）...")
    base_obj = "poisson" if np.mean(y_train) < 5 else "gamma"
    base_params = lgb_params_base.copy()
    base_params["objective"] = base_obj
    base_model = LGBMRegressor(**base_params)
    base_model.fit(
        X_train_df, y_train_log,
        eval_set=[(X_val_df, y_val_log)],
        eval_metric='rmse',
        callbacks=[early_stopping(800)]
    )

    train_base_pred_log = base_model.predict(X_train_df)
    train_base_pred = np.expm1(train_base_pred_log) if USE_LOG_TRANSFORM else train_base_pred_log
    seg_thresh = np.quantile(train_base_pred, SEGMENT_QUANTILE)
    print(f"分段阈值: {seg_thresh:.4f}")

    def base_predict_assign(model, X_df):
        p_log = model.predict(X_df)
        p = np.expm1(p_log) if USE_LOG_TRANSFORM else p_log
        label = (p > seg_thresh).astype(int)
        return label, p_log, p

    train_seg_label, _, _ = base_predict_assign(base_model, X_train_df)

    models = {}
    for seg, obj in [(0, "poisson"), (1, "gamma")]:
        idx = np.where(train_seg_label == seg)[0]
        if len(idx) < 50:
            models[seg] = None
            continue
        params = lgb_params_base.copy()
        params["objective"] = obj
        m = LGBMRegressor(**params)
        print(f"训练段 {seg} 模型 ({obj})，样本数={len(idx)}")
        m.fit(
            X_train_df.iloc[idx], y_train_log[idx],
            eval_set=[(X_val_df, y_val_log)],
            eval_metric='rmse',
            callbacks=[early_stopping(800)]
        )
        models[seg] = m

    def predict_segmented_both(X_df):
        seg_flag, base_p_log, base_p = base_predict_assign(base_model, X_df)
        preds_log = np.zeros(len(X_df))
        for seg in [0, 1]:
            mask = seg_flag == seg
            if mask.sum() == 0:
                continue
            if models.get(seg) is None:
                preds_log[mask] = base_model.predict(X_df[mask])
            else:
                preds_log[mask] = models[seg].predict(X_df[mask])
        preds_orig = np.expm1(preds_log) if USE_LOG_TRANSFORM else preds_log
        return preds_log, preds_orig
else:
    print("训练单一模型 (gamma)")
    params = lgb_params_base.copy()
    params["objective"] = "gamma"
    model_single = LGBMRegressor(**params)
    model_single.fit(
        X_train_df, y_train_log,
        eval_set=[(X_val_df, y_val_log)],
        eval_metric='rmse',
        callbacks=[early_stopping(800)]
    )

    def predict_single_both(X_df):
        preds_log = model_single.predict(X_df)
        preds_orig = np.expm1(preds_log) if USE_LOG_TRANSFORM else preds_log
        return preds_log, preds_orig

# ==================== 评估 ====================
def compute_metrics(y_true, y_pred_orig, y_true_log, y_pred_log, label):
    y_pred_orig = np.clip(y_pred_orig, 0, None)
    r2 = r2_score(y_true, y_pred_orig)
    rmse_orig = math.sqrt(((y_true - y_pred_orig) ** 2).mean())
    mae = mean_absolute_error(y_true, y_pred_orig)
    mape = np.mean(np.abs((y_true - y_pred_orig) / (y_true + 1e-8))) * 100
    rmse_log = math.sqrt(((y_true_log - y_pred_log) ** 2).mean())
    print(f"\n--- {label} ---")
    print(f"R²={r2:.4f} | RMSE={rmse_orig:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}% | Log-RMSE={rmse_log:.6f}")
    return {"r2": r2, "rmse_orig": rmse_orig, "mae": mae, "mape": mape, "rmse_log": rmse_log}

if SEGMENTED_MODEL:
    val_pred_log, val_pred   = predict_segmented_both(X_val_df)
    test_pred_log, test_pred = predict_segmented_both(X_test_df)
else:
    val_pred_log, val_pred   = predict_single_both(X_val_df)
    test_pred_log, test_pred = predict_single_both(X_test_df)

val_metrics  = compute_metrics(y_val,  val_pred,  y_val_log,  val_pred_log,  "验证集")
test_metrics = compute_metrics(y_test, test_pred, y_test_log, test_pred_log, "测试集")

perf_table = pd.DataFrame({
    "set": ["val", "test"],
    "r2": [val_metrics["r2"], test_metrics["r2"]],
    "rmse_orig": [val_metrics["rmse_orig"], test_metrics["rmse_orig"]],
    "mae": [val_metrics["mae"], test_metrics["mae"]],
    "mape": [val_metrics["mape"], test_metrics["mape"]],
    "rmse_log": [val_metrics["rmse_log"], test_metrics["rmse_log"]]
})
perf_table.to_csv(perf_table_csv, index=False)
print(f"性能表保存：{perf_table_csv}")

# ==================== 保存 val/test 预测与残差 ====================
val_out  = val_df.copy().assign(pred=val_pred, residual=val_pred - val_df['SOC_loss'])
test_out = test_df.copy().assign(pred=test_pred, residual=test_pred - test_df['SOC_loss'])
all_out = pd.concat([val_out, test_out], ignore_index=True)
all_out.to_csv(predictions_csv, index=False)
print(f"预测与残差保存：{predictions_csv}")

# ==================== 冻结全数据集预测 ====================
print("\n=== 冻结整个数据集预测 ===")
X_full = data_final[features]
if SEGMENTED_MODEL:
    full_pred_log, full_pred = predict_segmented_both(X_full)
else:
    full_pred_log, full_pred = predict_single_both(X_full)

df_out = data_final.copy()
df_out["预测_SOC_loss"] = full_pred
df_out["残差"]          = full_pred - df_out["SOC_loss"]
df_out.to_parquet(frozen_parquet_path)
print(f"冻结预测结果保存：{frozen_parquet_path}")

# ==================== 保存模型（.txt + .pkl） ====================
def save_model_both_formats(model, path_txt, path_pkl):
    if model is None:
        print(f"模型为 None，跳过：{path_txt}")
        return
    try:
        model.booster_.save_model(path_txt)
        print(f"保存 .txt: {path_txt}")
    except Exception as e:
        print(f".txt 保存失败: {e}")
    try:
        joblib.dump(model, path_pkl)
        print(f"保存 .pkl: {path_pkl}")
    except Exception as e:
        print(f".pkl 保存失败: {e}")

print("\n=== 保存模型（.txt 和 .pkl） ===")
if SEGMENTED_MODEL:
    save_model_both_formats(base_model,  model_save_path_base_txt,  model_save_path_base_pkl)
    if models.get(0) is not None:
        save_model_both_formats(models[0], model_save_path_low_txt, model_save_path_low_pkl)
    if models.get(1) is not None:
        save_model_both_formats(models[1], model_save_path_high_txt, model_save_path_high_pkl)
else:
    save_model_both_formats(model_single, model_save_path_base_txt, model_save_path_base_pkl)

joblib.dump(features, features_path)
print(f"特征列表保存：{features_path}")

print("\n全部完成。")
print("输出目录：", output_dir)
print("冻结预测文件：", frozen_parquet_path)
print("模型文件（.pkl 推荐使用）：", output_dir)