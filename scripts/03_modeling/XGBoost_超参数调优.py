import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import gc
import random
import matplotlib.pyplot as plt  # 用于绘图

# ==================== 配置参数 ====================
GRID_SIZE_NEW = 1500
USE_LOG_TRANSFORM = True  # 使用 log1p 变换目标变量
REMOVE_OUTLIERS = True
OUTLIER_PERCENTILE = 99
SEED = 42

# 强制使用 CPU（稳定，避免显存问题）
DEVICE = "cpu"

random.seed(SEED)
np.random.seed(SEED)

# ==================== 路径 ====================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
output_dir = r"./data\Data\GTNNWR\Result\xgboost_optimized"
os.makedirs(output_dir, exist_ok=True)
model_save_path = os.path.join(output_dir, "xgboost_final_model.json")

# 图片保存路径
rmse_curve_path = os.path.join(output_dir, "rmse_decline_curve.png")
report_metrics_path = os.path.join(output_dir, "report_metrics_table.png")
feature_importance_path = os.path.join(output_dir, "feature_importance.png")

def safe_load_data():
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
    'SOC_loss': 'mean', 'erosion': 'mean', 'R': 'mean', 'K': 'mean',
    'LS': 'mean', 'C': 'mean', 'P': 'mean', 't': 'mean',
    'x_new': 'first', 'y_new': 'first'
})
print(f"1500m 分辨率聚合后总行数: {len(data_1500m):,}")
del data, df_temp
gc.collect()

if REMOVE_OUTLIERS:
    print("去除 SOC_loss 极端异常值...")
    threshold = data_1500m['SOC_loss'].quantile(OUTLIER_PERCENTILE / 100)
    before = len(data_1500m)
    data_1500m = data_1500m[data_1500m['SOC_loss'] <= threshold].copy()
    print(f"去除后剩余行数: {len(data_1500m):,} (原 {before:,})")

print(f"最终用于建模的数据量: {len(data_1500m):,} 行")

# 特征工程
print("进行特征工程...")
data_final = data_1500m.copy()
data_final['R_K'] = data_final['R'] * data_final['K']
data_final['LS_C'] = data_final['LS'] * data_final['C']
data_final['R_LS'] = data_final['R'] * data_final['LS']
data_final['R_P'] = data_final['R'] * data_final['P']
data_final['year_sq'] = data_final['year'] ** 2
data_final['R_K_LS'] = data_final['R'] * data_final['K'] * data_final['LS']
data_final['R_C_P'] = data_final['R'] * data_final['C'] * data_final['P']
data_final['LS_P'] = data_final['LS'] * data_final['P']
data_final['year_trend'] = (data_final['year'] - 1990) / 34.0

year_stats = data_final.groupby('year')[['R', 'K', 'LS', 'C', 'P']].mean().add_prefix('year_mean_')
data_final = data_final.merge(year_stats, on='year', how='left')

spatial_stats = data_final.groupby(['grid_x_new', 'grid_y_new'])[
    ['R', 'K', 'LS', 'C', 'P', 'SOC_loss']].mean().add_prefix('grid_mean_')
data_final = data_final.merge(spatial_stats, on=['grid_x_new', 'grid_y_new'], how='left')

# 目标变量
if USE_LOG_TRANSFORM:
    data_final['SOC_loss_log'] = np.log1p(data_final['SOC_loss'])
    target = 'SOC_loss_log'
else:
    target = 'SOC_loss'

features = [
    'R', 'K', 'LS', 'C', 'P', 'year', 'R_K', 'LS_C', 'R_LS', 'R_P',
    'year_sq', 'year_trend', 'R_K_LS', 'R_C_P', 'LS_P',
    'year_mean_R', 'year_mean_K', 'year_mean_LS', 'year_mean_C', 'year_mean_P',
    'grid_mean_R', 'grid_mean_K', 'grid_mean_LS', 'grid_mean_C', 'grid_mean_P',
    'grid_mean_SOC_loss'
]

# 数据划分（每年份内随机）
print("\n=== 每年份内随机抽样划分（train 70% / val 15% / test 15%） ===")
train_dfs, val_dfs, test_dfs = [], [], []
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
X_val   = val_df[features].values
y_val   = val_df[target].values
X_test  = test_df[features].values
y_test  = test_df[target].values

print(f"训练集: {len(X_train):,} ({len(X_train)/len(data_final)*100:.1f}%)")
print(f"验证集: {len(X_val):,} ({len(X_val)/len(data_final)*100:.1f}%)")
print(f"测试集: {len(X_test):,} ({len(X_test)/len(data_final)*100:.1f}%)")

# ==================== 使用最佳参数直接训练最终模型 ====================
print("\n=== 使用调优得到的最佳参数训练最终模型 ===")

best_params = {
    "n_estimators": 12000,                 # 设置为 12000
    "learning_rate": 0.09052034737785881,
    "max_depth": 11,
    "min_child_weight": 7,
    "subsample": 0.9982243819097307,
    "colsample_bytree": 0.7862410638365948,
    "reg_lambda": 1.179748132268195,
    "reg_alpha": 0.003330536873817549,
    "tree_method": "hist",
    "device": DEVICE,
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "rmse",
    "early_stopping_rounds": 200,
}

print("最佳参数：")
for k, v in best_params.items():
    if k not in ["n_estimators", "early_stopping_rounds", "eval_metric"]:
        print(f"  {k}: {v}")

# 合并 train + val 训练最终模型
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

final_model = XGBRegressor(**best_params)
final_model.fit(
    X_full, y_full,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# ==================== 性能评估 ====================
print("\n=== 过拟合监控 ===")
train_pred = final_model.predict(X_train)
val_pred   = final_model.predict(X_val)
test_pred  = final_model.predict(X_test)

train_rmse = root_mean_squared_error(y_train, train_pred)
val_rmse   = root_mean_squared_error(y_val,   val_pred)
test_rmse  = root_mean_squared_error(y_test,  test_pred)

print(f"Train RMSE: {train_rmse:.6f}")
print(f"Val RMSE:   {val_rmse:.6f}")
print(f"Test RMSE:  {test_rmse:.6f}")
print(f"Train-Val Gap: {train_rmse - val_rmse:.6f}")

if train_rmse < val_rmse - 0.01:
    print("\n警告：可能轻微过拟合，建议进一步加强正则化")
else:
    print("\nTrain-Val 差距合理，模型泛化良好。")

# ==================== 绘制 RMSE 下降曲线 + 报告指标 + 特征重要性 ====================
print("\n=== 生成图片：RMSE 下降曲线、报告指标表格、特征重要性图 ===")

# 获取训练历史
evals_result = final_model.evals_result()
rmse_history = evals_result['validation_0']['rmse']
iterations = list(range(len(rmse_history)))

# 关键里程碑（可自定义）
milestones = [0, 100, 1000, 2000, 5000, 8000, 10000, len(iterations)-1]
milestone_labels = ['Start', '100', '1000', '2000', '5000', '8000', '10000', 'Final']

# 1. RMSE 下降曲线
plt.figure(figsize=(12, 7))
plt.plot(iterations, rmse_history, 'b-', linewidth=2, label='Validation RMSE')
for i, idx in enumerate(milestones):
    if idx < len(rmse_history):
        plt.scatter(idx, rmse_history[idx], color='red', s=80, zorder=5)
        plt.text(idx, rmse_history[idx] + 0.005, f"{rmse_history[idx]:.5f}\n({milestone_labels[i]})",
                 fontsize=10, ha='center', color='red', fontweight='bold')

plt.xlabel('Training Iterations (Trees)', fontsize=12)
plt.ylabel('Validation RMSE (log scale)', fontsize=12)
plt.title('XGBoost Validation RMSE Decline Curve (n_estimators=12000)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(rmse_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"RMSE 下降曲线已保存: {rmse_curve_path}")

# 2. 报告指标表格
report_data = []
for i, idx in enumerate(milestones):
    if idx < len(rmse_history):
        report_data.append({
            'Iterations': milestone_labels[i] if i < len(milestone_labels)-1 else f"{idx} (Final)",
            'Trees': idx if i < len(milestone_labels)-1 else len(iterations),
            'Val RMSE': f"{rmse_history[idx]:.6f}"
        })

report_df = pd.DataFrame(report_data)

# 打印表格
print("\n报告指标（关键里程碑）：")
print(report_df.to_string(index=False))

# 保存表格为图片
fig, ax = plt.subplots(figsize=(8, len(report_data)*0.8 + 1))
ax.axis('off')
table = ax.table(cellText=report_df.values, colLabels=report_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)
plt.title('Report Metrics: Key Milestones (n_estimators=12000)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(report_metrics_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"报告指标表格已保存: {report_metrics_path}")

# 3. 特征重要性图（Top 20）
from xgboost import plot_importance
plt.figure(figsize=(10, 8))
plot_importance(final_model, importance_type='gain', max_num_features=20, height=0.6)
plt.title('Top 20 Feature Importance (Gain)')
plt.tight_layout()
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"特征重要性图已保存: {feature_importance_path}")

# ==================== 保存模型 ====================
print(f"\n保存最终模型到: {model_save_path}")
final_model.save_model(model_save_path)
print("模型保存完成！（格式：JSON，可用 xgboost.Booster 加载）")

print("\n所有图片和模型已生成完成！")