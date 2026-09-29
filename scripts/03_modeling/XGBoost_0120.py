import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import gc
import random
from scipy.stats import uniform, loguniform, randint

# ==================== 配置参数 ====================
GRID_SIZE_ORIGINAL = 500
GRID_SIZE_NEW = 1500
SAMPLES_PER_YEAR_AFTER_AGG = None  # 使用全部聚合数据
USE_LOG_TRANSFORM = True           # 对目标变量做 log 变换
SEED = 42

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
NATURE_COLORS = {
    'red': '#e64b35', 'blue': '#4dbbd5', 'green': '#00a087',
    'dark_blue': '#3c5488', 'orange': '#f39b7f'
}

random.seed(SEED)
np.random.seed(SEED)

# ==================== 路径 ====================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
output_dir = r"./data\Data\GTNNWR\Result\xgboost"
zip_path = r"./data\Data\GTNNWR\Result\xgboost.zip"
os.makedirs(output_dir, exist_ok=True)

# 1. 加载数据并创建二维 NumPy 数组
print("正在加载原始数据")
loaded = np.load(npz_path)

columns = ['x', 'y', 't', 'SOC_loss', 'erosion', 'R', 'K', 'LS', 'C', 'P']
col_idx = {name: i for i, name in enumerate(columns)}

data = np.column_stack([
    loaded['x'], loaded['y'], loaded['t'],
    loaded['SOC_loss'], loaded['erosion'],
    loaded['R'], loaded['K'], loaded['LS'],
    loaded['C'], loaded['P']
])

del loaded
gc.collect()

print(f"原始数据行数: {len(data):,}")
print(f"内存占用约: {data.nbytes / 1024**3:.2f} GB")

# 2. 清洗 NaN
print("清洗 NaN...")
valid_mask = np.all(np.isfinite(data), axis=1)
data = data[valid_mask]
print(f"清洗 NaN 后剩余行数: {len(data):,}")

# 3. 计算年份
print("计算年份...")
year = np.round(data[:, col_idx['t']] * (2024 - 1990) + 1990).astype(np.int32)
data = np.column_stack([data, year])
col_idx['year'] = data.shape[1] - 1

# 4. 计算 1500m 网格索引
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

# 5. 转为 pandas 进行聚合
print("转为 pandas 并进行分组聚合...")
df_temp = pd.DataFrame(data, columns=list(col_idx.keys()))

agg_dict = {
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
}

data_1500m = df_temp.groupby(['year', 'grid_x_new', 'grid_y_new'], as_index=False).agg(agg_dict)
print(f"1500m 分辨率聚合后总行数: {len(data_1500m):,}")

del data, df_temp
gc.collect()

# 6. 使用全部聚合数据
data_final = data_1500m.copy()
print(f"最终用于建模的数据量（全量）: {len(data_final):,} 行")

# 7. 特征工程
print("进行特征工程...")
data_final['R_K'] = data_final['R'] * data_final['K']
data_final['LS_C'] = data_final['LS'] * data_final['C']
data_final['R_LS'] = data_final['R'] * data_final['LS']
data_final['R_P'] = data_final['R'] * data_final['P']
data_final['year_sq'] = data_final['year'] ** 2

# 8. 目标变量处理
if USE_LOG_TRANSFORM:
    data_final['SOC_loss_log'] = np.log1p(data_final['SOC_loss'])
    target = 'SOC_loss_log'
else:
    target = 'SOC_loss'

features = ['R', 'K', 'LS', 'C', 'P', 'year', 'R_K', 'LS_C', 'R_LS', 'R_P', 'year_sq']

X = data_final[features].values
y = data_final[target].values
y_original = data_final['SOC_loss'].values

# 9. 数据集划分
print("\n=== 随机划分数据集 ===")
X_train, X_temp, y_train, y_temp, y_orig_train, y_orig_temp = train_test_split(
    X, y, y_original, test_size=0.3, random_state=SEED, shuffle=True
)
X_val, X_test, y_val, y_test, y_orig_val, y_orig_test = train_test_split(
    X_temp, y_temp, y_orig_temp, test_size=0.5, random_state=SEED, shuffle=True
)

print(f"训练集: {len(X_train):,} 条")
print(f"验证集: {len(X_val):,} 条")
print(f"测试集: {len(X_test):,} 条")

# 10. 使用 RandomizedSearchCV 自动调参（替代 Optuna，不需要额外安装）
print("\n=== 开始 RandomizedSearchCV 自动调参 ===")
param_dist = {
    'n_estimators': randint(1000, 5000),
    'learning_rate': loguniform(0.005, 0.1),
    'max_depth': randint(6, 12),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'reg_lambda': loguniform(0.1, 10),
    'reg_alpha': loguniform(0.01, 10)
}

# 基础模型（使用 early_stopping_rounds 兼容旧版 XGBoost）
base_model = XGBRegressor(
    random_state=SEED,
    n_jobs=-1,
    tree_method='hist'
)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=50,                  # 搜索 50 次
    scoring='neg_root_mean_squared_error',
    cv=3,                       # 3-fold CV
    verbose=2,
    random_state=SEED,
    n_jobs=-1
)

# 为了 early stopping，需要用 fit 并传入 eval_set
# RandomizedSearchCV 不直接支持 early stopping，所以我们先用默认参数训练一个带 early stopping 的模型，再用搜索结果
# 简化：直接用随机搜索找到参数，然后手动训练带 early stopping 的模型

random_search.fit(X_train, y_train)  # 先搜索（不带 early stopping）

print("最佳参数:", random_search.best_params_)
print("最佳 CV RMSE (负值):", random_search.best_score_)

# 11. 使用最佳参数训练最终模型（带 early stopping）
best_params = random_search.best_params_
best_params.update({
    'random_state': SEED,
    'n_jobs': -1,
    'tree_method': 'hist'
})

print("\n=== 使用最佳参数训练最终模型（带 early stopping） ===")
final_model = XGBRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=200,   # 兼容旧版
    verbose=100
)

# 12. 最终评估（原尺度）
def print_metrics(y_true_orig, y_pred, name):
    r2 = r2_score(y_true_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred))
    mae = mean_absolute_error(y_true_orig, y_pred)
    print(f"\n{name} (原尺度):")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE  = {mae:.2f}")

y_pred_test = final_model.predict(X_test)
if USE_LOG_TRANSFORM:
    y_pred_test = np.expm1(y_pred_test)

y_pred_val = final_model.predict(X_val)
if USE_LOG_TRANSFORM:
    y_pred_val = np.expm1(y_pred_val)

print_metrics(y_orig_test, y_pred_test, "测试集 XGBoost")
print_metrics(y_orig_val, y_pred_val, "验证集 XGBoost")

# 13~15. 出图与打包（同之前）
# ...（复制之前的特征重要性、散点图、残差图、打包代码）

# 特征重要性图
plt.figure(figsize=(10, 6))
importance = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=True)
importance.plot(kind='barh', color='teal')
plt.title('XGBoost Feature Importance (Enhanced)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 测试集散点图
plt.figure(figsize=(9, 6))
plt.scatter(y_orig_test, y_pred_test, alpha=0.4, s=15, color='green')
minv = min(y_orig_test.min(), y_pred_test.min())
maxv = max(y_orig_test.max(), y_pred_test.max())
plt.plot([minv, maxv], [minv, maxv], '--', color='darkblue', lw=2)
plt.xlabel('True SOC_loss')
plt.ylabel('Predicted SOC_loss')
plt.title('True vs Predicted - SOC_loss (Test set)')
plt.savefig(os.path.join(output_dir, 'scatter_test_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 残差图
residuals = y_orig_test - y_pred_test
plt.figure(figsize=(9, 6))
plt.scatter(y_pred_test, residuals, alpha=0.4, s=15, color='orange')
plt.axhline(0, color='black', linestyle='--', lw=1.5)
plt.xlabel('Predicted SOC_loss')
plt.ylabel('Residuals')
plt.title('Residual Plot - SOC_loss (Test set)')
plt.savefig(os.path.join(output_dir, 'residual_test_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 打包
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            fp = os.path.join(root, file)
            arc = os.path.relpath(fp, output_dir)
            zf.write(fp, arc)

print(f"\n增强版模型训练完成！")
print(f"结果保存至: {output_dir}")