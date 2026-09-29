import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

# ==================== 字体设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

# ==================== 官方数据 ====================
official_data = {
    2001: 11010,
    2007: 20933.93,
    2011: 21305,
    2018: 18276,
    2019: 18009,
    2020: 17636.4,
    2021: 17370.47,
    2022: 17108.75,
    2023: 16848.64,
    2024: 16587
}

EROSION_COLS = [
    '轻度_Area_km2',
    '中度_Area_km2',
    '强烈_Area_km2',
    '极强烈_Area_km2',
    '剧烈_Area_km2'
]

# ==================== 数据读取 ====================
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df['Year'] = df['Year'].astype(int)

    df_val = df[df['Year'].isin(official_data.keys())].copy()
    df_val['Official_Erosion_Area'] = df_val['Year'].map(official_data)

    return df_val, df


# ==================== 多指标计算（安全版） ====================
def calc_all_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # ---- R2：单样本不计算（避免数学错误）----
    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    # ---- 分类指标（用于结构一致性评估）----
    threshold = np.median(y_true)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return mae, rmse, r2, mape, acc, prec, rec, f1


# ==================== 自动识别最佳口径 ====================
def auto_select_best_scheme(df_val):
    schemes = {
        '轻度+': EROSION_COLS,
        '中度+': EROSION_COLS[1:],
        '强烈+': EROSION_COLS[2:]
    }

    scheme_metrics = []
    best_rows = []

    for _, row in df_val.iterrows():
        year = row['Year']
        official = np.array([row['Official_Erosion_Area']])

        best_rmse = np.inf
        best_scheme = None
        best_value = None

        for name, cols in schemes.items():
            pred_val = row[cols].sum()
            pred = np.array([pred_val])

            mae, rmse, r2, mape, acc, prec, rec, f1 = calc_all_metrics(official, pred)

            scheme_metrics.append({
                'Year': year,
                'Scheme': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE_%': mape,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_scheme = name
                best_value = pred_val

        best_rows.append({
            'Year': year,
            'Best_Scheme': best_scheme,
            'Calculated_Erosion_Area': best_value
        })

    return pd.DataFrame(best_rows), pd.DataFrame(scheme_metrics)


# ==================== 精度验证 ====================
def validate_accuracy(df):
    calc = df['Calculated_Erosion_Area'].values
    off = df['Official_Erosion_Area'].values

    abs_err = np.abs(calc - off)
    rel_err = abs_err / off * 100

    results = pd.DataFrame({
        'Year': df['Year'],
        'Official_Area': off,
        'Calculated_Area': calc,
        'Absolute_Error': abs_err,
        'Relative_Error_%': rel_err
    })

    metrics = {
        'MAE': mean_absolute_error(off, calc),
        'RMSE': np.sqrt(mean_squared_error(off, calc)),
        'R2': r2_score(off, calc),
        'Avg_Relative_Error': rel_err.mean(),
        'Avg_Absolute_Error': abs_err.mean()
    }

    print("=" * 60)
    print("精度验证报告（自动最佳口径）")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if 'R2' in k else f"{k}: {v:,.2f}")

    print("\n详细结果：")
    print(results.round(2).to_string(index=False))

    return results, metrics


# ==================== 绘图 ====================
def plot_validation_results(df, res, metrics):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 面积对比
    axs[0, 0].plot(df['Year'], df['Official_Erosion_Area'], 'ro-', label='官方')
    axs[0, 0].plot(df['Year'], df['Calculated_Erosion_Area'], 'bo--', label='计算')
    axs[0, 0].set_title('水土流失面积对比')
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)

    # 相对误差
    axs[0, 1].bar(res['Year'], res['Relative_Error_%'])
    axs[0, 1].axhline(metrics['Avg_Relative_Error'], color='r', linestyle='--',
                      label=f"均值 {metrics['Avg_Relative_Error']:.1f}%")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)

    # 回归
    axs[1, 0].scatter(res['Official_Area'], res['Calculated_Area'])
    lim = [res[['Official_Area','Calculated_Area']].min().min(),
           res[['Official_Area','Calculated_Area']].max().max()]
    axs[1, 0].plot(lim, lim, 'r--')
    axs[1, 0].set_title(f'回归分析 (R2={metrics["R2"]:.3f})')
    axs[1, 0].grid(alpha=0.3)

    # 误差分布
    axs[1, 1].hist(res['Absolute_Error'], bins=8, edgecolor='black')
    axs[1, 1].axvline(metrics['MAE'], color='r', linestyle='--', label='MAE')
    axs[1, 1].axvline(metrics['RMSE'], color='orange', linestyle='--', label='RMSE')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"./data\结果\精度验证分析.png", dpi=300)
    plt.show()


# ==================== 主程序 ====================
def main():
    path = r"./data\结果\土壤侵蚀统计结果_矢量掩膜.csv"
    if not os.path.exists(path):
        path = r"./data\结果\土壤侵蚀统计结果分级.csv"

    df_val, df_full = load_and_prepare_data(path)

    # 自动识别最佳口径
    best_df, scheme_metrics = auto_select_best_scheme(df_val)
    df_val = df_val.merge(best_df, on='Year', how='left')

    # 保存多指标结果
    scheme_metrics.to_csv(
        r"./data\结果\逐年最佳侵蚀口径_多指标.csv",
        index=False,
        encoding='utf-8-sig'
    )

    # 精度验证
    res, metrics = validate_accuracy(df_val)
    plot_validation_results(df_val, res, metrics)

    # 保存结果
    res.to_csv(
        r"./data\结果\精度验证结果.csv",
        index=False,
        encoding='utf-8-sig'
    )

    print("\n逐年最佳侵蚀口径：")
    print(best_df[['Year', 'Best_Scheme']].to_string(index=False))


if __name__ == "__main__":
    main()
