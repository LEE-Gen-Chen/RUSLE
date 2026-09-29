import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 官方数据（仅2018年及以后） ====================
official_data = {
    2018: 18276,
    2019: 18009,
    2020: 17636.4,
    2021: 17370.47,
    2022: 17108.75,
    2023: 16848.64,
    2024: 16587
}

# ==================== 轻度权重（最终推荐值） ====================
LIGHT_WEIGHT = 0.95   # 推荐锁定此值，或0.9也可


# ==================== 读取并准备数据 ====================
def load_and_prepare_data(your_results_path):
    df = pd.read_csv(your_results_path)

    df['Calculated_Erosion_Area'] = (
        df['轻度_Area_km2'] * LIGHT_WEIGHT +
        df['中度_Area_km2'] +
        df['强烈_Area_km2'] +
        df['极强烈_Area_km2'] +
        df['剧烈_Area_km2']
    )

    # 计算轻度占比（参考）
    total_cols = ['轻度_Area_km2', '中度_Area_km2', '强烈_Area_km2', '极强烈_Area_km2', '剧烈_Area_km2']
    df['Total_Erosion'] = df[total_cols].sum(axis=1)
    df['Light_Proportion_%'] = (df['轻度_Area_km2'] / df['Total_Erosion'] * 100).round(2)

    official_years = list(official_data.keys())
    df_official_years = df[df['Year'].isin(official_years)].copy()
    df_official_years['Official_Erosion_Area'] = df_official_years['Year'].map(official_data)

    print("\n各年轻度侵蚀占比（参考）：")
    print(df_official_years[['Year', 'Light_Proportion_%']].to_string(index=False))

    return df_official_years, df


# ==================== 精度验证 ====================
def validate_accuracy(df_validation):
    if df_validation.empty:
        print("没有可用于验证的数据")
        return None, None

    recent_years = [y for y in df_validation['Year'] if y >= 2018]
    df_recent = df_validation[df_validation['Year'].isin(recent_years)].copy()

    if df_recent.empty:
        print("没有2018年及以后的验证数据")
        return None, None

    calculated = df_recent['Calculated_Erosion_Area'].values
    official = df_recent['Official_Erosion_Area'].values
    years = df_recent['Year'].values

    mae = mean_absolute_error(official, calculated)
    rmse = np.sqrt(mean_squared_error(official, calculated))
    r2 = r2_score(official, calculated)

    relative_errors = [abs(c - o) / o * 100 if o != 0 else 0 for c, o in zip(calculated, official)]
    absolute_errors = [abs(c - o) for c, o in zip(calculated, official)]

    avg_relative_error = np.mean(relative_errors)
    avg_absolute_error = np.mean(absolute_errors)

    all_abs_errors = []
    all_rel_errors = []
    for i in range(len(df_validation)):
        o = df_validation['Official_Erosion_Area'].iloc[i]
        c = df_validation['Calculated_Erosion_Area'].iloc[i]
        abs_e = abs(c - o)
        rel_e = (abs_e / o) * 100 if o != 0 else 0
        all_abs_errors.append(abs_e)
        all_rel_errors.append(rel_e)

    validation_results = pd.DataFrame({
        'Year': df_validation['Year'],
        'Official_Area': df_validation['Official_Erosion_Area'],
        'Calculated_Area': df_validation['Calculated_Erosion_Area'],
        'Abs_Error_km2': all_abs_errors,
        'Rel_Error_pct': all_rel_errors
    })

    print("=" * 60)
    print(f"精度验证报告（2018+ 数据 / 轻度权重 = {LIGHT_WEIGHT}）")
    print("=" * 60)
    print(f"验证年份数量: {len(recent_years)}")
    print(f"平均绝对误差 (MAE): {mae:,.2f} km²")
    print(f"均方根误差 (RMSE): {rmse:,.2f} km²")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"平均相对误差: {avg_relative_error:.2f}%")
    print(f"平均绝对误差: {avg_absolute_error:,.2f} km²")

    print(f"\n详细验证结果：")
    print(validation_results.round(2).to_string(index=False))

    metrics = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'Avg_Relative_Error': avg_relative_error,
        'Avg_Absolute_Error': avg_absolute_error,
        'Recent_Years': recent_years
    }

    return validation_results, metrics


# ==================== 绘图函数 ====================
def plot_validation_results(validation_results, metrics):
    recent_years = metrics.get('Recent_Years', [])
    if not recent_years:
        return

    df_plot = validation_results[validation_results['Year'].isin(recent_years)].copy()

    years = df_plot['Year']
    official = df_plot['Official_Area']
    calculated = df_plot['Calculated_Area']
    rel_errors = df_plot['Rel_Error_pct']
    abs_errors = df_plot['Abs_Error_km2']

    # ────────────────────────────────
    # 推荐论文级参数设置
    # ────────────────────────────────
    FIG_SIZE = (18, 11)          # 整体画布大小（宽×高）
    DPI = 400                    # 保存分辨率
    TITLE_SIZE = 16
    LABEL_SIZE = 12
    TICK_SIZE = 13
    LEGEND_SIZE = 13
    LINE_WIDTH = 3.0
    MARKER_SIZE = 9
    BAR_WIDTH = 0.7

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIG_SIZE)

    # 图1：面积对比
    ax1.plot(years, official, 'ro-', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='官方数据')
    ax1.plot(years, calculated, 'bo--', linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
             label=f'计算结果')
    ax1.set_ylabel('水土流失面积 (km$²$)', fontsize=LABEL_SIZE)
    ax1.set_xlabel('年份', fontsize=LABEL_SIZE)
    ax1.set_title('水土流失面积对比 ', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.legend(fontsize=LEGEND_SIZE, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)

    # 添加垂直误差线（可选，保持但调细）
    for i, year in enumerate(years):
        ax1.plot([year, year], [official.iloc[i], calculated.iloc[i]], 'k-', alpha=0.4, linewidth=1.2)

    # 图2：相对误差柱状图
    ax2.bar(years, rel_errors, color='orange', alpha=0.8, width=BAR_WIDTH, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=0, color='black', linewidth=1.2)
    ax2.set_ylabel('相对误差 (%)', fontsize=LABEL_SIZE)
    ax2.set_xlabel('年份', fontsize=LABEL_SIZE)
    ax2.set_title('相对误差分析 ', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.tick_params(axis='both', labelsize=TICK_SIZE)

    avg_error = metrics['Avg_Relative_Error']
    ax2.axhline(y=avg_error, color='red', linestyle='--', linewidth=2.5,
                label=f'平均误差: {avg_error:.1f}%')
    ax2.legend(fontsize=LEGEND_SIZE, loc='upper right')
    # 图3：散点图（只显示数据点 + 1:1参考线，并在每个点旁边标注年份）
    ax3.scatter(official, calculated,
                s=120, alpha=0.85, color='green',
                edgecolor='black', linewidth=1.0,
                label='数据点')

    # 画 1:1 参考线
    min_val = min(official.min(), calculated.min())
    max_val = max(official.max(), calculated.max())
    ax3.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2.5, label='1:1 参考线')

    # ─────────────── 新增：为每个点添加年份标签 ───────────────
    for i, year in enumerate(years):
        # 在点右侧上方 0.5% 的偏移位置放置年份（可根据需要调整偏移量）
        ax3.text(
            official.iloc[i] + (max_val - min_val) * 0.02,  # x 偏移：向右一点
            calculated.iloc[i] + (max_val - min_val) * -0.03,  # y 偏移：向上一点
            str(int(year)),  # 显示年份（整数形式）
            fontsize=11,  # 字体稍小，避免拥挤
            color='black',
            ha='left',  # 左对齐
            va='bottom',  # 底部对齐
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')  # 白色半透明背景框
        )

    # 坐标轴标签（使用 $²$ 显示上标）
    ax3.set_xlabel('官方水土流失面积 (km$^2$)', fontsize=LABEL_SIZE)
    ax3.set_ylabel('计算水土流失面积 (km$^2$)', fontsize=LABEL_SIZE)

    # 标题（推荐使用 mathtext 显示 R²）
    # ax3.set_title(r'散点图 (R$^2$ = {:.4f})'.format(metrics["R2"]),
    #               fontsize=TITLE_SIZE, fontweight='bold')

    ax3.legend(fontsize=LEGEND_SIZE, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', labelsize=TICK_SIZE)

    # 图4：绝对误差直方图
    ax4.hist(abs_errors, bins=8, color='purple', alpha=0.8, edgecolor='black', linewidth=0.8)
    ax4.axvline(x=metrics['MAE'], color='red', linestyle='--', linewidth=2.5,
                label=f'MAE: {metrics["MAE"]:,.0f} km$²$')
    ax4.axvline(x=metrics['RMSE'], color='orange', linestyle='--', linewidth=2.5,
                label=f'RMSE: {metrics["RMSE"]:,.0f} km$²$')
    ax4.set_xlabel('绝对误差 (km$²$)', fontsize=LABEL_SIZE)
    ax4.set_ylabel('频次', fontsize=LABEL_SIZE)
    ax4.set_title('绝对误差分布 ', fontsize=TITLE_SIZE, fontweight='bold')
    ax4.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.tick_params(axis='both', labelsize=TICK_SIZE)

    # 优化整体布局
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=3.0)
    # 如果还是觉得拥挤，可以再微调边距
    # plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)

    save_path = rf"./data\结果\精度验证分析_2018后_轻度x{LIGHT_WEIGHT}_优化版.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"图表已保存至: {save_path}")


# ==================== 主程序 ====================
def main():
    your_results_path = r"./data\结果\土壤侵蚀统计结果_矢量掩膜.csv"

    if not os.path.exists(your_results_path):
        alt_path = r"./data\结果\土壤侵蚀统计结果分级.csv"
        if os.path.exists(alt_path):
            your_results_path = alt_path
            print(f"使用替代文件: {alt_path}")
        else:
            print("找不到计算结果文件")
            return

    print("加载你的计算结果...")
    df_validation, df_full = load_and_prepare_data(your_results_path)

    if df_validation.empty:
        print("没有与官方数据重叠的年份")
        return

    validation_results, metrics = validate_accuracy(df_validation)

    if metrics is not None:
        plot_validation_results(validation_results, metrics)

        output_path = rf"./data\结果\精度验证结果_2018后_轻度x{LIGHT_WEIGHT}.csv"
        validation_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n验证结果已保存至: {output_path}")


if __name__ == "__main__":
    main()