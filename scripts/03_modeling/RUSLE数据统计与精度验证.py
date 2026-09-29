# -*- coding: utf-8 -*-
"""
最终版：广东省500m RUSLE数据统计与精度验证脚本（NPZ格式 + 广东省边界掩膜）

确认：
- x,y 坐标为 CGCS2000 3-degree Gauss-Kruger CM 114E（即 EPSG:4547，单位：米）
- SHP 为 EPSG:4547
→ 坐标系完全一致，无需任何投影转换，掩膜可直接进行（高效准确）

功能保持不变：
- 空间掩膜仅保留广东省内像素
- 分级统计（微度~剧烈面积、水土流失面积、平均模数、总carbon）
- 输出CSV + 趋势图（带趋势线）
- 详细诊断日志（t值分布、掩膜前后像素数、预计面积等）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from shapely.vectorized import contains

# ========================== 配置 ==========================
npz_path = r"./data\Data\GTNNWR\Result_data_all.npz"
shp_path = r"./data\Data\GD_SHP\GD.shp"

output_dir = r"./data\Data\GTNNWR\运行结果"
os.makedirs(output_dir, exist_ok=True)

stats_csv = os.path.join(output_dir, "广东省500m_RUSLE_分级统计结果_1990_2024_仅广东省.csv")
validation_png = os.path.join(output_dir, "广东省水土流失面积趋势图_1990_2024.png")

PIXEL_AREA_KM2 = 0.25  # 500m × 500m

official_data = {
    2001: 11010, 2007: 20933.93, 2011: 21305, 2018: 18276,
    2019: 18009, 2020: 17636.4, 2021: 17370.47, 2022: 17108.75,
    2023: 16848.64, 2024: 16587
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ========================== 数据处理 ==========================
def process_npz(npz_path, shp_path, stats_csv):
    print("正在加载NPZ文件...")
    data = np.load(npz_path, allow_pickle=True)
    print(f"NPZ文件中包含的键：{list(data.keys())}")

    x_arr = data['x'].astype(float)
    y_arr = data['y'].astype(float)
    t_arr = data['t'].astype(int)
    erosion_arr = data['erosion'].astype(float)
    soc_arr = data['SOC_loss'].astype(float)

    print(f"总像素数：{len(t_arr):,}")
    unique_t = np.unique(t_arr)
    print(f"t值范围：{unique_t.min()} ~ {unique_t.max()}（应为0~34对应1990~2024）")
    print(f"唯一t值数量：{len(unique_t)}（理想35年）")
    print(f"对应年份：{sorted(1990 + unique_t)}")

    # 有效值过滤
    valid_mask = ~np.isnan(erosion_arr) & (erosion_arr >= 0) & ~np.isnan(soc_arr)
    x_arr = x_arr[valid_mask]
    y_arr = y_arr[valid_mask]
    t_arr = t_arr[valid_mask]
    erosion_arr = erosion_arr[valid_mask]
    soc_arr = soc_arr[valid_mask]

    print(f"有效像素数（过滤无效值后）：{len(erosion_arr):,}")

    # ========================== 空间掩膜（CRS已确认一致） ==========================
    print("\n正在加载广东省边界SHP...")
    gd_gdf = gpd.read_file(shp_path)
    print(f"SHP CRS: {gd_gdf.crs}（确认与x,y坐标系一致：EPSG:4547）")
    gd_polygon = gd_gdf.unary_union

    # bounds粗过滤
    minx, miny, maxx, maxy = gd_polygon.bounds
    bounds_mask = (x_arr >= minx) & (x_arr <= maxx) & (y_arr >= miny) & (y_arr <= maxy)
    print(f"bounds粗过滤后剩余像素：{bounds_mask.sum():,} ({bounds_mask.mean() * 100:.2f}%)")

    x_filtered = x_arr[bounds_mask]
    y_filtered = y_arr[bounds_mask]

    # 精确掩膜
    print("正在进行精确空间掩膜（vectorized contains，可能需5-20分钟...）")
    in_gd_mask = contains(gd_polygon, x_filtered, y_filtered)

    final_mask = bounds_mask.copy()
    final_mask[bounds_mask] = in_gd_mask

    print(f"\n✅ 掩膜完成！广东省内像素数：{final_mask.sum():,} ({final_mask.mean() * 100:.3f}%)")

    gd_pixels_per_year = final_mask.sum() / len(unique_t) if len(unique_t) > 0 else 0
    expected_area = gd_pixels_per_year * PIXEL_AREA_KM2
    print(f"预计年均广东省面积：{expected_area:,.0f} km²（应接近179,800 km²，误差±5%正常）")

    if final_mask.sum() == 0:
        raise ValueError("❌ 掩膜后无像素！可能原因：1. 数据范围不覆盖广东省 2. 坐标值异常")

    # 应用掩膜
    years = 1990 + t_arr[final_mask]
    erosion_arr = erosion_arr[final_mask]
    soc_arr = soc_arr[final_mask]

    # ========================== 统计 ==========================
    df = pd.DataFrame({
        'Year': years,
        'erosion': erosion_arr,
        'SOC_loss': soc_arr
    })

    bins = [0, 500, 2500, 5000, 8000, 15000, np.inf]
    labels = ['微度', '轻度', '中度', '强烈', '极强烈', '剧烈']
    df['Grade'] = pd.cut(df['erosion'], bins=bins, labels=labels, include_lowest=True)

    print("\n正在按年份分级统计...")
    grouped = df.groupby(['Year', 'Grade'], observed=False).size().reset_index(name='Pixel_Count')
    grouped['Area_km2'] = grouped['Pixel_Count'] * PIXEL_AREA_KM2

    pivot = grouped.pivot(index='Year', columns='Grade', values='Area_km2').fillna(0)
    pivot.columns = [f'{col}_Area_km2' for col in pivot.columns]
    pivot = pivot.reset_index()

    yearly = df.groupby('Year').agg(
        Total_Pixels=('erosion', 'size'),
        Total_Area_km2=('erosion', lambda x: len(x) * PIXEL_AREA_KM2),
        Mean_Erosion=('erosion', 'mean'),
        Total_SOC_loss_t=('SOC_loss', 'sum')
    ).reset_index()

    result = pd.merge(pivot, yearly, on='Year', how='outer').fillna(0)
    erosion_cols = [col for col in result.columns if
                    col.endswith('_Area_km2') and col.startswith(('轻度', '中度', '强烈', '极强烈', '剧烈'))]
    result['Water_Erosion_Area_km2'] = result[erosion_cols].sum(axis=1)

    result = result.sort_values('Year').reset_index(drop=True)
    result.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 统计完成！结果保存至：{stats_csv}")

    avg_area = result['Total_Area_km2'].mean()
    print(f"掩膜后年均总面积：{avg_area:,.0f} km²")

    return result


# ========================== 趋势图 ==========================
def plot_trend(df_stats, validation_png, official_data):
    print("\n正在生成趋势图...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.plot(df_stats['Year'], df_stats['Water_Erosion_Area_km2'], 'b-o', linewidth=3, markersize=6,
            label='500m RUSLE 计算（轻度及以上，仅广东省）')

    # 官方公报点（若有重叠年份）
    overlap_years = [y for y in official_data.keys() if y in df_stats['Year'].values]
    if overlap_years:
        off_areas = [official_data[y] for y in overlap_years]
        ax.plot(overlap_years, off_areas, 'ro', markersize=10, label='广东省水土保持公报')

    ax.set_title('广东省水土流失面积时间序列（1990-2024，仅广东省境内）', fontsize=18, fontweight='bold')
    ax.set_xlabel('年份', fontsize=14)
    ax.set_ylabel('水土流失面积 (km²)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # 趋势线
    if len(df_stats) > 1:
        z = np.polyfit(df_stats['Year'], df_stats['Water_Erosion_Area_km2'], 1)
        p = np.poly1d(z)
        ax.plot(df_stats['Year'], p(df_stats['Year']), 'r--', linewidth=2,
                label=f'趋势线（年均变化: {z[0]:+.0f} km²/年）')
        ax.legend(fontsize=12)

        # 变化幅度
        first = df_stats.iloc[0]['Water_Erosion_Area_km2']
        last = df_stats.iloc[-1]['Water_Erosion_Area_km2']
        change_pct = (last - first) / first * 100
        print(f"首末年变化：{first:,.0f} → {last:,.0f} km²（{change_pct:+.1f}%）")

    plt.tight_layout()
    plt.savefig(validation_png, dpi=400, bbox_inches='tight')
    plt.show()
    print(f"趋势图已保存：{validation_png}")


# ========================== 主程序 ==========================
def main():
    if not os.path.exists(npz_path):
        print(f"错误：找不到NPZ文件 {npz_path}")
        return
    if not os.path.exists(shp_path):
        print(f"错误：找不到SHP文件 {shp_path}")
        return

    df_stats = process_npz(npz_path, shp_path, stats_csv)
    plot_trend(df_stats, validation_png, official_data)

    print("\n🎉 全部完成！")
    print("接下来建议：")
    print("1. 检查CSV中年份是否完整（1990-2024）")
    print("2. 若有2001年后年份，可手动与公报对比精度")
    print("3. 如需更详细精度验证图表，告诉我，我再加4合1版本")


if __name__ == '__main__':
    main()