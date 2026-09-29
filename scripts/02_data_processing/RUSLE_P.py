import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import fiona
from shapely.geometry import mapping
import geopandas as gpd


def load_mask_shapefile(shapefile_path):
    """
    加载掩膜矢量文件
    """
    try:
        # 使用geopandas读取矢量文件
        gdf = gpd.read_file(shapefile_path)
        print(f"成功加载矢量文件: {shapefile_path}")
        print(f"矢量文件CRS: {gdf.crs}")
        print(f"要素数量: {len(gdf)}")
        return gdf
    except Exception as e:
        print(f"加载矢量文件失败: {str(e)}")
        return None


YEAR_START = 1990
YEAR_END = 2024

def generate_p_factor_raster(input_folder, output_folder, mask_shapefile, year_range=(YEAR_START, YEAR_END + 1)):
    """
    根据CLCD土地分类数据生成P因子栅格数据，严格限制在矢量范围内

    参数:
    input_folder: CLCD数据文件夹路径
    output_folder: 输出P因子栅格文件夹路径
    mask_shapefile: 掩膜矢量文件路径
    year_range: 年份范围 (起始年, 结束年+1)
    """

    # P因子映射字典
    p_factor_mapping = {
        1: 0.4,  # Cropland -> 0.4
        2: 1.0,  # Forest -> 1
        3: 1.0,  # Shrub -> 1
        4: 1.0,  # Grassland -> 1
        5: 0.0,  # Water -> 0
        6: 0.0,  # Snow/Ice -> 0
        7: 1.0,  # Barren -> 1
        8: 0.0,  # Impervious -> 0
        9: 0.0  # Wetland -> 0
    }

    # 类别名称映射
    class_names = {
        1: "耕地",
        2: "森林",
        3: "灌木",
        4: "草地",
        5: "水域",
        6: "冰雪",
        7: "裸地",
        8: "不透水面",
        9: "湿地"
    }

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 加载掩膜矢量
    mask_gdf = load_mask_shapefile(mask_shapefile)
    if mask_gdf is None:
        print("无法加载掩膜矢量文件，退出处理")
        return []

    # 获取矢量几何
    geometries = mask_gdf.geometry.values
    shapes = [mapping(geom) for geom in geometries]

    # 存储统计信息
    stats_data = []

    # 使用tqdm显示进度条
    for year in tqdm(range(year_range[0], year_range[1]), desc="处理年份"):
        input_file = os.path.join(input_folder, f"CLCD{year}.tif")
        output_file = os.path.join(output_folder, f"P{year}.tif")

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件 {input_file} 不存在，跳过该年份")
            continue

        try:
            # 读取原始CLCD数据并进行掩膜
            with rasterio.open(input_file) as src:
                # 应用矢量掩膜
                try:
                    masked_data, masked_transform = mask(
                        src,
                        shapes,
                        crop=False,  # 不裁剪，保持原图范围但只保留掩膜区域内的数据
                        filled=True,  # 将掩膜外的值填充为nodata
                        nodata=0  # 设置掩膜外的值为0
                    )
                    landuse_data = masked_data[0]  # 获取第一个波段

                    # 更新profile
                    profile = src.profile.copy()
                    profile.update({
                        'transform': masked_transform,
                        'height': landuse_data.shape[0],
                        'width': landuse_data.shape[1]
                    })

                except Exception as e:
                    print(f"掩膜处理失败 {year}: {str(e)}，使用原始数据")
                    landuse_data = src.read(1)
                    profile = src.profile.copy()

                # 创建P因子数组
                p_factor_data = np.zeros_like(landuse_data, dtype=np.float32)

                # 统计各类别面积和P值
                year_stats = {'年份': year}

                # 创建有效像素掩膜（非0且非nodata）
                valid_mask = (landuse_data > 0)
                if src.nodata is not None:
                    valid_mask = valid_mask & (landuse_data != src.nodata)

                total_pixels = np.sum(valid_mask)  # 有效像素数

                # 计算加权平均P值
                weighted_p_sum = 0
                total_weighted_pixels = 0

                for class_id, p_value in p_factor_mapping.items():
                    mask_class = (landuse_data == class_id) & valid_mask
                    p_factor_data[mask_class] = p_value

                    # 统计各类别像素数量
                    class_pixels = np.sum(mask_class)

                    # 计算加权P值
                    weighted_p_sum += class_pixels * p_value
                    total_weighted_pixels += class_pixels

                    # 计算面积比例
                    if total_pixels > 0:
                        area_ratio = class_pixels / total_pixels * 100
                    else:
                        area_ratio = 0

                    year_stats[f'{class_names[class_id]}_面积比例'] = area_ratio
                    year_stats[f'{class_names[class_id]}_像素数'] = class_pixels

                # 计算加权平均P值
                if total_weighted_pixels > 0:
                    weighted_avg_p = weighted_p_sum / total_weighted_pixels
                else:
                    weighted_avg_p = 0

                year_stats['加权平均P值'] = weighted_avg_p
                year_stats['总有效像素'] = total_pixels
                year_stats['矢量范围内像素'] = total_weighted_pixels

                # 更新输出文件的元数据
                profile.update({
                    'dtype': rasterio.float32,
                    'nodata': -9999.0,
                    'count': 1
                })

                # 将掩膜外的区域设置为nodata
                p_factor_data[~valid_mask] = -9999.0

                # 写入P因子栅格文件
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(p_factor_data, 1)

                stats_data.append(year_stats)

        except Exception as e:
            print(f"处理 {year} 年数据时出错: {str(e)}")
            continue

    return stats_data


def create_seaborn_visualizations(stats_data, output_folder):
    """使用seaborn创建可视化图表"""

    if not stats_data:
        print("没有统计数据可用于可视化")
        return

    # 创建输出文件夹
    vis_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame(stats_data)

    # 设置seaborn样式
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 1. 绘制加权平均P值变化趋势
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='年份', y='加权平均P值', marker='o', linewidth=2.5)
    plt.title('P Factor Trend (1990-2024)', fontsize=14, fontweight='bold')
    plt.ylabel('加权平均P值')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, '加权平均P值变化趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 绘制各类土地面积比例变化趋势
    area_columns = [col for col in df.columns if '面积比例' in col]
    if area_columns:
        # 准备数据用于seaborn
        area_data = []
        for _, row in df.iterrows():
            for col in area_columns:
                land_type = col.replace('_面积比例', '')
                area_data.append({
                    '年份': row['年份'],
                    '土地类型': land_type,
                    '面积比例': row[col]
                })

        area_df = pd.DataFrame(area_data)

        plt.figure(figsize=(14, 8))
        sns.lineplot(data=area_df, x='年份', y='面积比例', hue='土地类型',
                     marker='o', linewidth=2, markersize=4)
        plt.title('Land Use Area Ratio Trend (1990-2024)', fontsize=14, fontweight='bold')
        plt.ylabel('面积比例 (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='土地类型')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_folder, '土地利用面积比例变化.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 绘制P值分布热力图（按年份和土地类型）
    p_values = {
        '耕地': 0.4, '森林': 1.0, '灌木': 1.0, '草地': 1.0,
        '水域': 0.0, '冰雪': 0.0, '裸地': 1.0, '不透水面': 0.0, '湿地': 0.0
    }

    # 创建P值热力图数据
    heatmap_data = []
    for _, row in df.iterrows():
        for land_type, p_val in p_values.items():
            area_col = f'{land_type}_面积比例'
            if area_col in row:
                heatmap_data.append({
                    '年份': row['年份'],
                    '土地类型': land_type,
                    'P值': p_val,
                    '面积加权贡献': row[area_col] * p_val / 100  # 标准化
                })

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot_table(index='土地类型', columns='年份', values='面积加权贡献', aggfunc='mean')

        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_df, annot=False, cmap='YlOrRd', linewidths=0.5,
                    cbar_kws={'label': '面积加权P值贡献'})
        plt.title('P Value Contribution Heatmap (1990-2024)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_folder, 'P值贡献热力图.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. 保存统计结果为CSV
    stats_csv_path = os.path.join(output_folder, "P因子统计结果.csv")
    df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存至: {stats_csv_path}")

    print(f"所有可视化图表已保存至: {vis_folder}")


def batch_process_clcd_data():
    """批量处理CLCD数据的主函数"""

    # 设置路径
    input_folder = r"./data/landcover"
    output_folder = r"./data/P_output"
    mask_shapefile = r"./data/boundary.shp"  # 矢量掩膜文件

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 {input_folder} 不存在")
        return

    # 检查掩膜文件是否存在
    if not os.path.exists(mask_shapefile):
        print(f"错误: 掩膜文件 {mask_shapefile} 不存在")
        return

    # 生成P因子栅格
    print("开始生成P因子栅格数据...")
    stats_data = generate_p_factor_raster(input_folder, output_folder, mask_shapefile)
    print("P因子栅格生成完成!")

    # 创建可视化
    if stats_data:
        print("开始创建可视化图表...")
        create_seaborn_visualizations(stats_data, output_folder)
        print("可视化完成!")
    else:
        print("没有生成统计数据，跳过可视化步骤")


def validate_p_values():
    """验证P值映射是否正确"""
    p_factor_mapping = {
        1: 0.4,  # Cropland
        2: 1.0,  # Forest
        3: 1.0,  # Shrub
        4: 1.0,  # Grassland
        5: 0.0,  # Water
        6: 0.0,  # Snow/Ice
        7: 1.0,  # Barren
        8: 0.0,  # Impervious
        9: 0.0  # Wetland
    }

    print("P因子映射关系验证:")
    print("ID\t类别\t\tP值")
    print("-" * 30)
    class_names = {
        1: "Cropland",
        2: "Forest",
        3: "Shrub",
        4: "Grassland",
        5: "Water",
        6: "Snow/Ice",
        7: "Barren",
        8: "Impervious",
        9: "Wetland"
    }

    for class_id, p_value in p_factor_mapping.items():
        class_name = class_names.get(class_id, "Unknown")
        print(f"{class_id}\t{class_name:<12}\t{p_value}")


if __name__ == "__main__":
    # 验证P值映射
    validate_p_values()
    print("\n")

    # 执行批量处理
    batch_process_clcd_data()