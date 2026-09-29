import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# CLCD分类系统
CLCD_CLASSES = {
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


def load_and_mask_raster(raster_path, shapefile_path):
    """加载栅格数据并用矢量文件掩膜"""
    try:
        # 读取矢量文件
        gdf = gpd.read_file(shapefile_path)

        # 读取栅格数据
        with rasterio.open(raster_path) as src:
            # 确保矢量文件和栅格数据的坐标系一致
            gdf = gdf.to_crs(src.crs)

            # 应用掩膜
            out_image, out_transform = mask(src, gdf.geometry, crop=True, filled=True, nodata=255)
            out_meta = src.meta.copy()

            # 更新元数据
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": 255
            })

        return out_image[0], out_meta  # 返回第一个波段

    except Exception as e:
        print(f"加载栅格数据出错 {raster_path}: {e}")
        return None, None


def calculate_transition_matrix(start_year, end_year, shapefile_path, base_path):
    """计算两个年份之间的土地利用转移矩阵"""

    # 构建文件路径
    start_raster_path = f"{base_path}/CLCD{start_year}.tif"
    end_raster_path = f"{base_path}/CLCD{end_year}.tif"

    print(f"正在计算 {start_year}-{end_year} 转移矩阵...")

    # 加载起始年和结束年数据
    start_data, start_meta = load_and_mask_raster(start_raster_path, shapefile_path)
    end_data, end_meta = load_and_mask_raster(end_raster_path, shapefile_path)

    if start_data is None or end_data is None:
        print(f"无法加载 {start_year} 或 {end_year} 年数据，跳过该时间段")
        return None, None

    # 如果形状不一致，尝试调整
    if start_data.shape != end_data.shape:
        print(f"警告: {start_year} 和 {end_year} 数据形状不一致")
        print(f"{start_year} 形状: {start_data.shape}")
        print(f"{end_year} 形状: {end_data.shape}")

        # 尝试找到最小公共形状
        min_height = min(start_data.shape[0], end_data.shape[0])
        min_width = min(start_data.shape[1], end_data.shape[1])

        # 裁剪到最小公共形状
        start_data = start_data[:min_height, :min_width]
        end_data = end_data[:min_height, :min_width]

        print(f"裁剪后形状: {start_data.shape}")

    # 获取NoData值
    start_nodata = start_meta.get('nodata', 255)
    end_nodata = end_meta.get('nodata', 255)

    # 确保数据类型为整数
    start_data = start_data.astype(np.int32)
    end_data = end_data.astype(np.int32)

    # 获取有效像素（非NoData且在CLCD分类范围内）
    valid_mask = ((start_data != start_nodata) &
                  (end_data != end_nodata) &
                  (start_data >= 1) & (start_data <= 9) &
                  (end_data >= 1) & (end_data <= 9))

    # 检查是否有有效数据
    if not np.any(valid_mask):
        print(f"警告: {start_year}-{end_year} 时间段没有有效数据")
        return None, None

    # 提取有效数据
    start_vals = start_data[valid_mask].flatten()
    end_vals = end_data[valid_mask].flatten()

    # 确保数据类型正确
    start_vals = start_vals.astype(np.int32)
    end_vals = end_vals.astype(np.int32)

    # 初始化9x9转移矩阵（对应9个土地类型）
    n_classes = 9
    transition_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    # 填充转移矩阵 - 使用更安全的方法
    for i in range(len(start_vals)):
        try:
            start_type = int(start_vals[i]) - 1  # 转换为0-based索引
            end_type = int(end_vals[i]) - 1  # 转换为0-based索引

            # 确保索引在有效范围内
            if 0 <= start_type < n_classes and 0 <= end_type < n_classes:
                transition_matrix[start_type, end_type] += 1
        except (ValueError, IndexError) as e:
            # 跳过无效值
            continue

    # 创建标签列表
    labels = [CLCD_CLASSES[i] for i in range(1, 10)]

    return transition_matrix, labels


def calculate_markov_matrix(transition_matrix):
    """计算Markov转移概率矩阵"""
    # 计算每行的和（每个起始类型的总像素数）
    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    # 避免除零错误
    row_sums[row_sums == 0] = 1

    # 计算转移概率
    markov_matrix = transition_matrix / row_sums

    return markov_matrix


def save_transition_matrix_to_csv(transition_matrix, labels, start_year, end_year, output_dir):
    """保存转移矩阵到CSV文件"""

    # 创建DataFrame
    df = pd.DataFrame(transition_matrix,
                      index=labels,
                      columns=labels)

    # 添加行和列的总计
    df['Row_Total'] = df.sum(axis=1)
    df.loc['Column_Total'] = df.sum(axis=0)

    # 保存到CSV
    csv_filename = f"{output_dir}/transition_matrix_{start_year}_{end_year}.csv"
    df.to_csv(csv_filename, encoding='utf-8-sig')

    print(f"转移矩阵已保存至: {csv_filename}")

    return df


def plot_transition_heatmap(transition_matrix, labels, start_year, end_year, output_dir):
    """绘制转移矩阵热力图并保存为TIFF格式"""

    # 创建DataFrame用于绘图
    plot_df = pd.DataFrame(transition_matrix, index=labels, columns=labels)

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 使用seaborn绘制热力图
    ax = sns.heatmap(plot_df,
                     annot=True,
                     fmt=',.0f',
                     cmap='YlOrRd',
                     cbar_kws={'label': 'Pixel Count'},
                     square=True)

    plt.title(f'Land Use Transition Matrix: {start_year}-{end_year}', fontsize=16, fontweight='bold')
    plt.xlabel(f'Land Use Type ({end_year})', fontsize=12)
    plt.ylabel(f'Land Use Type ({start_year})', fontsize=12)

    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 保存为TIFF格式
    tiff_filename = f"{output_dir}/transition_heatmap_{start_year}_{end_year}.tif"
    plt.savefig(tiff_filename, dpi=300, format='tiff', bbox_inches='tight')
    plt.close()

    print(f"热力图已保存至: {tiff_filename}")


def print_transition_summary(transition_matrix, labels, start_year, end_year):
    """打印转移矩阵摘要信息"""

    print(f"\n{'=' * 60}")
    print(f"土地利用转移矩阵摘要: {start_year}-{end_year}")
    print(f"{'=' * 60}")

    total_pixels = transition_matrix.sum()
    print(f"总有效像素数: {total_pixels:,}")

    # 计算不变像素和变化像素
    unchanged_pixels = np.trace(transition_matrix)
    changed_pixels = total_pixels - unchanged_pixels
    change_percentage = (changed_pixels / total_pixels) * 100

    print(f"未变化像素: {unchanged_pixels:,} ({unchanged_pixels / total_pixels * 100:.2f}%)")
    print(f"发生变化像素: {changed_pixels:,} ({change_percentage:.2f}%)")

    # 打印主要转移（前10个）
    print(f"\n主要土地利用转移 (前10个):")
    print("-" * 50)

    # 获取非对角线元素并排序
    transitions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and transition_matrix[i, j] > 0:
                transitions.append((i, j, transition_matrix[i, j]))

    # 按转移量降序排列
    transitions.sort(key=lambda x: x[2], reverse=True)

    for i, j, count in transitions[:10]:
        percentage = (count / total_pixels) * 100
        print(f"{labels[i]} → {labels[j]}: {count:,} 像素 ({percentage:.2f}%)")

    # 打印转移矩阵
    print(f"\n完整转移矩阵:")
    print("-" * 50)

    # 创建格式化输出
    header = ["From/To"] + labels
    print("\t".join(header))

    for i, label_from in enumerate(labels):
        row = [label_from]
        for j in range(len(labels)):
            row.append(f"{transition_matrix[i, j]:,}")
        print("\t".join(row))


def analyze_all_periods(years, shapefile_path, base_path, output_dir):
    """分析所有时间段的转移矩阵"""

    all_results = {}

    for i in range(len(years) - 1):
        start_year = years[i]
        end_year = years[i + 1]

        print(f"\n{'#' * 80}")
        print(f"处理时间段: {start_year} - {end_year}")
        print(f"{'#' * 80}")

        # 计算转移矩阵
        transition_matrix, labels = calculate_transition_matrix(start_year, end_year, shapefile_path, base_path)

        if transition_matrix is None:
            print(f"无法计算 {start_year}-{end_year} 转移矩阵，跳过该时间段")
            continue

        # 打印结果
        print_transition_summary(transition_matrix, labels, start_year, end_year)

        # 保存到CSV
        csv_df = save_transition_matrix_to_csv(transition_matrix, labels, start_year, end_year, output_dir)

        # 绘制并保存热力图
        plot_transition_heatmap(transition_matrix, labels, start_year, end_year, output_dir)

        # 计算Markov矩阵
        markov_matrix = calculate_markov_matrix(transition_matrix)

        # 保存Markov矩阵到CSV
        markov_df = pd.DataFrame(markov_matrix,
                                 index=labels,
                                 columns=labels)
        markov_filename = f"{output_dir}/markov_matrix_{start_year}_{end_year}.csv"
        markov_df.to_csv(markov_filename, encoding='utf-8-sig')
        print(f"Markov矩阵已保存至: {markov_filename}")

        # 存储结果
        all_results[f"{start_year}_{end_year}"] = {
            'transition_matrix': transition_matrix,
            'markov_matrix': markov_matrix,
            'labels': labels
        }

        print(f"\n完成时间段: {start_year} - {end_year}")

    return all_results


def main():
    """主函数"""

    # 设置文件路径
    base_path = r"./data\CLCD\project"
    shapefile_path = r"./data\Geoscene\GD.shp"
    output_dir = r"./data\CLCD\project"

    # 定义分析的时间段
    years = [1990, 2024]

    print("开始计算多时间段土地利用转移矩阵...")
    print(f"分析时间段: {years}")
    print(f"输出目录: {output_dir}")

    # 分析所有时间段
    all_results = analyze_all_periods(years, shapefile_path, base_path, output_dir)

    print(f"\n{'=' * 80}")
    print("所有时间段分析完成!")
    print(f"共分析了 {len(all_results)} 个时间段")

    # 生成汇总报告
    if all_results:
        print(f"\n生成汇总报告...")
        summary_data = []

        for period, result in all_results.items():
            start_year, end_year = period.split('_')
            transition_matrix = result['transition_matrix']

            total_pixels = transition_matrix.sum()
            unchanged_pixels = np.trace(transition_matrix)
            changed_pixels = total_pixels - unchanged_pixels
            change_percentage = (changed_pixels / total_pixels) * 100

            summary_data.append({
                'Period': f"{start_year}-{end_year}",
                'Total_Pixels': total_pixels,
                'Unchanged_Pixels': unchanged_pixels,
                'Changed_Pixels': changed_pixels,
                'Change_Percentage': change_percentage
            })

        # 保存汇总报告
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{output_dir}/transition_summary_all_periods.csv"
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        print(f"汇总报告已保存至: {summary_filename}")

        # 打印汇总报告
        print(f"\n{'=' * 80}")
        print("土地利用变化汇总报告")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))
    else:
        print("没有成功分析任何时间段，请检查数据和文件路径")


if __name__ == "__main__":
    main()