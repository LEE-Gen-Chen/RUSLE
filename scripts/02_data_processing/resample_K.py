import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm


def resample_k_to_30m(input_k_path, output_30m_path):
    """
    将K值栅格重采样到30米分辨率

    参数:
    input_k_path: 输入K值栅格路径
    output_30m_path: 输出30米分辨率K值栅格路径
    """
    print("正在重采样K值数据到30米分辨率...")

    with rasterio.open(input_k_path) as src:
        # 获取原始数据信息
        src_data = src.read(1)
        src_crs = src.crs
        src_bounds = src.bounds

        # 计算30米分辨率的变换参数
        transform_30m, width_30m, height_30m = calculate_default_transform(
            src_crs, src_crs,
            src.width, src.height,
            left=src_bounds.left, bottom=src_bounds.bottom,
            right=src_bounds.right, top=src_bounds.top,
            resolution=30
        )

        # 创建目标数组
        dst_data = np.zeros((height_30m, width_30m), dtype=np.float32)

        # 重采样
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform_30m,
            dst_crs=src_crs,
            resampling=Resampling.bilinear,
            num_threads=4
        )

        # 更新元数据
        profile_30m = src.profile.copy()
        profile_30m.update({
            'transform': transform_30m,
            'width': width_30m,
            'height': height_30m,
            'dtype': 'float32',
            'nodata': np.nan
        })

        # 保存30米分辨率K值栅格
        with rasterio.open(output_30m_path, 'w', **profile_30m) as dst:
            dst.write(dst_data, 1)

        print(f"重采样完成！30米分辨率K值数据已保存至: {output_30m_path}")
        print(f"新数据尺寸: {width_30m}x{height_30m}")
        print(f"新数据分辨率: {transform_30m[0]}米")

        return dst_data, profile_30m


def plot_k_distribution(k_data, output_plot_path):
    """
    绘制K值空间分布图

    参数:
    k_data: K值数据数组
    output_plot_path: 输出图像路径
    """
    print("正在绘制K值空间分布图...")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 空间分布图
    im1 = ax1.imshow(k_data, cmap='YlOrRd', vmin=0.05, vmax=0.35)
    ax1.set_title('广东省土壤可蚀性因子K值空间分布', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('K值', rotation=270, labelpad=15)

    # 统计直方图
    valid_k = k_data[~np.isnan(k_data)]
    ax2.hist(valid_k, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_title('K值分布直方图', fontsize=14, fontweight='bold')
    ax2.set_xlabel('K值')
    ax2.set_ylabel('像元数量')
    ax2.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = f"""统计信息:
最小值: {np.min(valid_k):.4f}
最大值: {np.max(valid_k):.4f}
平均值: {np.mean(valid_k):.4f}
标准差: {np.std(valid_k):.4f}
有效像元数: {len(valid_k):,}"""

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"K值分布图已保存至: {output_plot_path}")

    plt.show()


def analyze_k_distribution(k_data, output_csv_path):
    """
    分析K值分布并保存到CSV文件

    参数:
    k_data: K值数据数组
    output_csv_path: 输出CSV文件路径
    """
    print("正在分析K值分布并生成统计文件...")

    # 获取有效数据
    valid_k = k_data[~np.isnan(k_data)]

    # 创建K值分级
    k_bins = np.arange(0.05, 0.36, 0.01)  # 从0.05到0.35，步长0.01
    k_labels = [f"{k_bins[i]:.2f}-{k_bins[i + 1]:.2f}" for i in range(len(k_bins) - 1)]

    # 统计每个区间的像元数量
    hist, bin_edges = np.histogram(valid_k, bins=k_bins)

    # 计算百分比
    total_pixels = len(valid_k)
    percentages = (hist / total_pixels) * 100

    # 创建统计表格
    distribution_df = pd.DataFrame({
        'K值区间': k_labels,
        '像元数量': hist,
        '百分比(%)': percentages
    })

    # 添加累计百分比
    distribution_df['累计百分比(%)'] = distribution_df['百分比(%)'].cumsum()

    # 保存到CSV
    distribution_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"K值分布统计已保存至: {output_csv_path}")

    # 输出基本统计信息
    print(f"\nK值基本统计信息:")
    print(f"  最小值: {np.min(valid_k):.4f}")
    print(f"  最大值: {np.max(valid_k):.4f}")
    print(f"  平均值: {np.mean(valid_k):.4f}")
    print(f"  中位数: {np.median(valid_k):.4f}")
    print(f"  标准差: {np.std(valid_k):.4f}")
    print(f"  有效像元总数: {len(valid_k):,}")

    return distribution_df


def create_detailed_analysis(k_data, output_detailed_csv_path):
    """
    创建详细的K值统计分析

    参数:
    k_data: K值数据数组
    output_detailed_csv_path: 输出详细分析CSV路径
    """
    print("正在生成详细统计分析...")

    valid_k = k_data[~np.isnan(k_data)]

    # 创建更详细的统计
    detailed_stats = {
        '统计指标': ['最小值', '最大值', '平均值', '中位数', '标准差',
                     '变异系数', '偏度', '峰度', 'Q1(25%)', 'Q3(75%)',
                     '有效像元数', '缺失像元数'],
        '数值': [
            np.min(valid_k),
            np.max(valid_k),
            np.mean(valid_k),
            np.median(valid_k),
            np.std(valid_k),
            np.std(valid_k) / np.mean(valid_k),  # 变异系数
            0,  # 偏度 (需要scipy)
            0,  # 峰度 (需要scipy)
            np.percentile(valid_k, 25),
            np.percentile(valid_k, 75),
            len(valid_k),
            np.sum(np.isnan(k_data))
        ]
    }

    # 尝试计算偏度和峰度
    try:
        from scipy.stats import skew, kurtosis
        detailed_stats['数值'][6] = skew(valid_k)
        detailed_stats['数值'][7] = kurtosis(valid_k)
    except ImportError:
        print("注意: 未安装scipy，跳过偏度和峰度计算")

    detailed_df = pd.DataFrame(detailed_stats)
    detailed_df.to_csv(output_detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"详细统计分析已保存至: {output_detailed_csv_path}")

    return detailed_df


def main():
    """主函数"""

    # 输入输出文件路径
    input_k_path = r"./data\中国土壤数据集\HWSD_RASTER\RUSLE_K.tif"
    output_30m_path = r"./data\中国土壤数据集\HWSD_Resample\RUSLE_K_30m.tif"
    output_plot_path = r"./data\中国土壤数据集\HWSD_Resample\K_value_distribution.tif"
    output_csv_path = r"./data\中国土壤数据集\HWSD_Resample\K_value_distribution.csv"
    output_detailed_csv_path = r"./data\中国土壤数据集\HWSD_Resample\K_value_detailed_stats.csv"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_30m_path), exist_ok=True)

    try:
        # 1. 重采样到30米分辨率
        k_data_30m, profile_30m = resample_k_to_30m(input_k_path, output_30m_path)

        # 2. 绘制K值分布图
        plot_k_distribution(k_data_30m, output_plot_path)

        # 3. 分析K值分布并保存到CSV
        distribution_df = analyze_k_distribution(k_data_30m, output_csv_path)

        # 4. 创建详细统计分析
        detailed_df = create_detailed_analysis(k_data_30m, output_detailed_csv_path)

        print("\n所有处理完成！")
        print(f"生成的文件:")
        print(f"  30米分辨率K值栅格: {output_30m_path}")
        print(f"  K值分布图: {output_plot_path}")
        print(f"  K值分布统计: {output_csv_path}")
        print(f"  详细统计分析: {output_detailed_csv_path}")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()