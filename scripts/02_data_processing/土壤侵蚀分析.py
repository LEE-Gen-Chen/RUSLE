import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams


# 解决中文字体显示问题
def set_chinese_font():
    """设置中文字体，解决中文显示问题"""
    try:
        # 尝试使用系统中可能存在的支持中文的字体
        possible_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'STSong',
            'SimSun',
            'KaiTi',
            'FangSong',
            'Arial Unicode MS',
            'DejaVuSans'
        ]

        # 设置字体
        rcParams['font.sans-serif'] = possible_fonts
        rcParams['axes.unicode_minus'] = False
        rcParams['font.family'] = 'sans-serif'

        # 测试字体是否支持中文
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', fontsize=10, ha='center')
        plt.close()
        print("✅ 中文字体设置成功")

    except Exception as e:
        print(f"⚠️ 中文字体设置失败: {e}")
        print("将使用默认字体，中文可能显示为方框")


# 设置中文字体
set_chinese_font()


def plot_all_cities_summary():
    """
    将所有地市的土壤侵蚀模数变化趋势汇总到一张大图中
    """
    # 设置数据路径
    data_dir = r"./data\RUSLE\土壤侵蚀分析\地市土壤侵蚀数据"
    output_dir = r"./data\RUSLE\土壤侵蚀分析\地市侵蚀模数变化图"

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有地市CSV文件
    city_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and '土壤侵蚀模数' in f]

    if not city_files:
        print("未找到任何地市数据文件")
        return

    print(f"找到 {len(city_files)} 个地市数据文件")

    # 创建大图 - 使用5x5的子图布局，增加图形尺寸和间隔
    fig, axes = plt.subplots(5, 5, figsize=(25, 20))
    axes = axes.flatten()

    # 存储所有城市数据用于趋势分析
    all_cities_data = []

    # 为每个地市绘制子图
    for i, city_file in enumerate(city_files):
        if i >= len(axes):
            break

        try:
            # 提取地市名称
            city_name = city_file.replace('土壤侵蚀模数.csv', '')

            # 读取CSV文件
            file_path = os.path.join(data_dir, city_file)
            df = pd.read_csv(file_path)

            # 过滤掉统计信息行（只保留数值型年份数据）
            df_data = df[pd.to_numeric(df['YEAR'], errors='coerce').notna()].copy()

            # 确保数据类型正确
            df_data['YEAR'] = df_data['YEAR'].astype(int)
            df_data['MEAN'] = pd.to_numeric(df_data['MEAN'], errors='coerce')

            # 按年份排序
            df_data = df_data.sort_values('YEAR')

            # 确保有数据可绘制
            if len(df_data) == 0:
                print(f"警告: {city_name} 没有有效数据")
                axes[i].set_visible(False)
                continue

            # 在当前子图中绘制折线图
            ax = axes[i]
            ax.plot(df_data['YEAR'], df_data['MEAN'],
                    marker='o', linewidth=4, markersize=5,
                    color='#2E86AB', markerfacecolor='#F24236')

            # 设置子图标题和标签 - 增加字体大小
            ax.set_title(f'{city_name}', fontsize=20, fontweight='bold', pad=8)
            ax.set_xlabel('年份', fontsize=18)
            ax.set_ylabel('侵蚀模数 (t/km^2·a)', fontsize=18)

            # 设置x轴刻度（每5年显示一个标签，避免拥挤）
            years = df_data['YEAR'].unique()
            if len(years) > 10:
                # 如果年份太多，每隔5年显示一个标签
                ax.set_xticks(years[::5])
            else:
                ax.set_xticks(years)

            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

            # 设置y轴格式（显示2位小数）
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

            # 添加网格
            ax.grid(True, alpha=0.3)

            # 存储数据用于趋势分析
            all_cities_data.append({
                'city': city_name,
                'data': df_data
            })

        except Exception as e:
            print(f"处理 {city_file} 时出错: {e}")
            axes[i].set_visible(False)
            continue

    # 隐藏多余的子图
    for j in range(len(city_files), len(axes)):
        axes[j].set_visible(False)

    # 设置总标题 - 调整位置和字体大小
    plt.suptitle('广东省各地市土壤侵蚀模数变化趋势 (1990-2024)',
                 fontsize=40, fontweight='bold', y=0.98)

    # 调整布局 - 增加子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.5)  # 增加水平和垂直间距

    # 保存汇总图表
    output_file = os.path.join(output_dir, '广东省各地市土壤侵蚀模数变化汇总图.png')
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    print(f"已保存: 广东省各地市土壤侵蚀模数变化汇总图.png")


    # 关闭图表
    plt.close()

    return all_cities_data


def create_comparison_line_chart(all_cities_data):
    """
    创建所有地市在同一坐标系中的对比折线图
    """
    output_dir = r"./data\RUSLE\土壤侵蚀分析\地市侵蚀模数变化图"

    # 创建大图 - 增加图形尺寸
    plt.figure(figsize=(20, 12))

    # 颜色列表，为每个城市分配不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_cities_data)))

    # 为每个地市绘制趋势线
    legend_handles = []
    legend_labels = []

    for i, city_info in enumerate(all_cities_data):
        city_name = city_info['city']
        city_data = city_info['data']

        if len(city_data) > 0:
            # 绘制折线
            line, = plt.plot(city_data['YEAR'], city_data['MEAN'],
                             marker='o', linewidth=2, markersize=4,
                             color=colors[i], label=city_name, alpha=0.8)

            legend_handles.append(line)
            legend_labels.append(city_name)

    # 设置图表标题和标签 - 增加字体大小
    plt.title('广东省各地市土壤侵蚀模数变化趋势对比 (1990-2024)',
              fontsize=30, fontweight='bold', pad=20)
    plt.xlabel('年份', fontsize=16)
    plt.ylabel('土壤侵蚀模数 (t/km^2·a)', fontsize=16)

    # 设置坐标轴标签字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 设置y轴格式（显示2位小数）
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # 添加图例（放在图表右侧）- 增加字体大小
    plt.legend(handles=legend_handles, labels=legend_labels,
               bbox_to_anchor=(1.05, 1), loc='upper left',
               fontsize=12, ncol=2)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存对比图表
    output_file = os.path.join(output_dir, '广东省各地市土壤侵蚀模数变化对比图.png')
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    print(f"已保存: 广东省各地市土壤侵蚀模数变化对比图.png")

    # 关闭图表
    plt.close()


def analyze_regional_trends(all_cities_data):
    """
    分析区域趋势并创建分组图表
    """
    output_dir = r"./data\RUSLE\土壤侵蚀分析\地市侵蚀模数变化图"

    # 定义区域分组（可以根据实际情况调整）
    regions = {
        '珠三角地区': ['广州市', '深圳市', '珠海市', '佛山市', '江门市', '东莞市', '中山市', '惠州市', '肇庆市'],
        '粤东地区': ['汕头市', '潮州市', '揭阳市', '汕尾市'],
        '粤西地区': ['湛江市', '茂名市', '阳江市', '云浮市'],
        '粤北地区': ['韶关市', '清远市', '梅州市', '河源市']
    }

    # 创建区域分组图表 - 增加图形尺寸
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    for region_idx, (region_name, cities_in_region) in enumerate(regions.items()):
        if region_idx >= len(axes):
            break

        ax = axes[region_idx]

        for i, city_name in enumerate(cities_in_region):
            # 查找城市数据
            city_data = None
            for city_info in all_cities_data:
                if city_info['city'] == city_name:
                    city_data = city_info['data']
                    break

            if city_data is not None and len(city_data) > 0:
                ax.plot(city_data['YEAR'], city_data['MEAN'],
                        marker='o', linewidth=2.5, markersize=5,
                        color=colors[i], label=city_name, alpha=0.8)

        # 增加区域图表的字体大小
        ax.set_title(f'{region_name}土壤侵蚀模数变化', fontsize=18, fontweight='bold')
        ax.set_xlabel('年份', fontsize=18)
        ax.set_ylabel('土壤侵蚀模数 (t/km^2·a)', fontsize=18)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # 调整主标题 - 增加字体大小和调整位置
    plt.suptitle('广东省各地区土壤侵蚀模数变化趋势 (1990-2024)',
                 fontsize=30, fontweight='bold', y=0.98)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

    # 保存区域分组图表
    output_file = os.path.join(output_dir, '广东省各地区土壤侵蚀模数变化趋势.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: 广东省各地区土壤侵蚀模数变化趋势.png")

    plt.close()


# 主程序
if __name__ == "__main__":
    print("开始生成土壤侵蚀模数变化图表...")

    # 生成所有地市的汇总子图
    all_cities_data = plot_all_cities_summary()

    if all_cities_data:
        # 生成对比折线图
        create_comparison_line_chart(all_cities_data)

        # 生成区域分组图表
        analyze_regional_trends(all_cities_data)

        print("\n✅ 所有图表生成完成！")
        print("生成的图表保存在: H:\\毕业论文\\RUSLE\\土壤侵蚀分析\\地市侵蚀模数变化图")
        print("包括:")
        print("  - 广东省各地市土壤侵蚀模数变化汇总图.png (5x5子图汇总)")
        print("  - 广东省各地市土壤侵蚀模数变化对比图.png (所有城市在同一坐标系)")
        print("  - 广东省各地区土壤侵蚀模数变化趋势.png (按区域分组)")
    else:
        print("❌ 数据处理失败，请检查文件路径和格式")