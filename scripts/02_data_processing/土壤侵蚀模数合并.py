import pandas as pd
import numpy as np
import os


def process_soil_erosion_data():
    """
    处理1990-2024年土壤侵蚀模数数据，按地市切分保存
    """
    # 设置数据路径
    data_path = r"./data\RUSLE"
    output_base_path = r"./data\RUSLE\土壤侵蚀分析"

    # 存储所有年份的数据
    all_data = []

    # 遍历1990-2024年的数据文件
    for year in range(1990, 2025):
        filename = f"Avg_土壤侵蚀模数{year}.xls"
        file_path = os.path.join(data_path, filename)

        if os.path.exists(file_path):
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 添加年份列
                df['YEAR'] = year
                all_data.append(df)
                print(f"成功读取 {year} 年数据")
            except Exception as e:
                print(f"读取 {year} 年数据失败: {e}")
        else:
            print(f"文件不存在: {filename}")

    if not all_data:
        print("未找到任何数据文件")
        return None

    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)

    # 将AREA从平方米转换为平方千米
    combined_data['AREA_km2'] = combined_data['AREA'] / 1000000

    # 按地市名称排序
    combined_data = combined_data.sort_values('地市')

    # 创建输出目录
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    # 保存总的合并数据
    output_csv = os.path.join(output_base_path, "1990-2024_土壤侵蚀模数_合并数据.csv")
    combined_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"合并数据已保存为: {output_csv}")

    # 按地市切分并保存独立文件
    split_data_by_city(combined_data, output_base_path)

    return combined_data


def split_data_by_city(data, output_base_path):
    """
    按地市切分数据，每个地市保存为独立CSV文件
    """
    # 获取所有地市
    cities = data['地市'].unique()

    # 创建输出子目录
    output_dir = os.path.join(output_base_path, "地市土壤侵蚀数据")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每个地市创建独立文件
    for city in cities:
        city_data = data[data['地市'] == city].copy()

        # 按年份排序
        city_data = city_data.sort_values('YEAR')

        # 计算每个地市的统计信息
        city_stats = calculate_city_statistics(city_data)

        # 添加统计信息到数据中（作为新行）
        stats_df = create_statistics_dataframe(city_stats, city)

        # 合并原始数据和统计信息
        final_city_data = pd.concat([city_data, stats_df], ignore_index=True)

        # 生成文件名
        filename = f"{city}土壤侵蚀模数.csv"
        file_path = os.path.join(output_dir, filename)

        # 保存文件
        final_city_data.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"已保存: {filename}")


def calculate_city_statistics(city_data):
    """
    计算每个地市的统计信息
    """
    stats = {}

    # 基本统计
    stats['年份范围'] = f"{city_data['YEAR'].min()}-{city_data['YEAR'].max()}"
    stats['数据年数'] = len(city_data)

    # 侵蚀模数统计
    stats['平均侵蚀模数'] = city_data['MEAN'].mean()
    stats['侵蚀模数中位数'] = city_data['MEAN'].median()
    stats['侵蚀模数最大值'] = city_data['MEAN'].max()
    stats['侵蚀模数最小值'] = city_data['MEAN'].min()
    stats['侵蚀模数标准差'] = city_data['MEAN'].std()

    # 面积统计（平方千米）
    stats['平均面积_km2'] = city_data['AREA_km2'].mean()
    stats['总面积_km2'] = city_data['AREA_km2'].sum()

    # 变化趋势
    if len(city_data) > 1:
        early_period = city_data[city_data['YEAR'] <= 2010]
        late_period = city_data[city_data['YEAR'] >= 2011]

        if not early_period.empty:
            early_avg = early_period['MEAN'].mean()
        else:
            early_avg = None

        if not late_period.empty:
            late_avg = late_period['MEAN'].mean()
        else:
            late_avg = None

        stats['1990-2010平均'] = early_avg
        stats['2011-2024平均'] = late_avg

        if early_avg is not None and late_avg is not None and early_avg > 0:
            change_rate = ((late_avg - early_avg) / early_avg) * 100
            stats['变化率_%'] = change_rate
        else:
            stats['变化率_%'] = None
    else:
        stats['1990-2010平均'] = None
        stats['2011-2024平均'] = None
        stats['变化率_%'] = None

    return stats


def create_statistics_dataframe(stats, city_name):
    """
    创建统计信息的DataFrame
    """
    # 创建统计信息行
    stats_rows = []

    # 添加分隔行
    separator = {col: '=' * 20 for col in
                 ['OBJECTID', '地市', 'ZONE_CODE', 'COUNT', 'AREA', 'MEAN', 'YEAR', 'AREA_km2']}
    stats_rows.append(separator)

    # 添加统计信息标题行
    title_row = {
        'OBJECTID': '统计信息',
        '地市': city_name,
        'ZONE_CODE': '', 'COUNT': '', 'AREA': '', 'MEAN': '', 'YEAR': '', 'AREA_km2': ''
    }
    stats_rows.append(title_row)

    # 添加各项统计信息
    for key, value in stats.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value) if value is not None else "N/A"

        stats_row = {
            'OBJECTID': key,
            '地市': formatted_value,
            'ZONE_CODE': '', 'COUNT': '', 'AREA': '', 'MEAN': '', 'YEAR': '', 'AREA_km2': ''
        }
        stats_rows.append(stats_row)

    return pd.DataFrame(stats_rows)


def analyze_data_summary(data):
    """
    简单的数据汇总分析（仅输出到控制台）
    """
    print("=" * 50)
    print("土壤侵蚀模数数据汇总分析")
    print("=" * 50)

    # 基本统计信息
    print("\n1. 基本统计信息:")
    print(f"数据时间范围: {data['YEAR'].min()} - {data['YEAR'].max()}")
    print(f"包含城市数量: {data['地市'].nunique()}")
    print(f"总数据点数: {len(data)}")

    # 侵蚀模数统计
    print(f"\n2. 侵蚀模数统计 (t/km²·a):")
    print(f"平均值: {data['MEAN'].mean():.2f}")
    print(f"中位数: {data['MEAN'].median():.2f}")
    print(f"最大值: {data['MEAN'].max():.2f}")
    print(f"最小值: {data['MEAN'].min():.2f}")
    print(f"标准差: {data['MEAN'].std():.2f}")

    # 按地市统计
    print(f"\n3. 各地市数据量统计:")
    city_counts = data['地市'].value_counts()
    for city, count in city_counts.items():
        print(f"  {city}: {count} 年数据")


# 主程序
if __name__ == "__main__":
    # 处理数据
    combined_data = process_soil_erosion_data()

    if combined_data is not None:
        # 进行简单的数据汇总分析
        analyze_data_summary(combined_data)

        print("\n✅ 所有数据处理完成！")
        print("生成的文件路径: H:\\毕业论文\\RUSLE\\土壤侵蚀分析")
        print("生成的文件:")
        print("  - 1990-2024_土壤侵蚀模数_合并数据.csv (总合并数据)")
        print("  - 地市土壤侵蚀数据/ (目录，包含各地市的独立CSV文件)")
        print("  - 每个地市文件包含原始数据和统计信息")
    else:
        print("❌ 数据处理失败，请检查文件路径和格式")