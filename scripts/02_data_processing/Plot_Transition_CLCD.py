import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from matplotlib.patches import ConnectionPatch

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# CLCD分类系统 - 使用中文标签
CLCD_CLASSES = {
    'Cropland': '耕地',
    'Forest': '林地',
    'Shrub': '灌木',
    'Grassland': '草地',
    'Water': '水域',
    'Snow/Ice': '雪/冰',
    'Barren': '未利用地',
    'Impervious': '建设用地',
    'Wetland': '湿地'
}

# 使用您指定的配色方案
COLOR_SCHEME = {
    '耕地': '#FFFF00',  # 黄色
    '林地': '#006400',  # 深绿色
    '水域': '#0000FF',  # 蓝色
    '未利用地': '#808080',  # 灰色
    '草地': '#90EE90',  # 浅绿色
    '建设用地': '#FF0000',  # 红色
    # 为其他类别分配颜色
    '灌木': '#32CD32',  # LimeGreen
    '雪/冰': '#87CEEB',  # SkyBlue
    '湿地': '#20B2AA'  # LightSeaGreen
}

# 设置输出目录
OUTPUT_DIR = r"./data\CLCD\Plot"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pixels_to_hectares(pixels, resolution=30):
    """将像元数转换为公顷"""
    # 每个像元的面积（平方米）= 分辨率 * 分辨率
    pixel_area = resolution * resolution  # 平方米
    # 转换为公顷 (1公顷 = 10000平方米)
    hectares = pixels * pixel_area / 10000
    return hectares


def read_and_convert_transition_matrix(csv_file):
    """读取转移矩阵CSV文件并转换为公顷"""
    df = pd.read_csv(csv_file, index_col=0, encoding='utf-8-sig')

    # 移除汇总行和列
    df = df.drop('Column_Total', errors='ignore')
    df = df.drop('Row_Total', axis=1, errors='ignore')

    # 将像元数转换为公顷
    df_hectares = df.applymap(pixels_to_hectares)

    # 将索引和列名转换为中文
    df_hectares.index = [CLCD_CLASSES.get(idx, idx) for idx in df_hectares.index]
    df_hectares.columns = [CLCD_CLASSES.get(col, col) for col in df_hectares.columns]

    return df_hectares


def create_chord_diagram(transition_matrix, title, output_file):
    """创建优化版弦图（纯matplotlib实现）- 公顷单位"""
    # 准备数据
    labels = list(transition_matrix.index)
    matrix = transition_matrix.values

    # 计算弧段比例（基于行总和）
    row_sums = matrix.sum(axis=1)
    total_area = row_sums.sum()

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    ax.set_aspect('equal')

    # 参数设置
    inner_radius = 1.0
    outer_radius = 1.3
    label_radius = 1.45

    # 计算角度（按面积比例）
    angles = []
    current_angle = np.pi / 2

    for size in row_sums:
        proportion = size / total_area
        arc_angle = 2 * np.pi * proportion
        angles.append((current_angle, current_angle + arc_angle))
        current_angle += arc_angle

    # 绘制扇区（带渐变效果）
    for i, (label, (start, end), color) in enumerate(zip(labels, angles,
                                                         [COLOR_SCHEME[l] for l in labels])):
        # 创建径向渐变效果
        n_grad = 30
        for k, r in enumerate(np.linspace(inner_radius, outer_radius, n_grad)):
            alpha = 0.7 + 0.3 * (k / n_grad)  # 渐变透明度
            ax.fill(
                np.cos(np.linspace(start, end, 50)) * r,
                np.sin(np.linspace(start, end, 50)) * r,
                color=color, alpha=alpha, ec='none'
            )

        # 绘制扇区边框
        ax.plot([np.cos(start) * r for r in [inner_radius, outer_radius]],
                [np.sin(start) * r for r in [inner_radius, outer_radius]],
                color='white', lw=2)

        # 添加标签
        mid_angle = (start + end) / 2
        ax.text(
            np.cos(mid_angle) * label_radius, np.sin(mid_angle) * label_radius,
            f"{label}\n{row_sums[i]:,.0f} ha",
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', ec=color, lw=2, alpha=0.9)
        )

        # 添加百分比
        ax.text(
            np.cos(mid_angle) * (inner_radius + outer_radius) / 2,
            np.sin(mid_angle) * (inner_radius + outer_radius) / 2,
            f"{row_sums[i] / total_area * 100:.1f}%",
            ha='center', va='center', fontsize=9, color='white', fontweight='bold'
        )

    # 绘制和弦（优化曲线）
    max_value = matrix.max()
    threshold = max_value * 0.01

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and matrix[i, j] > threshold:
                start_mid = (angles[i][0] + angles[i][1]) / 2
                end_mid = (angles[j][0] + angles[j][1]) / 2

                # 计算控制点（曲线弯曲程度）
                dist = min(abs(end_mid - start_mid), 2 * np.pi - abs(end_mid - start_mid))
                control_factor = max(0.2, 1 - dist / np.pi * 0.6)
                control_radius = inner_radius * control_factor

                # 创建贝塞尔曲线
                t = np.linspace(0, 1, 100)
                x_start, y_start = np.cos(start_mid) * inner_radius, np.sin(start_mid) * inner_radius
                x_end, y_end = np.cos(end_mid) * inner_radius, np.sin(end_mid) * inner_radius
                x_control = np.cos((start_mid + end_mid) / 2) * control_radius
                y_control = np.sin((start_mid + end_mid) / 2) * control_radius

                x_curve = (1 - t) ** 2 * x_start + 2 * (1 - t) * t * x_control + t ** 2 * x_end
                y_curve = (1 - t) ** 2 * y_start + 2 * (1 - t) * t * y_control + t ** 2 * y_end

                # 线宽和透明度
                width = matrix[i, j] / max_value * 15 + 0.5
                alpha = min(matrix[i, j] / max_value * 0.6 + 0.2, 0.7)

                # 绘制曲线
                color = COLOR_SCHEME.get(labels[i], '#CCCCCC')
                ax.plot(
                    x_curve, y_curve,
                    color=color, lw=width, alpha=alpha, solid_capstyle='round',
                    zorder=2 if i < j else 3
                )

    # 中心统计信息
    change_area = total_area - np.trace(matrix)
    ax.text(0, 0, f"总转移:\n{change_area:,.0f} ha\n({change_area / total_area * 100:.1f}%)",
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    # 标题
    ax.set_title(f"{title}\n(单位: 公顷)", fontsize=18, fontweight='bold', pad=30)

    # 设置范围
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.axis('off')

    # 图例（右上角）
    legend_elements = [plt.Line2D([0], [0], color=c, lw=6, label=l, alpha=0.8)
                       for l, c in COLOR_SCHEME.items()]
    ax.legend(
        handles=legend_elements,
        loc='upper right', bbox_to_anchor=(1.18, 1.08),
        fontsize=10, frameon=True, fancybox=True, shadow=True, ncol=2, title="土地利用类型"
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"优化弦图已保存: {output_file}")


def create_heatmap_diagram(transition_matrix, title, output_file):
    """创建热力图作为弦图的替代方案"""
    # 准备数据
    labels = list(transition_matrix.index)
    matrix = transition_matrix.values

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建热力图
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # 添加数值标签
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > matrix.max() * 0.01:  # 只显示大于1%最大值的转移
                text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=8)

    # 设置标题
    ax.set_title(f"{title}\n(单位: 公顷)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('目标土地利用类型', fontsize=12)
    ax.set_ylabel('源土地利用类型', fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('转移面积 (公顷)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"热力图已保存: {output_file}")


def create_enhanced_sankey(transition_matrix, title, output_file):
    """创建增强版桑基图（使用公顷单位）"""

    labels = list(transition_matrix.index)
    matrix = transition_matrix.values

    # 准备桑基图数据
    source = []
    target = []
    value = []
    link_colors = []

    # 计算阈值，只显示显著的转移
    threshold = matrix.max() * 0.005  # 只显示大于0.5%最大值的转移

    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > threshold and i != j:
                source.append(i)
                target.append(j)
                value.append(matrix[i, j])

                # 设置连接颜色（基于源类型）
                color = COLOR_SCHEME.get(labels[i], '#CCCCCC')
                # 转换为RGBA格式
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                link_colors.append(f'rgba({r},{g},{b},0.6)')

    # 创建节点颜色
    node_colors = [COLOR_SCHEME.get(label, '#CCCCCC') for label in labels]

    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',  # 使用snap排列使节点更整齐
        node=dict(
            pad=15,  # 减少节点间距
            thickness=20,  # 调整节点厚度
            line=dict(color="black", width=1),
            label=labels,
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>面积: %{value:,.0f} 公顷<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            hovertemplate='<b>%{source.label} → %{target.label}</b><br>转移量: %{value:,.0f} 公顷<extra></extra>'
        )
    )])

    fig.update_layout(
        title={
            'text': f"{title} (单位: 公顷)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'black'}
        },
        font=dict(size=12, color='black'),
        width=1200,
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig.write_html(output_file)
    print(f"增强桑基图已保存: {output_file}")


def create_multi_year_sankey(all_matrices, output_file):
    """创建多年交互式桑基图（公顷单位）"""

    # 定义所有年份
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

    # 准备节点和连接数据
    nodes = []
    links = []

    # 为每个年份的每种土地类型创建节点
    node_indices = {}  # 存储节点索引
    current_index = 0

    # 添加节点
    for i, year in enumerate(years):
        for land_type in COLOR_SCHEME.keys():  # 使用COLOR_SCHEME的键作为土地类型
            node_name = f"{land_type}_{year}"
            nodes.append({
                "name": node_name,
                "year": year,
                "type": land_type,
                "color": COLOR_SCHEME.get(land_type, '#CCCCCC')
            })
            node_indices[node_name] = current_index
            current_index += 1

    # 添加连接（相邻年份之间的转移）
    for i in range(len(years) - 1):
        start_year = years[i]
        end_year = years[i + 1]

        # 获取对应的转移矩阵
        matrix_key = f"{start_year}_{end_year}"
        if matrix_key in all_matrices:
            transition_matrix = all_matrices[matrix_key]
            labels = list(transition_matrix.index)
            matrix_values = transition_matrix.values

            # 添加转移连接
            for j, source_type in enumerate(labels):
                for k, target_type in enumerate(labels):
                    value = matrix_values[j, k]

                    # 只添加显著的转移
                    if value > matrix_values.max() * 0.01 and value > 0:
                        source_node = f"{source_type}_{start_year}"
                        target_node = f"{target_type}_{end_year}"

                        # 修复颜色格式问题
                        color = COLOR_SCHEME.get(source_type, '#CCCCCC')
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        rgba_color = f'rgba({r},{g},{b},0.4)'

                        links.append({
                            "source": node_indices[source_node],
                            "target": node_indices[target_node],
                            "value": value,
                            "color": rgba_color
                        })

    # 提取节点属性
    node_names = [node["name"] for node in nodes]
    node_colors = [node["color"] for node in nodes]

    # 提取连接属性
    sources = [link["source"] for link in links]
    targets = [link["target"] for link in links]
    values = [link["value"] for link in links]
    link_colors = [link["color"] for link in links]

    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=1),
            label=node_names,
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>年份: %{customdata}<extra></extra>',
            customdata=[f"{node['year']}" for node in nodes]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='<b>%{source.label} → %{target.label}</b><br>转移量: %{value:,.0f} 公顷<extra></extra>'
        )
    )])

    # 更新布局
    fig.update_layout(
        title={
            'text': "广东省土地利用多年转移桑基图 (1990-2024)<br><sub>单位: 公顷</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'black'}
        },
        font=dict(size=12, color='black'),
        width=1600,
        height=900,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # 保存为HTML文件
    fig.write_html(output_file)
    print(f"多年桑基图已保存: {output_file}")

    return fig


def create_trend_analysis(all_matrices):
    """创建趋势分析图表"""

    # 提取主要转移的时间趋势
    major_transfers = {}
    periods = []

    for period, matrix_df in all_matrices.items():
        if period == "1990_2024":  # 跳过总时间段
            continue

        periods.append(period)
        matrix_values = matrix_df.values
        labels = list(matrix_df.index)

        # 找出主要转移
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and matrix_values[i, j] > matrix_values.max() * 0.05:
                    transfer_key = f"{labels[i]} → {labels[j]}"
                    if transfer_key not in major_transfers:
                        major_transfers[transfer_key] = {}
                    major_transfers[transfer_key][period] = matrix_values[i, j]

    # 创建时间趋势图
    if major_transfers:
        plt.figure(figsize=(14, 8))

        for i, (transfer, data) in enumerate(major_transfers.items()):
            values = [data.get(period, 0) for period in periods]
            if max(values) > 0:  # 只绘制有数据的转移
                plt.plot(periods, values, marker='o', linewidth=2, label=transfer, markersize=6)

        plt.title('主要土地利用转移时间趋势 (单位: 公顷)', fontsize=16, fontweight='bold')
        plt.xlabel('时间段', fontsize=12)
        plt.ylabel('转移面积 (公顷)', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = os.path.join(OUTPUT_DIR, "temporal_trends.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"时间趋势图已保存: {output_file}")


def create_change_rate_analysis(all_matrices):
    """创建变化率分析"""

    # 分析每个时间段的变化率
    change_rates = {}

    for period, matrix_df in all_matrices.items():
        if period == "1990_2024":  # 跳过总时间段
            continue

        matrix_values = matrix_df.values
        total_area = matrix_values.sum()
        unchanged_area = np.trace(matrix_values)
        changed_area = total_area - unchanged_area
        change_rate = (changed_area / total_area) * 100

        change_rates[period] = {
            'total_area': total_area,
            'unchanged_area': unchanged_area,
            'changed_area': changed_area,
            'change_rate': change_rate
        }

    # 创建变化率DataFrame
    change_df = pd.DataFrame.from_dict(change_rates, orient='index')
    change_df.index.name = 'Period'
    change_df.reset_index(inplace=True)

    # 保存变化率数据
    change_output = os.path.join(OUTPUT_DIR, "change_analysis_hectares.csv")
    change_df.to_csv(change_output, index=False, encoding='utf-8-sig')
    print(f"变化分析报告已保存: {change_output}")

    # 创建变化率图表
    plt.figure(figsize=(12, 6))
    periods = change_df['Period']
    change_rates = change_df['change_rate']

    bars = plt.bar(periods, change_rates, color='skyblue', alpha=0.7)
    plt.title('各时间段土地利用变化率 (单位: 公顷)', fontsize=16, fontweight='bold')
    plt.xlabel('时间段', fontsize=12)
    plt.ylabel('变化率 (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, rate in zip(bars, change_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{rate:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "change_rate_barchart.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"变化率柱状图已保存: {os.path.join(OUTPUT_DIR, 'change_rate_barchart.png')}")

    return change_df


def save_converted_matrices(all_matrices):
    """保存转换后的公顷矩阵到CSV"""
    for period, matrix_df in all_matrices.items():
        output_file = os.path.join(OUTPUT_DIR, f"transition_matrix_hectares_{period}.csv")
        matrix_df.to_csv(output_file, encoding='utf-8-sig')
        print(f"转换后的矩阵已保存: {output_file}")


def analyze_all_periods():
    """分析所有时间段的转移矩阵"""

    base_path = r"./data\CLCD\project"

    # 定义所有时间段
    periods = [
        (1990, 1995), (1995, 2000), (2000, 2005),
        (2005, 2010), (2010, 2015), (2015, 2020),
        (2020, 2024), (1990, 2024)
    ]

    all_matrices = {}

    for start_year, end_year in periods:
        csv_file = f"{base_path}/transition_matrix_{start_year}_{end_year}.csv"

        try:
            print(f"\n处理时间段: {start_year}-{end_year}")

            # 读取转移矩阵并转换为公顷
            transition_matrix = read_and_convert_transition_matrix(csv_file)
            all_matrices[f"{start_year}_{end_year}"] = transition_matrix

            print(f"成功读取转移矩阵，形状: {transition_matrix.shape}")
            print(f"总转移面积: {transition_matrix.values.sum():,.0f} 公顷")

            # 创建弦图（已替换为优化版）
            chord_output = os.path.join(OUTPUT_DIR, f"chord_diagram_{start_year}_{end_year}.png")
            create_chord_diagram(
                transition_matrix,
                f"土地利用转移弦图 ({start_year}-{end_year})",
                chord_output
            )

            # 创建热力图作为替代可视化
            heatmap_output = os.path.join(OUTPUT_DIR, f"heatmap_diagram_{start_year}_{end_year}.png")
            create_heatmap_diagram(
                transition_matrix,
                f"土地利用转移热力图 ({start_year}-{end_year})",
                heatmap_output
            )

            # 创建桑基图
            sankey_output = os.path.join(OUTPUT_DIR, f"sankey_diagram_{start_year}_{end_year}.html")
            create_enhanced_sankey(
                transition_matrix,
                f"土地利用转移 ({start_year}-{end_year})",
                sankey_output
            )

            # 打印主要转移
            print(f"\n{start_year}-{end_year} 主要土地利用转移 (公顷):")
            matrix_values = transition_matrix.values
            labels = list(transition_matrix.index)

            # 找出前5个最大的转移
            transfers = []
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if i != j and matrix_values[i, j] > 0:
                        transfers.append((labels[i], labels[j], matrix_values[i, j]))

            transfers.sort(key=lambda x: x[2], reverse=True)

            for i, (from_type, to_type, area) in enumerate(transfers[:5]):
                percentage = (area / matrix_values.sum()) * 100
                print(f"  {i + 1}. {from_type} → {to_type}: {area:,.0f} 公顷 ({percentage:.2f}%)")

        except FileNotFoundError:
            print(f"文件不存在: {csv_file}")
        except Exception as e:
            print(f"处理 {start_year}-{end_year} 时出错: {e}")

    return all_matrices


if __name__ == "__main__":
    print("开始土地利用转移可视化分析...")
    print(f"输出目录: {OUTPUT_DIR}")

    # 分析所有时间段
    all_matrices = analyze_all_periods()

    if all_matrices:
        # 保存转换后的矩阵
        save_converted_matrices(all_matrices)

        # 创建多年交互式桑基图
        multi_year_output = os.path.join(OUTPUT_DIR, "multi_year_sankey.html")
        try:
            create_multi_year_sankey(all_matrices, multi_year_output)
        except Exception as e:
            print(f"创建多年桑基图时出错: {e}")
            print("跳过多年桑基图的创建...")

        # 创建趋势分析
        create_trend_analysis(all_matrices)

        # 创建变化率分析
        change_df = create_change_rate_analysis(all_matrices)

        print("\n所有分析完成!")
        print(f"\n主要输出文件:")
        print(f"- 转换后的公顷矩阵CSV文件: {OUTPUT_DIR}/transition_matrix_hectares_*.csv")
        print(f"- 弦图: {OUTPUT_DIR}/chord_diagram_*.png")
        print(f"- 热力图: {OUTPUT_DIR}/heatmap_diagram_*.png")
        print(f"- 单时间段桑基图: {OUTPUT_DIR}/sankey_diagram_*.html")
        print(f"- 多年交互式桑基图: {multi_year_output}")
        print(f"- 时间趋势图: {OUTPUT_DIR}/temporal_trends.png")
        print(f"- 变化分析报告: {OUTPUT_DIR}/change_analysis_hectares.csv")
        print(f"- 变化率柱状图: {OUTPUT_DIR}/change_rate_barchart.png")

        # 打印变化率汇总
        print(f"\n各时间段变化率汇总:")
        for _, row in change_df.iterrows():
            print(f"  {row['Period']}: {row['change_rate']:.2f}%")
    else:
        print("没有成功读取任何转移矩阵数据")