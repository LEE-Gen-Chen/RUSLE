import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义多种模型
def exponential_model(x, alpha, beta):
    """指数衰减模型: C = α × exp(-β × EVI)"""
    return alpha * np.exp(-beta * x)


def linear_model(x, a, b):
    """线性模型: C = a × EVI + b"""
    return a * x + b


def power_model(x, a, b):
    """幂函数模型: C = a × EVI^b"""
    return a * np.power(x, b)


def logarithmic_model(x, a, b):
    """对数模型: C = a × ln(EVI + 0.1) + b (加0.1避免ln(0))"""
    return a * np.log(x + 0.1) + b


def quadratic_model(x, a, b, c):
    """二次多项式模型: C = a × EVI² + b × EVI + c"""
    return a * x ** 2 + b * x + c


def sigmoid_model(x, a, b, c):
    """S型函数模型: C = a / (1 + exp(-b × (EVI - c)))"""
    return a / (1 + np.exp(-b * (x - c)))


def fit_multiple_models(x_data, y_data):
    """
    拟合多种模型并比较结果
    """
    models = {
        '指数模型': {
            'function': exponential_model,
            'params': 2,
            'initial_guess': [1.0, 5.8],
            'bounds': ([0, 0], [10, 20])  # 约束参数范围
        },
        '线性模型': {
            'function': linear_model,
            'params': 2,
            'initial_guess': [-2, 1.2],
            'bounds': ([-10, -10], [10, 10])
        },
        '幂函数模型': {
            'function': power_model,
            'params': 2,
            'initial_guess': [1, -2],
            'bounds': ([0, -10], [10, 10])
        },
        '对数模型': {
            'function': logarithmic_model,
            'params': 2,
            'initial_guess': [-0.5, 1],
            'bounds': ([-10, -10], [10, 10])
        },
        '二次多项式': {
            'function': quadratic_model,
            'params': 3,
            'initial_guess': [10, -10, 3],
            'bounds': ([-100, -100, -100], [100, 100, 100])
        },
        'S型函数': {
            'function': sigmoid_model,
            'params': 3,
            'initial_guess': [1, 10, 0.5],
            'bounds': ([0, 0, 0], [2, 100, 1])
        }
    }

    results = {}

    for model_name, model_info in models.items():
        try:
            if model_info['params'] == 2:
                popt, pcov = curve_fit(
                    model_info['function'], x_data, y_data,
                    p0=model_info['initial_guess'],
                    bounds=model_info['bounds'],
                    maxfev=5000
                )
                y_pred = model_info['function'](x_data, *popt)
            elif model_info['params'] == 3:
                popt, pcov = curve_fit(
                    model_info['function'], x_data, y_data,
                    p0=model_info['initial_guess'],
                    bounds=model_info['bounds'],
                    maxfev=5000
                )
                y_pred = model_info['function'](x_data, *popt)

            # 计算R²
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # 计算RMSE
            rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))

            results[model_name] = {
                'parameters': popt,
                'r_squared': r_squared,
                'rmse': rmse,
                'predictions': y_pred,
                'function': model_info['function']
            }

            print(f"{model_name}: R² = {r_squared:.6f}, RMSE = {rmse:.6f}")

        except Exception as e:
            print(f"{model_name} 拟合失败: {e}")
            results[model_name] = None

    return results


def fit_evi_c_model(evi_data_folder, output_plot=True):
    """
    拟合EVI-C因子回归模型，尝试多种模型并选择最佳模型
    """
    # C因子映射表
    c_factor_map = {
        1: 0.18,  # Cropland
        2: 0.006,  # Forest
        3: 0.015,  # Shrub
        4: 0.13,  # Grassland
        5: 0,  # Water
        6: 0,  # Snow/Ice
        7: 1,  # Barren
        8: 0,  # Impervious
        9: 0.001  # Wetland
    }

    # 土地利用类型名称映射
    landuse_names = {
        1: "耕地",
        2: "林地",
        3: "灌木地",
        4: "草地",
        5: "水域",
        6: "冰雪",
        7: "裸地",
        8: "不透水面",
        9: "湿地"
    }

    # 读取并处理所有EVI数据
    all_data = []
    for year in range(1990, 2025):
        try:
            file_path = f"{evi_data_folder}/EVI_CLCD_{year}.xlsx"
            df = pd.read_excel(file_path)
            df['year'] = year
            all_data.append(df)
        except:
            continue

    if not all_data:
        print("未找到任何EVI数据文件")
        return None

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 剔除异常值 - 使用更严格的阈值
    combined_df = combined_df[
        (combined_df['evi_mean'] <= 1) &
        (combined_df['evi_mean'] >= 0) &  # EVI通常为正
        (combined_df['evi_std'] <= 0.5)  # 更严格的标准差阈值
        ]

    # 计算每个土地利用类型的多年平均EVI
    landuse_evi_stats = combined_df.groupby('class_id').agg({
        'evi_mean': 'mean',
        'count': 'sum',
        'evi_std': 'mean'
    }).reset_index()

    # 添加C因子和土地利用类型名称
    landuse_evi_stats['c_factor'] = landuse_evi_stats['class_id'].map(c_factor_map)
    landuse_evi_stats['landuse_name'] = landuse_evi_stats['class_id'].map(landuse_names)

    # 剔除C因子为0的类别（这些类别不参与回归拟合）
    regression_data = landuse_evi_stats[landuse_evi_stats['c_factor'] > 0].copy()

    print("参与回归拟合的数据:")
    print("=" * 60)
    for _, row in regression_data.iterrows():
        print(f"{row['landuse_name']}({row['class_id']}): EVI均值={row['evi_mean']:.4f}, C因子={row['c_factor']:.3f}")

    # 准备数据
    x_data = regression_data['evi_mean'].values
    y_data = regression_data['c_factor'].values

    print("\n" + "=" * 60)
    print("多种模型拟合结果比较:")
    print("=" * 60)

    # 拟合多种模型
    model_results = fit_multiple_models(x_data, y_data)

    # 选择最佳模型（最高R²）
    best_model_name = None
    best_r_squared = -np.inf

    for model_name, result in model_results.items():
        if result is not None and result['r_squared'] > best_r_squared:
            best_r_squared = result['r_squared']
            best_model_name = model_name

    if best_model_name is None:
        print("所有模型拟合均失败")
        return None

    best_result = model_results[best_model_name]

    print("\n" + "=" * 60)
    print(f"最佳模型: {best_model_name}")
    print("=" * 60)

    # 输出最佳模型详情
    regression_data['predicted_c'] = best_result['predictions']
    regression_data['residual'] = y_data - best_result['predictions']

    print(f"模型方程: {get_model_equation(best_model_name, best_result['parameters'])}")
    print(f"参数值: {best_result['parameters']}")
    print(f"R²: {best_result['r_squared']:.6f}")
    print(f"RMSE: {best_result['rmse']:.6f}")

    print("\n拟合详情:")
    print("=" * 60)
    for _, row in regression_data.iterrows():
        print(
            f"{row['landuse_name']}: 实际C={row['c_factor']:.3f}, 预测C={row['predicted_c']:.3f}, 残差={row['residual']:.3f}")

    # 输出拟合图表
    if output_plot:
        plot_all_models_results(regression_data, model_results, best_model_name, landuse_names)

    return {
        'best_model_name': best_model_name,
        'best_parameters': best_result['parameters'],
        'best_function': best_result['function'],
        'r_squared': best_result['r_squared'],
        'rmse': best_result['rmse'],
        'regression_data': regression_data,
        'all_data': landuse_evi_stats,
        'all_models': model_results
    }


def get_model_equation(model_name, parameters):
    """根据模型名称和参数生成方程字符串"""
    if model_name == '指数模型':
        return f"C = {parameters[0]:.6f} × exp(-{parameters[1]:.6f} × EVI)"
    elif model_name == '线性模型':
        return f"C = {parameters[0]:.6f} × EVI + {parameters[1]:.6f}"
    elif model_name == '幂函数模型':
        return f"C = {parameters[0]:.6f} × EVI^{parameters[1]:.6f}"
    elif model_name == '对数模型':
        return f"C = {parameters[0]:.6f} × ln(EVI) + {parameters[1]:.6f}"
    elif model_name == '二次多项式':
        return f"C = {parameters[0]:.6f} × EVI² + {parameters[1]:.6f} × EVI + {parameters[2]:.6f}"
    elif model_name == 'S型函数':
        return f"C = {parameters[0]:.6f} / (1 + exp(-{parameters[1]:.6f} × (EVI - {parameters[2]:.6f})))"
    else:
        return "未知模型"


def plot_all_models_results(regression_data, model_results, best_model_name, landuse_names):
    """绘制所有模型的拟合结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 生成平滑曲线数据
    x_fit = np.linspace(regression_data['evi_mean'].min() - 0.05,
                        regression_data['evi_mean'].max() + 0.05, 100)

    colors = plt.cm.Set3(np.linspace(0, 1, len(regression_data)))

    # 图1: 所有模型对比
    ax = axes[0]
    for i, (_, row) in enumerate(regression_data.iterrows()):
        ax.scatter(row['evi_mean'], row['c_factor'],
                   color=colors[i], s=100, label=row['landuse_name'], alpha=0.7, zorder=5)

    for model_name, result in model_results.items():
        if result is not None:
            try:
                if len(result['parameters']) == 2:
                    y_fit = result['function'](x_fit, *result['parameters'])
                else:
                    y_fit = result['function'](x_fit, *result['parameters'])

                linestyle = '-' if model_name == best_model_name else '--'
                linewidth = 3 if model_name == best_model_name else 1.5
                alpha = 1.0 if model_name == best_model_name else 0.7

                ax.plot(x_fit, y_fit, label=f'{model_name} (R²={result["r_squared"]:.4f})',
                        linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            except:
                continue

    ax.set_xlabel('EVI均值')
    ax.set_ylabel('C因子')
    ax.set_title('所有模型拟合对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图2: 最佳模型详细图
    ax = axes[1]
    best_result = model_results[best_model_name]

    for i, (_, row) in enumerate(regression_data.iterrows()):
        ax.scatter(row['evi_mean'], row['c_factor'],
                   color=colors[i], s=100, label=row['landuse_name'], alpha=0.7, zorder=5)
        ax.annotate(row['landuse_name'],
                    (row['evi_mean'], row['c_factor']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    if len(best_result['parameters']) == 2:
        y_fit = best_result['function'](x_fit, *best_result['parameters'])
    else:
        y_fit = best_result['function'](x_fit, *best_result['parameters'])

    ax.plot(x_fit, y_fit, 'r-', linewidth=3,
            label=f'{best_model_name}\nR²={best_result["r_squared"]:.4f}')

    ax.set_xlabel('EVI均值')
    ax.set_ylabel('C因子')
    ax.set_title(f'最佳模型: {best_model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: 残差分析
    ax = axes[2]
    ax.scatter(regression_data['predicted_c'], regression_data['residual'],
               color='blue', s=80, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    for i, (_, row) in enumerate(regression_data.iterrows()):
        ax.annotate(row['landuse_name'],
                    (row['predicted_c'], row['residual']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('预测C因子值')
    ax.set_ylabel('残差 (实际-预测)')
    ax.set_title('残差分析图')
    ax.grid(True, alpha=0.3)

    # 图4: R²比较图
    ax = axes[3]
    model_names = []
    r_squared_values = []

    for model_name, result in model_results.items():
        if result is not None:
            model_names.append(model_name)
            r_squared_values.append(result['r_squared'])

    colors_bar = ['red' if name == best_model_name else 'skyblue' for name in model_names]
    bars = ax.bar(model_names, r_squared_values, color=colors_bar, alpha=0.7)

    ax.set_xlabel('模型')
    ax.set_ylabel('R²值')
    ax.set_title('各模型R²值比较')
    ax.tick_params(axis='x', rotation=45)

    # 在柱子上添加数值
    for bar, value in zip(bars, r_squared_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('All_Models_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def apply_best_model_to_yearly_data(evi_data_folder, best_function, best_parameters):
    """将最佳模型应用到年度数据上，计算每年的C因子"""
    yearly_results = []

    for year in range(1990, 2025):
        try:
            file_path = f"{evi_data_folder}/EVI_CLCD_{year}.xlsx"
            df = pd.read_excel(file_path)

            # 剔除异常值
            df = df[
                (df['evi_mean'] <= 1) &
                (df['evi_mean'] >= -1) &
                (df['evi_std'] <= 1)
                ]

            # 计算每个类别的C因子
            df['predicted_c'] = best_function(df['evi_mean'], *best_parameters)

            # 按类别统计
            for _, row in df.iterrows():
                yearly_results.append({
                    'year': year,
                    'class_id': row['class_id'],
                    'evi_mean': row['evi_mean'],
                    'predicted_c': row['predicted_c'],
                    'count': row['count']
                })

        except Exception as e:
            print(f"处理{year}年数据时出错: {e}")
            continue

    yearly_df = pd.DataFrame(yearly_results)
    return yearly_df


# 主程序
if __name__ == "__main__":
    folder_path = r"./data\CLCD\EVI_stats"

    # 拟合回归模型
    print("开始拟合EVI-C因子回归模型...")
    result = fit_evi_c_model(folder_path, output_plot=True)

    if result is not None:
        best_function = result['best_function']
        best_parameters = result['best_parameters']
        best_model_name = result['best_model_name']

        print("\n" + "=" * 60)
        print("模型应用:")
        print("=" * 60)

        # 将最佳模型应用到年度数据
        yearly_predictions = apply_best_model_to_yearly_data(folder_path, best_function, best_parameters)

        # 计算每年的加权平均C因子
        yearly_weighted_c = []
        for year in yearly_predictions['year'].unique():
            year_data = yearly_predictions[yearly_predictions['year'] == year]
            total_count = year_data['count'].sum()
            weighted_c = np.average(year_data['predicted_c'], weights=year_data['count'])

            yearly_weighted_c.append({
                'year': year,
                'weighted_c_factor': weighted_c,
                'total_pixels': total_count
            })

        yearly_summary = pd.DataFrame(yearly_weighted_c)

        print("\n年度加权平均C因子:")
        print("=" * 40)
        print(yearly_summary.round(6))

        # 绘制时间序列图
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_summary['year'], yearly_summary['weighted_c_factor'],
                 marker='o', linewidth=2, markersize=6)
        plt.xlabel('年份')
        plt.ylabel('加权平均C因子')
        plt.title(f'1990-2024年广东省加权平均C因子时间序列 (使用{best_model_name})')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Yearly_C_Factor_Trend.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 保存结果
        yearly_predictions.to_excel(os.path.join(OUTPUT_DIR, 'EVI_C_Predictions_Yearly.xlsx'), index=False)
        yearly_summary.to_excel(os.path.join(OUTPUT_DIR, 'Yearly_Weighted_C_Factor.xlsx'), index=False)

        # 保存模型信息
        model_info = pd.DataFrame({
            '最佳模型': [best_model_name],
            '方程': [get_model_equation(best_model_name, best_parameters)],
            'R²': [result['r_squared']],
            'RMSE': [result['rmse']],
            '参数': [best_parameters]
        })
        model_info.to_excel(os.path.join(OUTPUT_DIR, 'Best_Model_Info.xlsx'), index=False)

        print(f"\n结果已保存到:")
        print("- EVI_C_Predictions_Yearly.xlsx (年度详细预测)")
        print("- Yearly_Weighted_C_Factor.xlsx (年度汇总)")
        print("- Best_Model_Info.xlsx (最佳模型信息)")
        print("- All_Models_Comparison.png (所有模型对比图)")
        print("- Yearly_C_Factor_Trend.png (时间序列图)")