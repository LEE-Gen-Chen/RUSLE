import rasterio
import numpy as np
import os
from rasterio.transform import from_bounds
import warnings

warnings.filterwarnings('ignore')


def debug_silt_data():
    """
    调试粉粒数据，找出问题所在
    """
    base_path = r"./data\中国土壤数据集\GEE_Soil"
    silt_path = os.path.join(base_path, "GD_silt.tif")

    print("=== 粉粒数据调试信息 ===")

    with rasterio.open(silt_path) as src:
        silt_raw = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata

        print(f"原始粉粒数据范围: {np.nanmin(silt_raw):.2f} - {np.nanmax(silt_raw):.2f}")
        print(f"NoData值: {nodata}")
        print(f"数据形状: {silt_raw.shape}")
        print(f"数据类型: {silt_raw.dtype}")

        # 统计原始数据分布
        unique_vals = np.unique(silt_raw[~np.isnan(silt_raw)])
        print(f"唯一值数量: {len(unique_vals)}")
        if len(unique_vals) < 20:  # 如果唯一值不多，打印出来
            print(f"唯一值: {unique_vals}")

        # 检查数据是否全为0或接近0
        zero_count = np.sum(silt_raw == 0)
        nan_count = np.sum(np.isnan(silt_raw))
        print(f"零值像素数量: {zero_count}")
        print(f"NaN像素数量: {nan_count}")
        print(f"总像素数量: {silt_raw.size}")

        # 尝试不同的变换方法
        print("\n=== 不同变换方法的结果 ===")

        # 方法1: 原始说明的变换
        silt_method1 = np.exp(silt_raw / 10) - 1
        print(f"方法1 (exp(x/10)-1) 范围: {np.nanmin(silt_method1):.4f} - {np.nanmax(silt_method1):.4f}")

        # 方法2: 可能是指数底数问题
        silt_method2 = np.exp(silt_raw / 10.0) - 1.0
        print(f"方法2 (exp(x/10.0)-1.0) 范围: {np.nanmin(silt_method2):.4f} - {np.nanmax(silt_method2):.4f}")

        # 方法3: 可能是10的指数变换
        silt_method3 = np.power(10, silt_raw / 10) - 1
        print(f"方法3 (10^(x/10)-1) 范围: {np.nanmin(silt_method3):.4f} - {np.nanmax(silt_method3):.4f}")

        # 方法4: 直接除以10（可能是简单的缩放）
        silt_method4 = silt_raw / 10.0
        print(f"方法4 (x/10) 范围: {np.nanmin(silt_method4):.4f} - {np.nanmax(silt_method4):.4f}")

        # 方法5: 可能不需要变换
        silt_method5 = silt_raw
        print(f"方法5 (原始数据) 范围: {np.nanmin(silt_method5):.4f} - {np.nanmax(silt_method5):.4f}")

        return silt_raw, profile


def calculate_rusle_k_local():
    """
    基于本地TIFF文件计算RUSLE K值
    """
    # 数据路径
    base_path = r"./data\中国土壤数据集\GEE_Soil"
    sand_path = os.path.join(base_path, "GD_sand.tif")
    silt_path = os.path.join(base_path, "GD_silt.tif")
    clay_path = os.path.join(base_path, "GD_clay.tif")
    soc_path = os.path.join(base_path, "GD_soc.tif")

    # 输出路径
    output_path = os.path.join(base_path, "GD_K.tif")

    print("=== 数据读取与预处理 ===")

    # 读取砂粒数据
    with rasterio.open(sand_path) as src:
        sand = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # 读取黏粒数据
    with rasterio.open(clay_path) as src:
        clay = src.read(1).astype(np.float32)

    # 调试粉粒数据
    silt_raw, _ = debug_silt_data()

    # 根据调试结果选择合适的变换方法
    # 如果原始数据范围合理，可能不需要变换
    if np.nanmin(silt_raw) >= 0 and np.nanmax(silt_raw) <= 100:
        print("使用原始粉粒数据（可能不需要变换）")
        silt = silt_raw
    else:
        # 尝试不同的变换方法，选择产生合理范围的那个
        silt = np.exp(silt_raw / 10.0) - 1.0
        print(f"使用变换后的粉粒数据，范围: {np.nanmin(silt):.2f} - {np.nanmax(silt):.2f}%")

    # 读取有机碳数据并转换单位: (5g/kg * 5) / 10 = 2.5%
    with rasterio.open(soc_path) as src:
        soc_raw = src.read(1).astype(np.float32)
        soc_percent = (soc_raw * 5) / 10  # 转换为百分比

    # 处理无效值
    def handle_nodata(data, nodata_value):
        if nodata_value is not None:
            data[data == nodata_value] = np.nan
        data[data < 0] = np.nan
        data[data > 1000] = np.nan  # 处理异常高值
        return data

    sand = handle_nodata(sand, nodata)
    silt = handle_nodata(silt, nodata)
    clay = handle_nodata(clay, nodata)
    soc_percent = handle_nodata(soc_percent, nodata)

    # 确保土壤成分在合理范围内 (0-100%)
    sand = np.clip(sand, 0, 100)
    silt = np.clip(silt, 0, 100)
    clay = np.clip(clay, 0, 100)
    soc_percent = np.clip(soc_percent, 0, 20)  # 有机碳通常不超过20%

    print("\n=== 数据质量检查 ===")
    print(f"砂粒范围: {np.nanmin(sand):.2f} - {np.nanmax(sand):.2f}%")
    print(f"粉粒范围: {np.nanmin(silt):.2f} - {np.nanmax(silt):.2f}%")
    print(f"黏粒范围: {np.nanmin(clay):.2f} - {np.nanmax(clay):.2f}%")
    print(f"有机碳范围: {np.nanmin(soc_percent):.2f} - {np.nanmax(soc_percent):.2f}%")

    # 检查土壤成分总和
    soil_total = sand + silt + clay
    print(f"土壤成分总和范围: {np.nanmin(soil_total):.1f} - {np.nanmax(soil_total):.1f}%")

    # 计算SN1 = 1 - SAN/100
    SN1 = 1 - sand / 100

    # 初始化K值数组
    K = np.zeros_like(sand, dtype=np.float32)

    # 创建有效数据掩码
    valid_mask = (~np.isnan(sand)) & (~np.isnan(silt)) & (~np.isnan(clay)) & (~np.isnan(soc_percent))
    valid_mask = valid_mask & (sand + silt + clay > 0)  # 确保土壤成分之和大于0

    # 提取有效数据
    sand_valid = sand[valid_mask]
    silt_valid = silt[valid_mask]
    clay_valid = clay[valid_mask]
    soc_valid = soc_percent[valid_mask]
    SN1_valid = SN1[valid_mask]

    print(f"\n=== 有效数据统计 ===")
    print(f"有效像素数量: {np.sum(valid_mask)}")

    # 如果粉粒数据仍然有问题，使用替代方法
    if np.nanmax(silt_valid) < 1:  # 如果粉粒最大值小于1%
        print("警告：粉粒数据可能有问题，尝试使用替代方法...")
        # 方法1: 根据土壤质地估算粉粒
        # 粉粒 ≈ 100 - 砂粒 - 黏粒
        silt_estimated = 100 - sand - clay
        silt_estimated = np.clip(silt_estimated, 0, 100)
        silt_valid = silt_estimated[valid_mask]
        print(f"估算粉粒范围: {np.nanmin(silt_valid):.2f} - {np.nanmax(silt_valid):.2f}%")

    # 计算K值的各个部分
    print("\n=== K值计算 ===")

    # 第一部分: 0.2 + 0.3 × exp[-0.0256 × SAN × (1 - SIL/100)]
    part1 = 0.2 + 0.3 * np.exp(-0.0256 * sand_valid * (1 - silt_valid / 100))
    print(f"第一部分范围: {np.nanmin(part1):.4f} - {np.nanmax(part1):.4f}")

    # 第二部分: (SIL / (CLA + SIL))^0.3
    denominator = clay_valid + silt_valid
    denominator[denominator == 0] = np.nan
    part2 = np.power(silt_valid / denominator, 0.3)
    print(f"第二部分范围: {np.nanmin(part2):.4f} - {np.nanmax(part2):.4f}")

    # 第三部分: 1 - (0.25 × C) / (C + exp(3.72 - 2.95 × C))
    numerator3 = 0.25 * soc_valid
    denominator3 = soc_valid + np.exp(3.72 - 2.95 * soc_valid)
    denominator3[denominator3 == 0] = np.nan
    part3 = 1 - numerator3 / denominator3
    print(f"第三部分范围: {np.nanmin(part3):.4f} - {np.nanmax(part3):.4f}")

    # 第四部分: 1 - (0.7 × SN1) / (SN1 + exp(-5.51 + 22.9 × SN1))
    numerator4 = 0.7 * SN1_valid
    denominator4 = SN1_valid + np.exp(-5.51 + 22.9 * SN1_valid)
    denominator4[denominator4 == 0] = np.nan
    part4 = 1 - numerator4 / denominator4
    print(f"第四部分范围: {np.nanmin(part4):.4f} - {np.nanmax(part4):.4f}")

    # 计算最终的K值
    K_valid = part1 * part2 * part3 * part4

    # 将有效值填充回K数组
    K[valid_mask] = K_valid

    # 处理可能的异常值
    K = np.clip(K, 0, 0.5)  # K值通常在0-0.5之间
    K[~valid_mask] = -9999  # 设置无效值为-9999

    print(f"\n=== 最终结果 ===")
    print(f"K值范围: {np.nanmin(K_valid):.4f} - {np.nanmax(K_valid):.4f}")
    print(f"K值均值: {np.nanmean(K_valid):.4f}")

    # 更新输出文件的元数据
    profile.update({
        'dtype': rasterio.float32,
        'nodata': -9999,
        'count': 1,
        'compress': 'lzw'
    })

    # 写入输出文件
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(K.astype(np.float32), 1)

    print(f"K因子计算完成，结果已保存至: {output_path}")

    return K, output_path


if __name__ == "__main__":
    # 计算K因子
    K_result, output_file = calculate_rusle_k_local()
    print("RUSLE K因子计算完成！")