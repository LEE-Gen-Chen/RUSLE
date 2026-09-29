import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import math


def resample_dem_to_slope_size(dem_path, slope_path, output_path):
    """
    将DEM重采样到坡度数据的尺寸
    """
    # 读取坡度数据获取目标尺寸和变换
    with rasterio.open(slope_path) as slope_src:
        slope_profile = slope_src.profile.copy()
        slope_shape = slope_src.shape
        slope_transform = slope_src.transform
        slope_crs = slope_src.crs

    # 读取DEM数据
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_profile = dem_src.profile.copy()

    # 创建目标数组
    dem_resampled = np.zeros(slope_shape, dtype=rasterio.float32)

    # 重投影和重采样
    reproject(
        source=dem_data,
        destination=dem_resampled,
        src_transform=dem_profile['transform'],
        src_crs=dem_profile['crs'],
        dst_transform=slope_transform,
        dst_crs=slope_crs,
        resampling=Resampling.bilinear
    )

    # 保存重采样后的DEM
    slope_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )

    with rasterio.open(output_path, 'w', **slope_profile) as dst:
        dst.write(dem_resampled, 1)

    return dem_resampled, slope_profile


def calculate_dynamic_lambda(dem_data, slope_degrees, cell_size):
    """
    修正的λ值计算 - 投影坡长
    """
    slope_radians = np.radians(slope_degrees)

    # 投影坡长 = 网格大小 / cosθ
    cos_theta = np.cos(slope_radians)
    cos_theta = np.where(cos_theta < 0.001, 0.001, cos_theta)  # 避免除零

    lambda_projected = cell_size / cos_theta

    # 应用限制范围
    lambda_projected = np.where(lambda_projected < 30, 30, lambda_projected)
    lambda_projected = np.where(lambda_projected > 1000, 1000, lambda_projected)

    return lambda_projected


def calculate_slope_factor(slope_degrees):
    """
    修正的坡度因子S计算
    """
    slope_radians = np.radians(slope_degrees)
    sin_theta = np.sin(slope_radians)

    s_factor = np.zeros_like(slope_degrees)

    mask1 = slope_degrees <= 5
    mask2 = (slope_degrees > 5) & (slope_degrees < 10)
    mask3 = slope_degrees >= 10

    s_factor[mask1] = 10.8 * sin_theta[mask1] + 0.03
    s_factor[mask2] = 16.8 * sin_theta[mask2] - 0.50
    s_factor[mask3] = 21.97 * sin_theta[mask3] - 0.96  # 修正系数

    return s_factor


def calculate_length_factor(slope_degrees, lambda_values):
    """
    计算坡长因子L
    """
    slope_radians = np.radians(slope_degrees)
    sin_theta = np.sin(slope_radians)

    beta = (sin_theta / 0.0896) / (3 * np.power(sin_theta, 0.8) + 0.56)
    m = beta / (1 + beta)
    L = np.power(lambda_values / 22.13, m)

    return L


def main():
    # 文件路径
    dem_path = r"./data\Geoscene\GD_DEM.tif"
    slope_path = r"./data\Geoscene\Slope.tif"
    dem_resampled_path = r"./data\Geoscene\DEM_Resampled.tif"
    output_s_path = r"./data\Geoscene\S_Factor.tif"
    output_l_path = r"./data\Geoscene\L_Factor.tif"
    output_ls_path = r"./data\Geoscene\LS_Factor.tif"  # 新增LS综合因子

    try:
        print("步骤1: 重采样DEM到坡度数据尺寸...")
        dem_resampled, profile = resample_dem_to_slope_size(dem_path, slope_path, dem_resampled_path)

        # 读取坡度数据
        with rasterio.open(slope_path) as src_slope:
            slope_data = src_slope.read(1)

        # 获取网格大小（从变换矩阵中）
        cell_size = profile['transform'][0]  # 通常为网格分辨率

        # 计算坡度因子S（修正后）
        print("计算坡度因子S...")
        s_factor = calculate_slope_factor(slope_data)

        # 计算投影坡长λ
        print("计算投影坡长λ...")
        lambda_projected = calculate_dynamic_lambda(dem_resampled, slope_data, cell_size)

        # 计算坡长因子L
        print("计算坡长因子L...")
        l_factor = calculate_length_factor(slope_data, lambda_projected)

        # 计算LS综合因子
        ls_factor = l_factor * s_factor

        # 保存结果
        profile.update(dtype=rasterio.float32, compress='lzw')

        with rasterio.open(output_s_path, 'w', **profile) as dst:
            dst.write(s_factor.astype(rasterio.float32), 1)

        with rasterio.open(output_l_path, 'w', **profile) as dst:
            dst.write(l_factor.astype(rasterio.float32), 1)

        with rasterio.open(output_ls_path, 'w', **profile) as dst:
            dst.write(ls_factor.astype(rasterio.float32), 1)

        print("所有因子计算完成并保存！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()