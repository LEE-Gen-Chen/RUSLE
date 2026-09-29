import os
import rasterio
from rasterio import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
import numpy as np
import geopandas as gpd
from rasterio.transform import from_origin
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class RUSLECalculator:
    def __init__(self, target_resolution=30, target_crs="EPSG:4547", resolution_tolerance=0.005):
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.boundary = None
        self.resolution_tolerance = resolution_tolerance

    def load_boundary(self, boundary_path):
        """加载广东省边界"""
        print("加载广东省边界...")
        self.boundary = gpd.read_file(boundary_path)
        self.boundary = self.boundary.to_crs(self.target_crs)
        print(f"广东省边界加载完成，坐标系: {self.target_crs}")

    def is_resolution_within_tolerance(self, actual_resolution):
        """检查分辨率是否在容差范围内"""
        lower_bound = self.target_resolution * (1 - self.resolution_tolerance)
        upper_bound = self.target_resolution * (1 + self.resolution_tolerance)
        return lower_bound <= actual_resolution <= upper_bound

    def check_raster_compatibility(self, input_path):
        """检查栅格数据是否与目标坐标系和分辨率兼容"""
        with rasterio.open(input_path) as src:
            # 检查坐标系
            crs_match = src.crs == self.target_crs

            # 检查分辨率（使用容差）
            resolution_match = self.is_resolution_within_tolerance(src.res[0])

            # 检查范围是否在广东省边界内（简单检查）
            if self.boundary is not None:
                src_bounds = src.bounds
                bounds = self.boundary.total_bounds
                # 如果源数据范围小于或等于广东边界范围，认为可能已经裁剪过
                bounds_check = (src_bounds.left >= bounds[0] and
                                src_bounds.right <= bounds[2] and
                                src_bounds.top <= bounds[3] and
                                src_bounds.bottom >= bounds[1])
            else:
                bounds_check = False

            return crs_match and resolution_match, bounds_check

    def unify_raster(self, input_path, output_path=None):
        """统一栅格数据的坐标系和分辨率，如果需要的话"""
        print(f"处理栅格数据: {os.path.basename(input_path)}")

        # 检查是否已经符合要求
        compatible, _ = self.check_raster_compatibility(input_path)
        if compatible:
            print(f"  已符合目标坐标系和分辨率，跳过处理")
            return input_path

        with rasterio.open(input_path) as src:
            # 检查是否需要重投影或重采样
            crs_match = src.crs == self.target_crs
            resolution_match = self.is_resolution_within_tolerance(src.res[0])

            if not crs_match or not resolution_match:
                print(f"  需要重投影/重采样: {src.crs} -> {self.target_crs}, "
                      f"分辨率: {src.res[0]:.6f} -> {self.target_resolution}m")

                # 计算重投影后的transform和尺寸
                transform, width, height = calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height, *src.bounds,
                    resolution=self.target_resolution
                )

                # 创建目标文件metadata
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                if output_path is None:
                    output_path = input_path.replace('.tif', '_unified.tif')

                # 执行重投影和重采样
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=self.target_crs,
                            resampling=Resampling.bilinear
                        )

                print(f"  统一完成: {output_path}")
                return output_path
            else:
                print(f"  无需处理，已符合目标坐标系和分辨率")
                return input_path

    def clip_to_study_area(self, input_path, output_path=None):
        """将栅格数据裁剪到广东省范围内，如果需要的话"""
        if self.boundary is None:
            raise ValueError("请先加载广东省边界")

        print(f"检查栅格数据: {os.path.basename(input_path)}")

        # 检查是否已经裁剪过
        compatible, bounds_check = self.check_raster_compatibility(input_path)
        if compatible and bounds_check:
            print(f"  已符合目标要求且范围合适，跳过裁剪")
            return input_path

        with rasterio.open(input_path) as src:
            # 确保坐标系一致
            if src.crs != self.boundary.crs:
                raise ValueError(f"栅格数据坐标系 {src.crs} 与边界坐标系 {self.boundary.crs} 不一致")

            try:
                print(f"  执行裁剪...")
                # 执行裁剪
                out_image, out_transform = mask.mask(src, self.boundary.geometry,
                                                     crop=True, all_touched=True)
                out_meta = src.meta.copy()

                # 更新metadata
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                if output_path is None:
                    output_path = input_path.replace('.tif', '_clipped.tif')

                # 保存裁剪结果
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"  裁剪完成: {output_path}")
                return output_path

            except Exception as e:
                print(f"  裁剪失败: {e}")
                return input_path

    def find_common_extent(self, raster_paths):
        """找到所有栅格数据的公共范围"""
        bounds_list = []
        transforms = []
        shapes = []

        for path in raster_paths:
            with rasterio.open(path) as src:
                bounds_list.append(src.bounds)
                transforms.append(src.transform)
                shapes.append(src.shape)

        # 计算公共边界
        common_left = max(bounds.left for bounds in bounds_list)
        common_right = min(bounds.right for bounds in bounds_list)
        common_bottom = max(bounds.bottom for bounds in bounds_list)
        common_top = min(bounds.top for bounds in bounds_list)

        return (common_left, common_bottom, common_right, common_top)

    def align_rasters(self, R_path, K_path, LS_path, C_path, P_path):
        """对齐所有栅格数据到公共范围"""
        print("对齐所有栅格数据到公共范围...")

        raster_paths = [R_path, K_path, LS_path, C_path, P_path]
        common_bounds = self.find_common_extent(raster_paths)

        print(f"公共范围: {common_bounds}")

        aligned_data = {}
        common_shape = None
        profiles = {}

        # 首先确定公共形状
        for path in raster_paths:
            with rasterio.open(path) as src:
                # 计算窗口范围
                window = src.window(*common_bounds)

                # 确保窗口是整数
                row_start = int(np.floor(window.row_off))
                row_stop = int(np.ceil(window.row_off + window.height))
                col_start = int(np.floor(window.col_off))
                col_stop = int(np.ceil(window.col_off + window.width))

                # 确保窗口不超出范围
                row_start = max(0, row_start)
                row_stop = min(src.height, row_stop)
                col_start = max(0, col_start)
                col_stop = min(src.width, col_stop)

                # 计算实际窗口大小
                window_height = row_stop - row_start
                window_width = col_stop - col_start

                # 使用最大的窗口大小作为公共形状
                if common_shape is None:
                    common_shape = (window_height, window_width)
                else:
                    common_shape = (max(common_shape[0], window_height),
                                    max(common_shape[1], window_width))

        print(f"公共形状: {common_shape}")

        # 然后对齐所有栅格
        for path in raster_paths:
            with rasterio.open(path) as src:
                # 计算窗口范围
                window = src.window(*common_bounds)

                # 确保窗口是整数
                row_start = int(np.floor(window.row_off))
                row_stop = int(np.ceil(window.row_off + window.height))
                col_start = int(np.floor(window.col_off))
                col_stop = int(np.ceil(window.col_off + window.width))

                # 确保窗口不超出范围
                row_start = max(0, row_start)
                row_stop = min(src.height, row_stop)
                col_start = max(0, col_start)
                col_stop = min(src.width, col_stop)

                # 读取数据
                data = src.read(1, window=((row_start, row_stop), (col_start, col_stop)))

                # 保存profile信息
                profiles[path] = src.profile

                # 如果数据形状与公共形状不匹配，进行填充
                if data.shape != common_shape:
                    print(f"  调整 {os.path.basename(path)} 的形状: {data.shape} -> {common_shape}")

                    # 创建一个新的数组，用原始数据的NoData值或0填充
                    if src.nodata is not None:
                        fill_value = src.nodata
                    else:
                        fill_value = 0

                    new_data = np.full(common_shape, fill_value, dtype=data.dtype)

                    # 将原始数据复制到新数组中
                    min_rows = min(data.shape[0], common_shape[0])
                    min_cols = min(data.shape[1], common_shape[1])
                    new_data[:min_rows, :min_cols] = data[:min_rows, :min_cols]

                    data = new_data

                aligned_data[path] = data

        return aligned_data, common_shape, profiles, common_bounds

    def calculate_rusle(self, R_path, K_path, LS_path, C_path, P_path, output_path):
        """计算RUSLE土壤侵蚀模型"""
        print(f"计算RUSLE土壤侵蚀量...")

        try:
            # 对齐所有栅格数据到公共范围
            aligned_data, common_shape, profiles, common_bounds = self.align_rasters(R_path, K_path, LS_path, C_path,
                                                                                     P_path)

            # 提取对齐后的数据
            R = aligned_data[R_path]
            K = aligned_data[K_path]
            LS = aligned_data[LS_path]
            C = aligned_data[C_path]
            P = aligned_data[P_path]

            # 确保所有数组形状一致
            print(f"  对齐后数据形状 - R: {R.shape}, K: {K.shape}, LS: {LS.shape}, C: {C.shape}, P: {P.shape}")

            # 在无效值处理前检查原始数据范围
            print("  原始数据范围检查:")
            print(f"    R因子范围: {np.nanmin(R)} ~ {np.nanmax(R)}")
            print(f"    K因子范围: {np.nanmin(K)} ~ {np.nanmax(K)}")
            print(f"    LS因子范围: {np.nanmin(LS)} ~ {np.nanmax(LS)}")
            print(f"    C因子范围: {np.nanmin(C)} ~ {np.nanmax(C)}")
            print(f"    P因子范围: {np.nanmin(P)} ~ {np.nanmax(P)}")

            # 处理无效值：只将明显的无效值转换为NaN
            print("  处理无效值...")

            # 首先处理NoData值
            R_nodata = profiles[R_path].get('nodata', None)
            K_nodata = profiles[K_path].get('nodata', None)
            LS_nodata = profiles[LS_path].get('nodata', None)
            C_nodata = profiles[C_path].get('nodata', None)
            P_nodata = profiles[P_path].get('nodata', None)

            if R_nodata is not None:
                R = np.where(R == R_nodata, np.nan, R)
            if K_nodata is not None:
                K = np.where(K == K_nodata, np.nan, K)
            if LS_nodata is not None:
                LS = np.where(LS == LS_nodata, np.nan, LS)
            if C_nodata is not None:
                C = np.where(C == C_nodata, np.nan, C)
            if P_nodata is not None:
                P = np.where(P == P_nodata, np.nan, P)

            # 然后处理极端的无效值（根据各因子的合理范围）
            R = np.where(R < 0, np.nan, R)  # R因子应为非负
            K = np.where(K < 0, np.nan, K)  # K因子应为非负
            LS = np.where(LS < 0, np.nan, LS)  # LS因子应为非负
            C = np.where((C < 0) | (C > 1), np.nan, C)  # C因子应在0-1之间
            P = np.where((P < 0) | (P > 1), np.nan, P)  # P因子应在0-1之间

            # 统计各因子的有效像元数
            print(f"  有效像元统计 - R: {np.sum(~np.isnan(R))}, "
                  f"K: {np.sum(~np.isnan(K))}, "
                  f"LS: {np.sum(~np.isnan(LS))}, "
                  f"C: {np.sum(~np.isnan(C))}, "
                  f"P: {np.sum(~np.isnan(P))}")

            # 检查处理后的数据范围
            print("  处理后数据范围:")
            valid_R = R[~np.isnan(R)]
            valid_K = K[~np.isnan(K)]
            valid_LS = LS[~np.isnan(LS)]
            valid_C = C[~np.isnan(C)]
            valid_P = P[~np.isnan(P)]

            if len(valid_R) > 0:
                print(f"    R因子范围: {valid_R.min():.4f} ~ {valid_R.max():.4f}")
            if len(valid_K) > 0:
                print(f"    K因子范围: {valid_K.min():.4f} ~ {valid_K.max():.4f}")
            if len(valid_LS) > 0:
                print(f"    LS因子范围: {valid_LS.min():.4f} ~ {valid_LS.max():.4f}")
            if len(valid_C) > 0:
                print(f"    C因子范围: {valid_C.min():.4f} ~ {valid_C.max():.4f}")
            if len(valid_P) > 0:
                print(f"    P因子范围: {valid_P.min():.4f} ~ {valid_P.max():.4f}")

            # RUSLE计算: A = R × K × LS × C × P
            print("  执行RUSLE计算...")
            A = R * K * LS * C * P

            # 获取R因子的profile作为输出模板
            profile = profiles[R_path].copy()

            # 更新profile - 使用公共边界重新计算transform
            profile.update({
                'dtype': 'float32',
                'nodata': -9999,
                'height': common_shape[0],
                'width': common_shape[1],
                'transform': rasterio.transform.from_bounds(
                    common_bounds[0], common_bounds[1],  # left, bottom
                    common_bounds[2], common_bounds[3],  # right, top
                    common_shape[1], common_shape[0]  # width, height
                )
            })

            # 将NaN值设置为NoData
            A = np.where(np.isnan(A), -9999, A)

            # 保存结果
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(A.astype('float32'), 1)

            print(f"  RUSLE计算完成: {output_path}")

            # 统计信息
            valid_A = A[A != -9999]
            if len(valid_A) > 0:
                print(f"  侵蚀量统计 - 最小值: {valid_A.min():.4f}, "
                      f"最大值: {valid_A.max():.4f}, "
                      f"平均值: {valid_A.mean():.4f} t/ha/yr")
                print(f"  有效像元数量: {len(valid_A)}")
            else:
                print("  警告: 没有有效的计算结果!")

            return output_path

        except Exception as e:
            print(f"  RUSLE计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_factor(self, input_path, output_path=None, force_reprocess=False):
        """预处理因子：统一坐标系、分辨率并裁剪到广东省范围（如果需要）"""
        if output_path is None:
            output_dir = os.path.join(os.path.dirname(input_path), "preprocessed")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir,
                                       f"preprocessed_{os.path.basename(input_path)}")

        # 如果输出文件已存在且不强制重处理，则直接返回
        if os.path.exists(output_path) and not force_reprocess:
            print(f"使用已预处理的文件: {os.path.basename(output_path)}")
            return output_path

        # 统一坐标系和分辨率（如果需要）
        unified_path = self.unify_raster(input_path)

        # 裁剪到广东省范围（如果需要）
        final_path = self.clip_to_study_area(unified_path, output_path)
        return final_path


def main():
    # 初始化计算器，设置分辨率容差为0.005（0.5%）
    calculator = RUSLECalculator(target_resolution=30, target_crs="EPSG:4547", resolution_tolerance=0.005)

    # 文件路径配置
    base_path = r"./data"
    boundary_path = os.path.join(base_path, "./data", "GD.shp")

    # 因子路径模板
    R_template = os.path.join(base_path, "Precipitation", "R_output", "R_{year}.tif")
    C_template = os.path.join(base_path, "NDVI", "C_value_30m", "C_NDVI_{year}.tif")
    K_path = os.path.join(base_path, "soil", "HWSD_Resample", "RUSLE_K_30m.tif")
    LS_path = os.path.join(base_path, "./data", "LS_Factor.tif")
    P_template = os.path.join(base_path, "CLCD", "P_factor", "P{year}.tif")

    # 输出目录
    output_dir = os.path.join(base_path, "RUSLE_Results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载广东省边界
    calculator.load_boundary(boundary_path)

    # 2. 预处理静态因子（K和LS因子）
    print("\n" + "=" * 50)
    print("处理静态因子...")

    # 预处理K因子（只有在需要时）
    K_final = calculator.preprocess_factor(K_path)

    # 预处理LS因子（只有在需要时）
    LS_final = calculator.preprocess_factor(LS_path)

    # 3. 逐年计算RUSLE
    print("\n" + "=" * 50)
    print("开始逐年计算RUSLE...")

    # 创建年份列表
    years = list(range(1990, 2025))

    # 使用tqdm显示进度条
    progress_bar = tqdm(years, desc="处理年份", unit="年")

    successful_years = []
    failed_years = []

    for year in progress_bar:
        progress_bar.set_description(f"处理 {year} 年")

        # 构建文件路径
        R_path = R_template.format(year=year)
        C_path = C_template.format(year=year)
        P_path = P_template.format(year=year)

        # 检查文件是否存在
        missing_files = []
        for path, name in [(R_path, "R因子"), (C_path, "C因子"), (P_path, "P因子")]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")

        if missing_files:
            progress_bar.write(f"  {year}年缺少文件，跳过:")
            for missing in missing_files:
                progress_bar.write(f"    {missing}")
            failed_years.append(year)
            continue

        try:
            # 预处理动态因子（只有在需要时）
            R_final = calculator.preprocess_factor(R_path)
            C_final = calculator.preprocess_factor(C_path)
            P_final = calculator.preprocess_factor(P_path)

            # 计算RUSLE
            output_path = os.path.join(output_dir, f"RUSLE_{year}.tif")
            result = calculator.calculate_rusle(R_final, K_final, LS_final, C_final, P_final, output_path)

            if result:
                progress_bar.write(f"  {year}年土壤侵蚀计算完成!")
                successful_years.append(year)
            else:
                progress_bar.write(f"  {year}年土壤侵蚀计算失败!")
                failed_years.append(year)

        except Exception as e:
            progress_bar.write(f"  处理{year}年时出错: {e}")
            failed_years.append(year)
            continue

    # 关闭进度条
    progress_bar.close()

    # 输出总结信息
    print("\n" + "=" * 50)
    print("RUSLE计算完成!")
    print(f"成功处理的年份: {len(successful_years)}年")
    if successful_years:
        print(f"成功年份列表: {successful_years}")
    if failed_years:
        print(f"失败/跳过的年份: {len(failed_years)}年")
        print(f"失败年份列表: {failed_years}")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()