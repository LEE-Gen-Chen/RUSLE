'''
本环节通过计算 Arnoldus[26] 修正的月降雨侵蚀力公式,来模拟降雨侵蚀力。
Wei S G. A China data set of soil properties for land surface  modeling[J]. Journal of Advances in Modeling Earth Systems,  2013, 2(5): 212 − 224.
选择原因：基于月雨量指标的月降雨侵蚀力模型对数据要求精度不高,运算简捷,年度降雨侵蚀力的估算效果较好。
1、适合广东省降水变化大的区域，广东省为亚热带季风区，降水年际不稳定；
2、土壤侵蚀估计必须考虑降雨的年内分布，季节性分布对侵蚀动力学至关重要；
3、当降水极不均匀时，公式需体现降水集中度的影响，这正是第一种公式的设计初衷。相比之下，线性公式忽略了月度极端降水的非线性效应，在广东这种季风气候下容易低估高强度暴雨的侵蚀力。
'''
# RUSLE_R.py - 使用Numba GPU加速版本
import os
import glob
import rasterio
import rasterio.mask
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import cuda, jit
import math

# =========== 配置区域 =============
GD_SHP = r"./data/boundary.shp"
MONTHLY_DIR = r"./data/precipitation/monthly"
ANNUAL_DIR = r"./data/precipitation/annual"
OUT_DIR = r"./data/R_output"
os.makedirs(OUT_DIR, exist_ok=True)

# 读取广东 shp
gdf = gpd.read_file(GD_SHP)
if gdf.empty:
    raise ValueError("GD.shp 内容为空或路径错误")
geoms = [feature["geometry"] for feature in gdf.__geo_interface__['features']]

# 分块大小
CHUNK_SIZE = 500

# 检查GPU可用性
try:
    # 检查是否有可用的CUDA设备
    if cuda.is_available():
        USE_GPU = True
        gpu_device = cuda.get_current_device()
        print(f"GPU加速已启用 - 使用 {gpu_device.name}")
        print(f"GPU计算能力: {gpu_device.compute_capability}")
    else:
        USE_GPU = False
        print("GPU不可用，使用CPU计算")
except:
    USE_GPU = False
    print("GPU检测失败，使用CPU计算")


def check_output_exists(year, out_dir):
    """检查输出文件是否已存在"""
    tif_path = os.path.join(out_dir, f"R_{year}.tif")
    csv_path = os.path.join(out_dir, f"R_{year}.csv")

    # 检查两个文件是否都存在
    tif_exists = os.path.exists(tif_path) and os.path.getsize(tif_path) > 0
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    if tif_exists and csv_exists:
        print(f"年份 {year} 的输出文件已存在，跳过处理")
        return True
    elif tif_exists or csv_exists:
        # 如果只有一个文件存在，可能是之前处理中断，重新处理
        print(f"年份 {year} 的输出文件不完整，重新处理")
        return False
    else:
        # 两个文件都不存在
        return False


@cuda.jit
def compute_R_kernel(P, Pi, R_sum):
    """GPU核函数：计算R值"""
    i, j = cuda.grid(2)

    if i < P.shape[0] and j < P.shape[1]:
        # 检查有效像素
        if (not math.isnan(P[i, j]) and not math.isnan(Pi[i, j]) and
                P[i, j] > 0 and Pi[i, j] > 0):
            # 计算公式: term = 1.735 * 10 ** (1.5 * log10(Pi^2 / P) - 0.8188)
            log_term = 2.0 * math.log10(Pi[i, j]) - math.log10(P[i, j])
            exponent = 1.5 * log_term - 0.8188
            term = 1.735 * math.pow(10.0, exponent)

            R_sum[i, j] += term


def compute_R_for_year_numba(year, annual_path, monthly_dir, shapes, out_dir):
    """使用Numba GPU加速的计算R值函数"""
    try:
        # 首先检查输出是否已存在
        if check_output_exists(year, out_dir):
            return None, None

        # 获取月度文件列表
        monthly_files = []
        for m in range(1, 13):
            ym = f"{year}{m:02d}"
            path = os.path.join(monthly_dir, f"ERA5_Precipitation_Monthly_{ym}.tif")
            if os.path.exists(path):
                monthly_files.append(path)

        if not monthly_files:
            print(f"年份 {year} 没有找到月度文件")
            return None, None

        # 获取栅格信息
        with rasterio.open(annual_path) as src:
            height, width = src.height, src.width
            profile = src.profile
            print(f"处理年份 {year}, 数据尺寸: {width}x{height}")

        # 准备输出文件
        out_meta = profile.copy()
        out_meta.update({
            "dtype": "float32",
            "count": 1,
            "nodata": -9999,
            "compress": "lzw"
        })

        out_tif = os.path.join(out_dir, f"R_{year}.tif")

        # 创建输出文件
        with rasterio.open(out_tif, "w", **out_meta) as dst:
            # 分块处理
            chunks = list(range(0, height, CHUNK_SIZE))
            monthly_stats = []

            for chunk_start in tqdm(chunks, desc=f"处理 {year}"):
                chunk_size = min(CHUNK_SIZE, height - chunk_start)
                window = Window(0, chunk_start, width, chunk_size)

                # 读取年度数据的当前块
                with rasterio.open(annual_path) as annual_src:
                    P_cpu = annual_src.read(1, window=window).astype('float32')

                # 初始化R_sum
                R_sum_cpu = np.zeros_like(P_cpu, dtype='float32')
                valid_mask_P = np.isfinite(P_cpu)

                # 处理每个月的块
                for idx, mpath in enumerate(monthly_files, 1):
                    with rasterio.open(mpath) as monthly_src:
                        Pi_cpu = monthly_src.read(1, window=window).astype('float32')

                    if USE_GPU:
                        try:
                            # 将数据传输到GPU
                            P_gpu = cuda.to_device(P_cpu)
                            Pi_gpu = cuda.to_device(Pi_cpu)
                            R_sum_gpu = cuda.to_device(R_sum_cpu)

                            # 配置GPU网格和块大小
                            threadsperblock = (16, 16)
                            blockspergrid_x = math.ceil(P_cpu.shape[0] / threadsperblock[0])
                            blockspergrid_y = math.ceil(P_cpu.shape[1] / threadsperblock[1])
                            blockspergrid = (blockspergrid_x, blockspergrid_y)

                            # 启动GPU核函数
                            compute_R_kernel[blockspergrid, threadsperblock](P_gpu, Pi_gpu, R_sum_gpu)

                            # 将结果复制回CPU
                            R_sum_cpu = R_sum_gpu.copy_to_host()

                            # 月度统计（只在第一个块收集）
                            if chunk_start == 0:
                                valid = (np.isfinite(Pi_cpu)) & (np.isfinite(P_cpu)) & (Pi_cpu > 0) & (P_cpu > 0)
                                if np.any(valid):
                                    # 重新计算term用于统计
                                    term = np.zeros_like(P_cpu, dtype='float32')
                                    term_valid = valid & (Pi_cpu > 0) & (P_cpu > 0)
                                    if np.any(term_valid):
                                        log_term = 2.0 * np.log10(Pi_cpu[term_valid]) - np.log10(P_cpu[term_valid])
                                        exponent = 1.5 * log_term - 0.8188
                                        term[term_valid] = 1.735 * (10.0 ** exponent)
                                        region_mean = float(term[term_valid].mean())
                                        monthly_stats.append({
                                            "date": f"{year}-{idx:02d}",
                                            "R": region_mean
                                        })
                                    else:
                                        monthly_stats.append({
                                            "date": f"{year}-{idx:02d}",
                                            "R": np.nan
                                        })
                                else:
                                    monthly_stats.append({
                                        "date": f"{year}-{idx:02d}",
                                        "R": np.nan
                                    })

                        except Exception as gpu_error:
                            print(f"GPU处理失败: {gpu_error}，回退到CPU")
                            R_sum_cpu = process_chunk_cpu(P_cpu, Pi_cpu, R_sum_cpu)
                    else:
                        # CPU处理
                        R_sum_cpu = process_chunk_cpu(P_cpu, Pi_cpu, R_sum_cpu)

                        # 月度统计（只在第一个块收集）
                        if chunk_start == 0:
                            valid = (np.isfinite(Pi_cpu)) & (np.isfinite(P_cpu)) & (Pi_cpu > 0) & (P_cpu > 0)
                            if np.any(valid):
                                term = np.zeros_like(P_cpu, dtype='float32')
                                term[valid] = 1.735 * (10.0 ** (
                                            1.5 * (2.0 * np.log10(Pi_cpu[valid]) - np.log10(P_cpu[valid])) - 0.8188))
                                region_mean = float(term[valid].mean())
                                monthly_stats.append({
                                    "date": f"{year}-{idx:02d}",
                                    "R": region_mean
                                })
                            else:
                                monthly_stats.append({
                                    "date": f"{year}-{idx:02d}",
                                    "R": np.nan
                                })

                # 设置无效值
                R_sum_cpu[~valid_mask_P] = -9999

                # 写入输出文件
                dst.write(R_sum_cpu, 1, window=window)

                # 强制垃圾回收
                del P_cpu, R_sum_cpu

        print(f"写出 R 影像：{out_tif}")

        # 创建CSV文件
        csv_path = os.path.join(out_dir, f"R_{year}.csv")
        df = pd.DataFrame(monthly_stats)
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"写出 CSV：{csv_path}")

        return out_tif, csv_path

    except Exception as e:
        print(f"处理年份 {year} 时出错: {e}")
        return None, None


def process_chunk_cpu(P_cpu, Pi_cpu, R_sum_cpu):
    """CPU处理一个数据块"""
    valid = (np.isfinite(Pi_cpu)) & (np.isfinite(P_cpu)) & (Pi_cpu > 0) & (P_cpu > 0)

    if np.any(valid):
        log_term = 2.0 * np.log10(Pi_cpu[valid]) - np.log10(P_cpu[valid])
        exponent = 1.5 * log_term - 0.8188
        term_valid = 1.735 * (10.0 ** exponent)

        term = np.zeros_like(P_cpu, dtype='float32')
        term[valid] = term_valid
        R_sum_cpu += term

    return R_sum_cpu


def main():
    """主函数"""
    annual_pattern = os.path.join(ANNUAL_DIR, "ERA5_Precipitation_Annual_*.tif")
    annual_files = sorted(glob.glob(annual_pattern))

    if not annual_files:
        raise FileNotFoundError(f"未发现年降水文件：{annual_pattern}")

    print(f"找到 {len(annual_files)} 个年度文件")

    # 统计变量
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 顺序处理每个年份
    for annual_path in annual_files:
        fname = os.path.basename(annual_path)
        import re
        m = re.search(r"(\d{4})", fname)
        if m:
            year = m.group(1)
            print(f"\n开始处理年份: {year}")

            # 检查输出是否已存在
            if check_output_exists(year, OUT_DIR):
                skipped_count += 1
                continue

            try:
                result = compute_R_for_year_numba(year, annual_path, MONTHLY_DIR, geoms, OUT_DIR)
                if result[0] is not None and result[1] is not None:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"处理年份 {year} 时发生异常: {e}")
                error_count += 1
                continue
        else:
            print(f"无法从文件名解析年份：{fname}，跳过")

    # 输出统计信息
    print("\n" + "="*50)
    print("处理完成统计:")
    print(f"成功处理: {processed_count} 个年份")
    print(f"跳过处理: {skipped_count} 个年份")
    print(f"处理出错: {error_count} 个年份")
    print(f"总计文件: {len(annual_files)} 个年份")
    print("="*50)


if __name__ == "__main__":
    main()