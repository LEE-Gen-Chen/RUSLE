import os
import re
import numpy as np
import rasterio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================
# 参数设置
# =============================
input_folder = r"./data\NDVI"
output_folder_c = r"./data\NDVI\C_value"
output_folder_f = r"./data\NDVI\Fcov"

os.makedirs(output_folder_c, exist_ok=True)
os.makedirs(output_folder_f, exist_ok=True)

ndvi_files = sorted([
    f for f in os.listdir(input_folder)
    if f.startswith("NDVI_Annual_Stats_") and f.endswith(".tif")
])
print(f"检测到 {len(ndvi_files)} 个 NDVI 文件。")

# =============================
# 定义计算函数
# =============================
def process_ndvi(ndvi_file):
    try:
        year_match = re.search(r'\d{4}', ndvi_file)
        year = year_match.group(0) if year_match else "unknown"
        ndvi_path = os.path.join(input_folder, ndvi_file)

        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(4, masked=True).astype(float).filled(np.nan)
            meta = src.meta.copy()

        ndvi_valid = ndvi[~np.isnan(ndvi)]
        NDVI_max = np.percentile(ndvi_valid, 98)
        NDVI_min = np.percentile(ndvi_valid, 2)

        NDVI_range = NDVI_max - NDVI_min if (NDVI_max - NDVI_min) != 0 else 1e-6
        Fcov = ((ndvi - NDVI_min) / NDVI_range) ** 2
        Fcov = np.clip(Fcov, 0, 1)

        C = np.ones_like(Fcov, dtype=float)
        mask_valid = Fcov > 0
        C[mask_valid] = 0.6508 - 0.3436 * np.log10(Fcov[mask_valid] * 100)
        C[Fcov > 0.783] = 0
        C = np.clip(C, 0, 1)

        meta.update(dtype=rasterio.float32, count=1)

        fcov_path = os.path.join(output_folder_f, f"Fcov_{year}.tif")
        with rasterio.open(fcov_path, 'w', **meta) as dst:
            dst.write(Fcov.astype(np.float32), 1)

        c_path = os.path.join(output_folder_c, f"C_NDVI_{year}.tif")
        with rasterio.open(c_path, 'w', **meta) as dst:
            dst.write(C.astype(np.float32), 1)

        return f"{year} 年完成"
    except Exception as e:
        return f"{ndvi_file} 出错: {e}"

# =============================
# 多线程运行
# =============================
MAX_WORKERS = min(8, os.cpu_count())  # 可调整线程数，例如8线程

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_ndvi, f) for f in ndvi_files]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="计算 Fcov 和 C 中"):
        print(fut.result())

print("\n✅ 所有年份的 Fcov 和 C 值计算完成！")
