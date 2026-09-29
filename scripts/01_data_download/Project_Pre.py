# -*- coding: utf-8 -*-
"""
WGS84 → CGCS2000_3Degree_GK_CM114 (wkid：EPSG:4547)  30 m 最近邻
月度 + 年度降水批量转换
"""
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tqdm import tqdm

# ----------- 1. 路径 -----------
in_m  = r"./data\GEE\data_Precipitation\pre_monthly"
in_y  = r"./data\GEE\data_Precipitation\pre_annual"
out_m = r"./data\Precipitation\Monthly_30m"
out_y = r"./data\Precipitation\Annual_30m"
os.makedirs(out_m, exist_ok=True)
os.makedirs(out_y, exist_ok=True)

# ----------- 2. 投影与分辨率 -----------
src_crs   = 'EPSG:4326'
dst_crs   = 'EPSG:4547'      # CGCS2000 / 3° GK CM 114°
dst_res   = 30               # 30 m

# ----------- 3. 通用转换函数 -----------
def reproject_30m(src_path, dst_path):
    with rasterio.open(src_path) as src:
        # 计算目标变换+形状（指定分辨率）
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs,
            src.width, src.height,
            *src.bounds,
            resolution=(dst_res, dst_res))   # 关键：强制30 m
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw',
            'dtype': 'float32'
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)   # ArcGIS 最近邻

# ----------- 4. 生成文件列表 -----------
tasks = []
for y in range(1985, 2025):
    # 年度
    src = os.path.join(in_y, f"ERA5_Precipitation_Annual_{y}.tif")
    dst = os.path.join(out_y, f"ERA5_Precipitation_Annual_{y}.tif")
    if os.path.isfile(src):
        tasks.append((src, dst))
    # 月度
    for m in range(1, 13):
        mm  = f"{m:02d}"
        src = os.path.join(in_m, f"ERA5_Precipitation_Monthly_{y}{mm}.tif")
        dst = os.path.join(out_m, f"ERA5_Precipitation_Monthly_{y}{mm}.tif")
        if os.path.isfile(src):
            tasks.append((src, dst))

# ----------- 5. 批量运行 -----------
for src, dst in tqdm(tasks, desc="Reproject 30 m", unit="file"):
    if os.path.exists(dst):
        continue
    reproject_30m(src, dst)

print("全部 30 m CGCS2000 投影转换完成！")