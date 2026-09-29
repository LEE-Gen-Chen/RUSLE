# -*- coding: utf-8 -*-
"""
无需 ArcPy，利用 rasterio + numpy 批量月→年降水累加
python 3.x 可直接运行
"""
import os
import numpy as np
import rasterio
from glob import glob
from tqdm import tqdm   # pip install tqdm

# 1. 路径设置
in_dir  = r"./data\GEE\data_Precipitation\pre_monthly"
out_dir = r"./data\GEE\data_Precipitation\pre_annual"
os.makedirs(out_dir, exist_ok=True)

# 2. 生成 1985–2024 年份列表
years = range(1985, 2025)

# 3. 按年处理
for year in tqdm(years, desc="Annual Sum"):
    # 拼出当年12个文件
    pattern = os.path.join(in_dir, f"ERA5_Precipitation_Monthly_{year}*.tif")
    tifs = sorted(glob(pattern))
    if len(tifs) == 0:
        print(f"{year} 无匹配文件，跳过")
        continue

    # 读取第一个月，建立模板
    with rasterio.open(tifs[0]) as src:
        meta = src.meta.copy()
        nodata = src.nodata
        annual = src.read(1).astype('float32')
        if nodata is not None:
            annual[annual == nodata] = np.nan

    # 累加其余月份
    for tif in tifs[1:]:
        with rasterio.open(tif) as src:
            data = src.read(1).astype('float32')
            if nodata is not None:
                data[data == nodata] = np.nan
            annual += data

    # 写年度结果
    out_path = os.path.join(out_dir, f"ERA5_Precipitation_Annual_{year}.tif")
    meta.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(annual, 1)

print("全部年份处理完成！")