import ee
import os
import requests
import concurrent.futures
from tqdm import tqdm
from datetime import datetime

# ============================================
# 初始化 Earth Engine
# ============================================
ee.Initialize(project='YOUR_PROJECT_ID')

# 研究区（广东）
guangdong = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD")

# 数据集：ERA5-Land 月总降水量
dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate('1985-01-01', '2024-12-31') \
    .select('total_precipitation_sum')

# 输出路径
output_dir = r"./data\GEE\data_Precipitation"
os.makedirs(output_dir, exist_ok=True)

# 分辨率
scale = 1000

# ============================================
# 单月下载函数
# ============================================
def download_month(year, month):
    """下载指定年月的 ERA5-Land 月总降水 GeoTIFF"""
    try:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month')
        img = dataset.filterDate(start, end).mean()  # 取月均值（等价于总量）
        img_mm = img.multiply(1000).rename('precipitation_mm').clip(guangdong)

        # 文件名
        month_str = f"{month:02d}"
        file_name = f"ERA5_Precipitation_Monthly_{year}{month_str}.tif"
        out_path = os.path.join(output_dir, file_name)

        # 如果已存在则跳过
        if os.path.exists(out_path):
            return f"✅ 已存在，跳过：{file_name}"

        # 生成下载链接
        url = img_mm.getDownloadURL({
            'region': guangdong.geometry(),
            'scale': scale,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        })

        # 下载文件
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()

        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return f"✅ 下载完成：{file_name}"

    except Exception as e:
        return f"❌ {year}-{month:02d} 出错：{e}"

# ============================================
# 构建所有年月任务
# ============================================
tasks = [(year, month) for year in range(1985, 2025) for month in range(1, 13)]

# ============================================
# 多线程批量下载
# ============================================
max_workers = 8  # 并行线程数（建议 6–10，根据带宽调整）
print(f"🚀 开始多线程下载 ERA5-Land 月降水数据（共 {len(tasks)} 个文件）...")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(lambda p: download_month(*p), tasks),
                        total=len(tasks),
                        desc="下载进度"))

# ============================================
# 输出日志
# ============================================
log_path = os.path.join(output_dir, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_path, 'w', encoding='utf-8') as f:
    for res in results:
        f.write(res + '\n')

print(f"\n🎉 全部下载任务完成！日志已保存至：{log_path}")
