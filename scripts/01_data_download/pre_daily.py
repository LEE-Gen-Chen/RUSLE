import ee
import os
import requests
import concurrent.futures
from tqdm import tqdm
from datetime import datetime, timedelta
import time

# ============================================
# 初始化 Earth Engine
# ============================================
ee.Initialize(project='YOUR_PROJECT_ID')

# 研究区（广东）
guangdong = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD")

# 数据集：ERA5-Land 日总降水量（单位 m）
dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
    .filterDate('1985-01-01', '2024-12-31') \
    .select('total_precipitation_sum')

# 输出路径
output_dir = r"./data\GEE\data_Precipitation\pre_daily"
os.makedirs(output_dir, exist_ok=True)

# 分辨率
scale = 1000

# ============================================
# 单日下载函数（带自动重试）
# ============================================
def download_day(date_str, retries=3):
    """下载指定日期的 ERA5-Land 日总降水 GeoTIFF"""
    for attempt in range(1, retries + 1):
        try:
            date = ee.Date(date_str)
            img = dataset.filterDate(date, date.advance(1, 'day')).first()
            if img is None:
                return f"⚠️ 无数据：{date_str}"

            # 转换单位（m → mm）
            img_mm = img.multiply(1000).rename('precipitation_mm').clip(guangdong)

            # 文件名与路径
            file_name = f"ERA5_Precipitation_Daily_{date_str}.tif"
            out_path = os.path.join(output_dir, file_name)

            # 若已存在则跳过
            if os.path.exists(out_path):
                return f"✅ 已存在，跳过：{file_name}"

            # 生成下载链接
            url = img_mm.getDownloadURL({
                'region': guangdong.geometry(),
                'scale': scale,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            })

            # 下载
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()

            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return f"✅ 下载完成：{file_name}"

        except Exception as e:
            if attempt < retries:
                time.sleep(5)  # 等待 5 秒再重试
                continue
            return f"❌ {date_str} 失败：{e}"

# ============================================
# 构建所有日期任务
# ============================================
start_date = datetime(1985, 1, 1)
end_date = datetime(2024, 12, 31)
tasks = []

current = start_date
while current <= end_date:
    tasks.append(current.strftime("%Y-%m-%d"))
    current += timedelta(days=1)

print(f"📅 共需下载 {len(tasks)} 天数据（1985–2024）")

# ============================================
# 多线程下载
# ============================================
max_workers = 8  # 并行线程数
print(f"🚀 开始多线程下载 ERA5-Land 日降水数据（共 {len(tasks)} 个文件）...")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(download_day, tasks),
                        total=len(tasks),
                        desc="下载进度"))

# ============================================
# 输出日志
# ============================================
log_path = os.path.join(output_dir, f"download_log_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_path, 'w', encoding='utf-8') as f:
    for res in results:
        f.write(res + '\n')

print(f"\n🎉 全部下载任务完成！日志已保存至：{log_path}")
