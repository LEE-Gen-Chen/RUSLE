import ee
import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 初始化 Earth Engine
ee.Authenticate()
ee.Initialize(project='YOUR_PROJECT_ID')

# 输出目录：请确保该路径存在或由代码自动创建
out_dir = r"./data\GEE\data"
os.makedirs(out_dir, exist_ok=True)

# 定义区域和输出范围
region = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD_Area")
region_geom = region.geometry()
region_geojson = region_geom.getInfo()

# 下载重试策略
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)


# 掩膜函数（保留高质量像元）
def maskByQA(image):
    qa = image.select('SummaryQA')
    mask = qa.eq(0)
    return image.updateMask(mask)


# 年度 NDVI/EVI 平均值导出
for year in range(2000, 2021):
    print(f"Processing year {year}...")
    start = f'{year}-01-01'
    end = f'{year}-12-31'

    # 获取年度合成
    col = (ee.ImageCollection("MODIS/061/MOD13A2")
           .filterDate(start, end)
           .filterBounds(region_geom)
           .map(maskByQA)
           .select(['NDVI', 'EVI']))

    composite = col.mean().multiply(0.0001)

    for band in ['NDVI', 'EVI']:
        img = composite.select(band)
        out_path = os.path.join(out_dir, f"{band}_{year}.tif")
        print(f"  Downloading {band}_{year}.tif ...")

        # 获取 GEE 下载链接
        try:
            url = img.getDownloadURL({
                'scale': 500,
                'crs': 'EPSG:4326',
                'region': region_geojson,
                'fileFormat': 'GeoTIFF'
            })

            # 下载图像
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  Error downloading {band}_{year}: {e}")
            time.sleep(5)
