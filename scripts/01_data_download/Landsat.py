import ee
import os
import requests
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import rasterio
from rasterio.merge import merge
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ============================================
# 初始化 Earth Engine
# ============================================
ee.Initialize(project='YOUR_PROJECT_ID')

# 广东省边界与市级行政区（假设你的资产包含每个市的 Feature）
guangdong_fc = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD_City")
city_list = guangdong_fc.aggregate_array("City").getInfo()

print("城市字段示例：", city_list[:10])
print("城市数量：", len(city_list))

# ============================================
# 合并 Landsat 数据集（L2，大气校正）
# ============================================
landsat_all = (
    ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
    .merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"))
    .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
    .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
    .filterDate("1985-01-01", "2025-01-01")
    .filterBounds(guangdong_fc)
)

# ============================================
# 云与阴影掩膜函数（使用 QA_PIXEL）
# ============================================
def maskLandsatSR(image):
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud.And(shadow))

# ============================================
# 波段重命名（统一波段名）
# ============================================
def renameBands(image):
    sensor = ee.String(image.get("SPACECRAFT_ID"))
    is_8_9 = sensor.compareTo("LANDSAT_8").eq(0).Or(sensor.compareTo("LANDSAT_9").eq(0))
    return ee.Image(ee.Algorithms.If(
        is_8_9,
        image.select(["SR_B1","SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                     ["UltraBlue","Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]),
        image.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                     ["UltraBlue","Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    ))

landsat_all = landsat_all.map(maskLandsatSR).map(renameBands)

# ============================================
# 输出路径
# ============================================
output_root = r"./data\Landsat"
os.makedirs(output_root, exist_ok=True)
scale = 30

# ============================================
# 下载函数（按市区分块）
# ============================================
def download_best_landsat(year, month, city_name):
    try:
        region = guangdong_fc.filter(ee.Filter.eq("City", city_name)).geometry()  # 修正
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        month_col = landsat_all.filterDate(start, end).filterBounds(region)

        count = month_col.size().getInfo()
        print(f"[DEBUG] {city_name} {year}-{month:02d} -> count={count}")
        if count == 0:
            return f"⚠️ {city_name} {year}-{month:02d} 无影像"

        best_img = month_col.sort("CLOUD_COVER").first().clip(region)
        system_id = best_img.get("LANDSAT_PRODUCT_ID").getInfo()
        file_name = f"{city_name}_{year}{month:02d}_{system_id}.tif"
        out_dir = os.path.join(output_root, f"{year}_{month:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, file_name)

        if os.path.exists(out_path):
            return f"✅ 已存在：{file_name}"

        # 下载链接
        url = best_img.getDownloadURL({
            "region": region,
            "scale": scale,
            "crs": "EPSG:4490",
            "format": "GEO_TIFF"
        })

        # 下载文件
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return f"✅ {city_name} {year}-{month:02d} 下载完成"

    except Exception as e:
        return f"❌ {city_name} {year}-{month:02d} 出错：{e}"

# ============================================
# 拼接函数（同年月市级影像 → 省级）
# ============================================
def mosaic_month(year, month):
    folder = os.path.join(output_root, f"{year}_{month:02d}")
    tif_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
    if not tif_list:
        return f"⚠️ {year}-{month:02d} 无可拼接影像"

    try:
        src_files_to_mosaic = [rasterio.open(fp) for fp in tif_list]
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": "EPSG:4490"
        })
        out_file = os.path.join(output_root, f"GD_Landsat_{year}_{month:02d}.tif")
        with rasterio.open(out_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in src_files_to_mosaic:
            src.close()

        return f"🧩 拼接完成：{year}-{month:02d}"

    except Exception as e:
        return f"❌ 拼接出错 {year}-{month:02d}：{e}"

# ============================================
# 主执行逻辑
# ============================================
tasks = [(year, month, city) for year in range(1987, 2025) for month in range(1, 13) for city in city_list]
print(f"🚀 开始按市下载 Landsat 影像，共 {len(tasks)} 个任务")

# 多线程下载
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(lambda p: download_best_landsat(*p), tasks),
                        total=len(tasks), desc="下载进度"))

# 拼接省级结果
for year in range(1987, 2025):
    for month in range(1, 13):
        msg = mosaic_month(year, month)
        print(msg)

# 日志保存
log_path = os.path.join(output_root, f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_path, "w", encoding="utf-8") as f:
    for res in results:
        f.write(res + "\n")

print(f"\n🎉 下载与拼接完成！日志保存至：{log_path}")
