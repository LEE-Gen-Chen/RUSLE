import ee
import time
from datetime import datetime
from tqdm import tqdm

# ============================================
# 初始化 Earth Engine
# ============================================
ee.Initialize(project='YOUR_PROJECT_ID')

# ============================================
# 参数设置
# ============================================
roi = ee.FeatureCollection('projects/YOUR_PROJECT_ID/assets/YOUR_ASSET_NAME')
roi_geometry = roi.geometry()
drive_folder = 'GEE_NDVI_EVI'
scale = 30
crs = 'YOUR_CRS_EPSG'
MAX_CONCURRENT_TASKS = 50  # 最大并发任务数，可根据需要调整（建议 < 100）


# ============================================
# 函数定义
# ============================================
def applyScaleFactors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)


def rmCloudNew(image):
    cloudShadowBitMask = (1 << 4)
    cloudsBitMask = (1 << 3)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def addIndices(image):
    ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('nir'), 'RED': image.select('red'), 'BLUE': image.select('blue')}
    ).rename('EVI')
    return image.addBands([ndvi, evi])


def export_to_drive(image, name):
    """提交导出任务到 Google Drive"""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=drive_folder,
        region=roi_geometry,
        scale=scale,
        crs=crs,
        maxPixels=1e13
    )
    task.start()
    print(f"🚀 已提交任务: {name}")
    return task


def get_active_task_count():
    """统计当前正在运行的任务数量"""
    task_list = ee.data.getTaskList()
    active = [t for t in task_list if t['state'] in ['RUNNING', 'READY']]
    return len(active)


def monitor_tasks(interval=120):
    """定期输出任务状态"""
    task_list = ee.data.getTaskList()
    total = len(task_list)
    while True:
        task_list = ee.data.getTaskList()
        states = [t['state'] for t in task_list]
        completed = states.count('COMPLETED')
        failed = states.count('FAILED')
        running = states.count('RUNNING')
        ready = states.count('READY')
        progress = (completed + failed) / total * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"✅ 完成 {completed}/{total} | ⚙️ 运行 {running} | ⏳ 等待 {ready} | ❌ 失败 {failed} | 进度 {progress:.1f}%")
        if completed + failed == total:
            print("🎉 所有任务已完成！")
            break
        time.sleep(interval)


# ============================================
# 加载 Landsat 数据
# ============================================
L8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterBounds(roi)
      .filter(ee.Filter.calendarRange(2014, 2024, 'year'))
      .map(applyScaleFactors)
      .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
              ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
      .map(rmCloudNew)
      .map(addIndices))

L7 = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
      .filterBounds(roi)
      .filter(ee.Filter.calendarRange(2012, 2013, 'year'))
      .map(applyScaleFactors)
      .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL'],
              ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
      .map(rmCloudNew)
      .map(addIndices))

L5 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
      .filterBounds(roi)
      .filter(ee.Filter.calendarRange(1985, 2011, 'year'))
      .map(applyScaleFactors)
      .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL'],
              ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
      .map(rmCloudNew)
      .map(addIndices))

Landsat = ee.ImageCollection(L8.merge(L7).merge(L5)).sort("system:time_start")
print(f"✅ Landsat 数据加载完成，共 {Landsat.size().getInfo()} 景影像")

# ============================================
# 提交年度导出任务（异步并发）
# ============================================
years = list(range(1985, 2025))
all_tasks = []

for year in tqdm(years, desc="提交年度任务"):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    year_data = Landsat.filterDate(start_date, end_date)

    if year_data.size().getInfo() == 0:
        print(f"⚠️ {year} 无影像，跳过")
        continue

    ndvi_mean = year_data.select('NDVI').mean().clip(roi)
    evi_mean = year_data.select('EVI').mean().clip(roi)

    # 等待任务数量低于上限再提交
    while get_active_task_count() >= MAX_CONCURRENT_TASKS:
        print(f"⏳ 达到最大任务数({MAX_CONCURRENT_TASKS})，等待中...")
        time.sleep(60)

    all_tasks.append(export_to_drive(ndvi_mean, f"NDVI_mean_{year}"))
    all_tasks.append(export_to_drive(evi_mean, f"EVI_mean_{year}"))

    time.sleep(2)

# 多年均值导出
overall_mean = Landsat.select(['NDVI', 'EVI']).mean().clip(roi)
while get_active_task_count() >= MAX_CONCURRENT_TASKS:
    print(f"⏳ 等待任务数低于 {MAX_CONCURRENT_TASKS} ...")
    time.sleep(60)
all_tasks.append(export_to_drive(overall_mean.select('NDVI'), "overall_NDVI_mean"))
all_tasks.append(export_to_drive(overall_mean.select('EVI'), "overall_EVI_mean"))

print("🎯 所有导出任务已提交，开始监控任务进度...")
monitor_tasks(interval=180)
