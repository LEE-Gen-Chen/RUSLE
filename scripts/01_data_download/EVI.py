import ee
import geemap
import os
import time
from datetime import datetime
from tqdm import tqdm

# 检查并安装缺失的依赖（保持你原有的流程）
try:
    import geedim
except ImportError:
    print("正在安装geedim...")
    os.system("pip install geedim")
    import geedim

# GEE 身份验证与初始化
try:
    ee.Initialize(project='YOUR_PROJECT_ID')
    print("✅ GEE已初始化")
except Exception as e:
    if "No credentials" in str(e) or "not found" in str(e):
        print("🔐 需要进行GEE身份验证...")
        ee.Authenticate()
        ee.Initialize(project='YOUR_PROJECT_ID')
        print("✅ GEE身份验证成功并已初始化")
    else:
        raise e

# ========== 配置 ==========
roi_fc = ee.FeatureCollection('projects/YOUR_PROJECT_ID/assets/GD_City')
roi = roi_fc.geometry().bounds()
local_save_path = r'./data\EVI'   # 保存目录（你可以改名为 NDVI/EVI）
scale = 90
crs = 'EPSG:4547'
os.makedirs(local_save_path, exist_ok=True)

# 仅下载 EVI
download_evi = True

# ========== 工具函数（重试/下载） ==========
def get_info_with_retry(ee_object, max_retries=3, delay=10):
    for i in range(max_retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if i < max_retries - 1:
                print(f"⚠️ getInfo 失败，{delay}秒后重试... (错误: {e})")
                time.sleep(delay)
            else:
                raise e

def download_ee_image_with_retry(image, filename, region, scale=100, max_retries=3):
    for attempt in range(max_retries):
        try:
            geemap.download_ee_image(
                image=image,
                filename=filename,
                region=region,
                scale=scale,
                crs=crs,
            )
            print(f"✅ 成功下载: {os.path.basename(filename)}")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"⚠️ 下载失败，{wait_time}秒后重试... (错误: {e})")
                time.sleep(wait_time)
            else:
                print(f"❌ 下载失败 {os.path.basename(filename)}: {e}")
                return False

# ========== 影像预处理 ==========
def applyScaleFactors(image):
    # 如果 SR_B.* 名称在某些传感器不一致，这里容错处理
    try:
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        image = image.addBands(opticalBands, None, True)
    except Exception:
        pass
    try:
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        image = image.addBands(thermalBands, None, True)
    except Exception:
        pass
    return image

def rmCloudNew(image):
    cloudShadowBitMask = (1 << 4)
    cloudsBitMask = (1 << 3)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

# ========== 只计算 EVI ==========
def addEVI(image):
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('nir'), 'RED': image.select('red'), 'BLUE': image.select('blue')}
    ).rename('EVI')
    return image.addBands(evi)

# 统一使用 addIndices 名称以兼容之前调用位置（实际只添加 EVI）
def addIndices(image):
    return addEVI(image)

# ========== 年度统计函数（针对任意单波段，如 EVI） ==========
def calculateAnnualIndexStats(annual_collection, band_name, prefix=None):
    if prefix is None:
        prefix = band_name
    b_sel = annual_collection.select(band_name)
    stat_max = b_sel.max().rename(f'{prefix}_max')
    stat_min = b_sel.min().rename(f'{prefix}_min')
    stat_sum = b_sel.sum().rename(f'{prefix}_sum')
    stat_mean = b_sel.mean().rename(f'{prefix}_mean')
    stats_image = ee.Image.cat([stat_max, stat_min, stat_sum, stat_mean])
    return stats_image

# ========== 加载 Landsat（分段以避免单次请求过大） ==========
print("📡 正在加载Landsat数据...")
print(f"研究区域特征数量: {get_info_with_retry(roi_fc.size())}")
print(f"研究区域边界: {get_info_with_retry(roi.bounds())}")

def load_landsat_data():
    print("🔄 分批加载Landsat数据...")

    L5 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
          .filterBounds(roi)
          .filter(ee.Filter.calendarRange(1985, 1999, 'year'))
          .map(applyScaleFactors)
          .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL'],
                  ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
          .map(rmCloudNew)
          .map(addIndices))

    L5_2 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
            .filterBounds(roi)
            .filter(ee.Filter.calendarRange(2000, 2011, 'year'))
            .map(applyScaleFactors)
            .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL'],
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

    L8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(roi)
          .filter(ee.Filter.calendarRange(2014, 2025, 'year'))
          .map(applyScaleFactors)
          .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
                  ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
          .map(rmCloudNew)
          .map(addIndices))

    Landsat = ee.ImageCollection(L5.merge(L5_2).merge(L7).merge(L8)).sort("system:time_start")
    return Landsat

Landsat = load_landsat_data()

try:
    landsat_count = get_info_with_retry(Landsat.size())
    print(f"✅ Landsat 数据加载完成，共 {landsat_count} 景影像")
except Exception as e:
    print(f"⚠️ 无法获取数据总量，但将继续处理: {e}")
    landsat_count = "未知"

# ========== 年度循环：仅 EVI 的 年度统计并下载 ==========
years = list(range(1985, 2025))  # 测试区间，改为你需要的年份范围，如 range(1985, 2025)

successful_downloads = []
failed_downloads = []

print("📊 开始下载年度 EVI 统计量...")

for year in tqdm(years, desc="下载年度数据"):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    try:
        year_collection = Landsat.filterDate(start_date, end_date)
        year_count = get_info_with_retry(year_collection.size())
    except Exception as e:
        print(f"⚠️ 无法获取{year}年数据量: {e}")
        failed_downloads.append(str(year))
        continue

    if year_count == 0:
        print(f"⚠️ {year} 年无有效影像，跳过")
        failed_downloads.append(str(year))
        continue

    print(f"📅 {year} 年有 {year_count} 景影像")

    if download_evi:
        try:
            evi_stats = calculateAnnualIndexStats(year_collection, 'EVI', prefix='EVI').clip(roi)
            evi_stats = evi_stats.set({'band_names': ['EVI_max', 'EVI_min', 'EVI_sum', 'EVI_mean'], 'year': year})
            evi_filename = os.path.join(local_save_path, f"EVI_Annual_Stats_{year}.tif")
            if os.path.exists(evi_filename):
                print(f"⏭️ {year} 年 EVI 文件已存在，跳过")
                successful_downloads.append(f"EVI_{year}")
            else:
                ok = download_ee_image_with_retry(evi_stats, evi_filename, roi, scale)
                if ok:
                    successful_downloads.append(f"EVI_{year}")
                else:
                    failed_downloads.append(f"EVI_{year}")
        except Exception as e:
            print(f"❌ 处理 {year} 年 EVI 时出错: {e}")
            failed_downloads.append(f"EVI_{year}")

    time.sleep(10)  # 减缓请求频率

# ========== 汇总 ==========
print("\n" + "=" * 50)
print("📋 下载结果统计:")
print(f"✅ 成功下载: {len(successful_downloads)} 个文件/条目")
print(f"❌ 下载失败: {len(failed_downloads)} 个文件/条目")
if successful_downloads:
    print("成功项（示例）:", successful_downloads[:20])
if failed_downloads:
    print("失败项（示例）:", failed_downloads[:20])
print(f"\n💾 数据保存位置: {local_save_path}")

print("""
📋 说明：
每个年度 EVI 文件包含 4 个波段：
  波段1: EVI_max  - 年度最大值
  波段2: EVI_min  - 年度最小值
  波段3: EVI_sum  - 年度累计值
  波段4: EVI_mean - 年度平均值
""")
