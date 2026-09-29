import ee
import geemap
import os
import time
from datetime import datetime
from tqdm import tqdm

# 检查并安装缺失的依赖
try:
    import geedim
except ImportError:
    print("正在安装geedim...")
    os.system("pip install geedim")
    import geedim

# GEE身份验证和初始化
try:
    ee.Initialize(project='YOUR_PROJECT_ID')
    print("✅ GEE已初始化")
except Exception as e:
    if "No credentials" in str(e) or "not found" in str(e):
        print("🔐 需要进行GEE身份验证...")
        print("请按照以下步骤操作：")
        print("1. 在弹出的浏览器窗口中登录您的Google账户")
        print("2. 授予GEE必要的权限")
        print("3. 复制验证码并粘贴到命令行中")

        ee.Authenticate()
        ee.Initialize(project='YOUR_PROJECT_ID')
        print("✅ GEE身份验证成功并已初始化")
    else:
        raise e

# 配置参数
roi_fc = ee.FeatureCollection('projects/YOUR_PROJECT_ID/assets/YOUR_ASSET_NAME')
# 将FeatureCollection转换为Geometry（使用边界框）
roi = roi_fc.geometry().bounds()  # 使用边界框而不是完整几何，减少复杂度

local_save_path = r'./data\NDVI'
scale = 100  # 增大分辨率以减少数据量
crs = 'YOUR_CRS_EPSG'  # 坐标系

# 确保本地目录存在
os.makedirs(local_save_path, exist_ok=True)


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
    return image.addBands(ndvi)  # 只计算NDVI，减少计算量


def calculateAnnualNDVIStats(annual_collection):
    """计算NDVI的年统计量"""
    # 年最大值
    ndvi_max = annual_collection.select('NDVI').max().rename('NDVI_max')

    # 年最小值
    ndvi_min = annual_collection.select('NDVI').min().rename('NDVI_min')

    # 年总值（累计值）
    ndvi_sum = annual_collection.select('NDVI').sum().rename('NDVI_sum')

    # 年平均值
    ndvi_mean = annual_collection.select('NDVI').mean().rename('NDVI_mean')

    # 将四个统计量合并为一个4波段图像
    stats_image = ee.Image.cat([ndvi_max, ndvi_min, ndvi_sum, ndvi_mean])

    return stats_image


def get_info_with_retry(ee_object, max_retries=3, delay=10):
    """带重试的getInfo函数"""
    for i in range(max_retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if i < max_retries - 1:
                print(f"⚠️ 获取信息失败，{delay}秒后重试... (错误: {e})")
                time.sleep(delay)
            else:
                raise e


def download_ndvi_stats(image, filename, region, scale=100, max_retries=3):
    """使用geemap下载NDVI统计数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            # 使用geemap的download_ee_image函数直接下载到本地
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
                wait_time = 30 * (attempt + 1)  # 指数退避
                print(f"⚠️ 下载失败，{wait_time}秒后重试... (错误: {e})")
                time.sleep(wait_time)
            else:
                print(f"❌ 下载失败 {os.path.basename(filename)}: {e}")
                return False


# ============================================
# 加载 Landsat 数据（分批加载，避免超时）
# ============================================
print("📡 正在加载Landsat数据...")

# 获取研究区域信息用于调试
print(f"研究区域特征数量: {get_info_with_retry(roi_fc.size())}")
print(f"研究区域边界: {get_info_with_retry(roi.bounds())}")


# 分批加载数据，避免单次请求过大
def load_landsat_data():
    print("🔄 分批加载Landsat数据...")

    # 分时间段加载
    L5 = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
          .filterBounds(roi)
          .filter(ee.Filter.calendarRange(1985, 1999, 'year'))  # 分时间段
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
          .filter(ee.Filter.calendarRange(2014, 2024, 'year'))
          .map(applyScaleFactors)
          .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
                  ['blue', 'green', 'red', 'nir', 'QA_PIXEL'])
          .map(rmCloudNew)
          .map(addIndices))

    # 合并所有Landsat数据
    Landsat = ee.ImageCollection(L5.merge(L5_2).merge(L7).merge(L8)).sort("system:time_start")
    return Landsat


# 加载数据
Landsat = load_landsat_data()

# 获取数据量（带重试）
try:
    landsat_count = get_info_with_retry(Landsat.size())
    print(f"✅ Landsat 数据加载完成，共 {landsat_count} 景影像")

    if landsat_count > 5000:
        print("⚠️ 数据量较大，建议：")
        print("   - 使用更大的scale值（如500米）")
        print("   - 分省份或地区下载")
        print("   - 减少时间范围")
except Exception as e:
    print(f"⚠️ 无法获取数据总量，但将继续处理: {e}")
    landsat_count = "未知"

# ============================================
# 下载年度NDVI统计数据
# ============================================
# 先测试最近几年，确保流程正常
# years = list(range(1985, 2025))  # 原始范围
years = list(range(1985, 2000))  # 测试范围，先下载最近5年

successful_downloads = []
failed_downloads = []

print("📊 开始下载年度NDVI统计量...")

for year in tqdm(years, desc="下载年度数据"):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # 筛选年度数据（带重试）
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

    # 计算年度NDVI统计量（4个波段）
    annual_stats = calculateAnnualNDVIStats(year_collection).clip(roi)

    # 设置波段描述
    annual_stats = annual_stats.set({
        'band_names': ['NDVI_max', 'NDVI_min', 'NDVI_sum', 'NDVI_mean'],
        'year': year
    })

    # 构建文件名和路径
    filename = os.path.join(local_save_path, f"NDVI_Annual_Stats_{year}.tif")

    # 如果文件已存在，跳过下载
    if os.path.exists(filename):
        print(f"⏭️ {year} 年数据已存在，跳过")
        successful_downloads.append(str(year))
        continue

    # 下载数据（带重试）
    success = download_ndvi_stats(annual_stats, filename, roi, scale)

    if success:
        successful_downloads.append(str(year))
    else:
        failed_downloads.append(str(year))

    # 添加延迟避免请求过于频繁
    time.sleep(10)

# ============================================
# 下载结果统计
# ============================================
print("\n" + "=" * 50)
print("📋 下载结果统计:")
print(f"✅ 成功下载: {len(successful_downloads)} 个文件")
print(f"❌ 下载失败: {len(failed_downloads)} 个文件")

if successful_downloads:
    # 分离年份进行排序
    years_success = [y for y in successful_downloads if y.isdigit()]
    others_success = [y for y in successful_downloads if not y.isdigit()]
    if years_success:
        print("成功下载的年份:", sorted(years_success, key=int))
    if others_success:
        print("其他成功文件:", others_success)

if failed_downloads:
    years_failed = [y for y in failed_downloads if y.isdigit()]
    others_failed = [y for y in failed_downloads if not y.isdigit()]
    if years_failed:
        print("下载失败的年份:", sorted(years_failed, key=int))
    if others_failed:
        print("其他失败文件:", others_failed)

print(f"\n💾 数据保存位置: {local_save_path}")
print("""
📋 数据说明：
每个年度文件包含4个波段：
  波段1: NDVI_max  - 年度最大值
  波段2: NDVI_min  - 年度最小值  
  波段3: NDVI_sum  - 年度累计值
  波段4: NDVI_mean - 年度平均值

后续步骤：
1. 确认测试年份下载成功后，可修改years列表下载完整数据
2. 如需多年总体统计，可取消注释相关代码
""")

# ============================================
# 可选：下载多年总体统计量（注释掉以节省时间）
# ============================================
"""
print("📈 下载多年总体统计量...")

# 多年NDVI最大值
overall_max = Landsat.select('NDVI').max().rename('NDVI_max_overall')

# 多年NDVI最小值  
overall_min = Landsat.select('NDVI').min().rename('NDVI_min_overall')

# 多年NDVI平均值
overall_mean = Landsat.select('NDVI').mean().rename('NDVI_mean_overall')

# 合并为多波段图像
overall_stats = ee.Image.cat([overall_max, overall_min, overall_mean]).clip(roi)
overall_stats = overall_stats.set({
    'band_names': ['NDVI_max_overall', 'NDVI_min_overall', 'NDVI_mean_overall'],
    'period': '1985-2024'
})

# 下载总体统计数据
overall_filename = os.path.join(local_save_path, "NDVI_Overall_Stats_1985_2024.tif")

if not os.path.exists(overall_filename):
    overall_success = download_ndvi_stats(overall_stats, overall_filename, roi, scale)
    if overall_success:
        successful_downloads.append("Overall_Stats")
    else:
        failed_downloads.append("Overall_Stats")
else:
    print("⏭️ 多年总体统计数据已存在，跳过")
    successful_downloads.append("Overall_Stats")
"""