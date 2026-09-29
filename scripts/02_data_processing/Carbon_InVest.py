import ee, os, requests, time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 初始化 Earth Engine
ee.Authenticate()
ee.Initialize(project='YOUR_PROJECT_ID')

# 输出路径
out_dir = r"./data\GEE\Carbon_InVset"
os.makedirs(out_dir, exist_ok=True)

# 研究区
region = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD_Area")
region_geojson = region.geometry().getInfo()

# 重试策略
retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

# 土壤碳：静态
ocd = ee.Image("projects/soilgrids-isric/soilgrids/ocd/mean_0-30cm_depth")  # g/kg
soil_carbon = ocd.multiply(1.3).multiply(0.3).multiply(0.1)  # SOC (Mg/ha)

# 目标年份（ESA 支持从2010起）
years = [2010, 2015, 2020]

# 下载函数
def download_img(image: ee.Image, path: str, name: str):
    try:
        print(f"Downloading {name}...")
        url = image.getDownloadURL({
            'scale': 500,
            'region': region_geojson,
            'crs': 'EPSG:4326',
            'fileFormat': 'GeoTIFF'
        })
        with session.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        time.sleep(5)

# 遍历年份下载
for year in years:
    print(f"\n===== Processing {year} =====")

    # ESA Biomass 数据集，每年一个影像
    biomass = ee.Image(f"ESA/BIOMASS_CCI/Biomass/Maps/v1_0/{year}_biomass")

    # 估算不同碳库（Mg/ha）
    above = biomass                            # Aboveground
    below = above.multiply(0.24)              # Belowground ≈ 24% of AGB
    dead  = above.multiply(0.2)               # Dead ≈ 20% of AGB
    soil  = soil_carbon                       # Soil 静态估计

    # 文件路径
    download_img(above, os.path.join(out_dir, f"Carbon_Aboveground_{year}.tif"), f"Aboveground_{year}")
    download_img(below, os.path.join(out_dir, f"Carbon_Belowground_{year}.tif"), f"Belowground_{year}")
    download_img(dead,  os.path.join(out_dir, f"Carbon_Dead_{year}.tif"),        f"Dead_{year}")
    download_img(soil,  os.path.join(out_dir, f"Carbon_Soil_{year}.tif"),        f"Soil_{year}")
