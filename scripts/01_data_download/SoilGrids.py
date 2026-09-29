import ee
import os

# 1. 初始化 Earth Engine（需提前通过 earthengine authenticate 完成认证）
ee.Initialize()

# 2. 加载研究区 FeatureCollection，并提取其几何范围
region_fc = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/YOUR_ASSET_NAME")
region = region_fc.geometry()

# 3. 导出参数配置
EXPORT_FOLDER = 'GEE_SoilGrids'  # Google Drive 中的文件夹名
EXPORT_SCALE = 250  # 导出分辨率，单位：米
CRS = 'EPSG:4326'  # 坐标参考系

# 4. 要下载的土壤属性列表
soil_bands = {
    'sand_0-5cm': 'sand_content_0-5cm',
    'silt_0-5cm': 'silt_content_0-5cm',
    'clay_0-5cm': 'clay_content_0-5cm',
    'bdod_0-5cm': 'bulk_density_0-5cm',
    'soc_0-5cm': 'organic_carbon_0-5cm'
}

# 5. 加载 SoilGrids v2 数据集
soil = ee.Image('ISRIC/SoilGrids250m/v2_0')

# 6. 发起导出任务
for name, band in soil_bands.items():
    # 6.1 选波段并裁剪到研究区范围
    img = soil.select(band).clip(region)

    # 6.2 构造并启动导出任务
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f'SoilGrids_{name}',
        folder=EXPORT_FOLDER,
        fileNamePrefix=f'SoilGrids_{name}',
        region=region,  # 直接使用几何范围
        scale=EXPORT_SCALE,
        crs=CRS,
        maxPixels=1e13
    )
    task.start()
    print(f"Started export task for {name}: {task.id}")

print("所有导出任务已启动，请到 Earth Engine 任务面板监控并完成，导出完成后可通过 Google Drive 同步到本地。")
