import ee
import requests
import os

# 初始化 Earth Engine
ee.Initialize(project='YOUR_PROJECT_ID')

# 研究区（广东省边界）
study_area = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD_City")

# 加载 SRTM DEM 数据
dem = ee.Image("USGS/SRTMGL1_003").clip(study_area)

# 设置导出参数
task_config = {
    'image': dem,
    'description': 'Guangdong_DEM_30m',
    'scale': 30,                # 分辨率 30m
    'region': study_area.geometry(),
    'fileFormat': 'GeoTIFF',
    'crs': 'EPSG:4490',         # 可改为 EPSG:4490 (CGCS2000)
    'folder': 'GEE_DEM',         # 导出到 Google Drive 的文件夹
    'maxPixels': 2e9
}

# 启动导出任务
task = ee.batch.Export.image.toDrive(**task_config)
task.start()

print("✅ 任务已提交，请前往 GEE Tasks 页面查看导出进度。")
