import ee

# 初始化 GEE
ee.Initialize(project='YOUR_PROJECT_ID')

# 研究区：广东省市级边界
study_area = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/YOUR_ASSET_NAME")

# 土壤数据集
soil_datasets = {
    "sand": ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"),
    "silt": ee.Image("ISDASOIL/Africa/v1/silt_content"),
    "clay": ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"),
    "soc": ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"),
    "texture": ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02"),
    "soil_class": ee.Image("OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01"),
}

# 循环创建导出任务
for key, img in soil_datasets.items():
    task = ee.batch.Export.image.toDrive(
        image=img.clip(study_area),
        description=f"GD_{key}_soil",
        folder="GEE_SoilData",   # Google Drive 中的文件夹名（可自定义）
        fileNamePrefix=f"GD_{key}",
        region=study_area.geometry(),
        scale=250,               # 默认分辨率
        crs="YOUR_CRS_EPSG",         # 中国大地2000
        fileFormat="GeoTIFF",
        maxPixels=1e13           # 允许大图导出
    )
    task.start()
    print(f"🚀 已启动任务: {key}")

print("✅ 所有导出任务已提交，请在 GEE Task 管理页面查看进度。")
