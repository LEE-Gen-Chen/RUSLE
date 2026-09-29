import ee
import pandas as pd
import os

# 1. 初始化
ee.Authenticate()
ee.Initialize(project='YOUR_PROJECT_ID')

# 2. 区域
region = ee.FeatureCollection("projects/YOUR_PROJECT_ID/assets/GD_Area").geometry()

# 3. 碳池影像模板
above_template = "NASA/ORNL/biomass_carbon_density/v1/{year}"
soil_template = "projects/soilgrids-isric/soilgrids/ocd/mean_0-30cm_depth"

# 4. 空列表存储结果
records = []

for year in range(2000, 2021):
    # 地上
    above = ee.Image(above_template.format(year=year))
    # 地下、枯死按系数估算
    below = above.multiply(0.24)
    dead = above.multiply(0.2)
    # 土壤（静态）
    soil = ee.Image(soil_template)
    soil = soil.multiply(1.3).multiply(0.3).multiply(0.1)  # 转 Mg/ha


    # 计算区域平均
    def mean_val(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e13
        )
        return stats.getInfo()['b1']


    rec = {
        'year': year,
        'above_MgC_ha': mean_val(above),
        'below_MgC_ha': mean_val(below),
        'dead_MgC_ha': mean_val(dead),
        'soil_MgC_ha': mean_val(soil)
    }
    print(f"Year {year}: {rec}")
    records.append(rec)

# 5. 保存为 CSV
df = pd.DataFrame(records)
out_csv = r'./data\GEE\data\carbon_pools_2000_2020.csv'
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"Saved CSV to {out_csv}")
