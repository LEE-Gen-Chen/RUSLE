# RUSLE-Guangdong

基于 **RUSLE（修正通用土壤流失方程）** 的广东省土壤侵蚀遥感估算与建模。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Ready-orange)

## 项目概述

本项目实现了基于 RUSLE 模型的土壤侵蚀估算全流程，包含：

- 🌦️ **R 因子** — 降雨侵蚀力（ERA5-Land 降水数据）
- 🌱 **C 因子** — 植被覆盖与管理因子（MODIS EVI/NDVI 时序数据）
- 🪨 **K 因子** — 土壤可蚀性（HWSD 土壤数据集）
- ⛰️ **LS 因子** — 坡长坡度因子（SRTM DEM）
- 🚜 **P 因子** — 水土保持措施因子
- 📊 **有机碳流失** — RUSLE–SOC 耦合模型
- 🤖 **机器学习预测** — LightGBM 地表有机碳空间预测

## 技术流程

```
遥感数据获取（GEE）
    ├── ERA5-Land 降水  → R 因子
    ├── MODIS EVI/NDVI  → C 因子
    ├── HWSD 土壤属性    → K 因子
    └── SRTM DEM        → LS 因子
         ↓
  RUSLE 主方程计算 → 土壤侵蚀模数
         ↓
有机碳密度建模 → 碳流失估算
         ↓
  LightGBM 空间降尺度预测
```

## 目录结构

```
RUSLE-Guangdong/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── GEE_数据下载与处理流程.ipynb
├── scripts/
│   ├── 01_data_download/      # GEE遥感数据下载
│   │   ├── main.py           # NDVI/EVI年度数据下载
│   │   ├── pre.py            # ERA5月降水数据下载
│   │   ├── pre_daily.py      # ERA5日降水数据下载
│   │   ├── Annual_pre.py     # 年尺度降水汇总
│   │   ├── Project_Pre.py    # 指定区域降水统计
│   │   ├── EVI.py            # EVI数据下载与处理
│   │   ├── NDVI.py           # NDVI数据下载与处理
│   │   ├── NDVI_EVI.py       # NDVI/EVI联合处理
│   │   ├── DEM.py            # DEM下载（SRTM）
│   │   ├── Soil.py           # 土壤数据下载（HWSD）
│   │   ├── SoilGrids.py      # SoilGrids数据下载
│   │   └── Landsat.py        # Landsat数据下载
│   ├── 02_data_processing/  # 本地栅格处理
│   │   ├── RUSLE.py          # RUSLE主方程
│   │   ├── RUSLE_R.py        # R因子计算（Arnoldus公式）
│   │   ├── RUSLE_K.py        # K因子计算（EPIC模型）
│   │   ├── RUSLE_LS.py       # LS因子计算
│   │   ├── RUSLE_C.py        # C因子计算（EVI-NDVI方法）
│   │   ├── RUSLE_P.py        # P因子估算
│   │   ├── HWSD_K.py         # HWSD土壤可蚀性处理
│   │   ├── resample_K.py     # K因子重采样
│   │   ├── Resample_C.py     # C因子重采样
│   │   ├── EVI_STATS.py      # EVI统计
│   │   ├── Carbon_InVest.py  # InVEST碳储量估算
│   │   ├── carbon_dataset.py # 碳密度数据集处理
│   │   ├── tiff_to_npz.py    # 栅格转npz格式
│   │   └── 碳流失方程.py     # 有机碳流失方程
│   └── 03_modeling/          # 建模与预测
│       ├── LightGBM+GTNNWR.py   # 耦合模型预测
│       ├── Markov_CLCD.py       # Markov土地利用转移
│       ├── Plot_Transition_CLCD.py  # 转移矩阵可视化
│       ├── RUSLE_Statistics.py  # 侵蚀统计
│       ├── 重心迁移.py          # 重心迁移分析
│       └── 知识图谱.py         # 知识图谱构建
```

## 环境配置

```bash
pip install -r requirements.txt
```

主要依赖：
- `earthengine-api` — GEE 接口
- `geemap` / `geedim` — GEE 可视化与处理
- `rasterio` / `geopandas` — 栅格/矢量数据处理
- `lightgbm` / `torch` — 机器学习建模
- `numba` — GPU 加速
- `matplotlib` / `seaborn` — 可视化

## 使用说明

### 1. GEE 数据下载

```python
# 配置你的 GEE 项目
ee.Initialize(project='YOUR_PROJECT_ID')

# 研究区资产路径（替换占位符）
roi = ee.FeatureCollection('projects/YOUR_PROJECT_ID/assets/YOUR_ASSET_NAME')
```

### 2. RUSLE 计算

```python
from RUSLE import RUSLECalculator

calc = RUSLECalculator(target_resolution=30, target_crs="EPSG:4547")
calc.load_boundary('./data/boundary.shp')
```

### 3. 有机碳预测

```python
from LightGBM_GTNNWR import train_and_predict

model = train_and_predict(X_train, Y_train, X_predict)
```

## 数据来源

| 因子 | 数据集 | 来源 |
|------|--------|------|
| R | ERA5-Land 月/日降水 | ECMWF |
| C | MODIS MOD13A2 EVI/NDVI | NASA |
| K | HWSD 土壤属性 | ISRIC |
| LS | SRTM DEM | NASA/USGS |
| P | 估算值（经验公式） | — |

> **注意**：研究区边界（shapefile）需自行上传至 Google Earth Engine Assets

## 参考引用

如果本项目对你的研究有帮助，请引用：

```
李创杰. 基于RUSLE模型的广东省土壤侵蚀与地表有机碳流失时空耦合研究.
惠州学院本科毕业论文, 2026.
```

## 许可

MIT License
