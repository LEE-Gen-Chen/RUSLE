import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================== 基础路径与参数 ==================
base_path = r"./data\Data"
out_root = Path(base_path) / "GTNNWR"
out_root.mkdir(parents=True, exist_ok=True)

shp_path = Path(base_path) / "GD_SHP" / "GD.shp"

years = list(range(1990, 2025))
agg_resolution = 300  # 30m × 10（严格整除）

folders = {
    'C': 'C_NDVI_{year}.tif',
    'K': 'K.tif',
    'LS': 'LS.tif',
    'P': 'P{year}.tif',
    'R': 'R_{year}.tif',
    'SOC': 'SOC{year}.tif',
    'erosion': 'RUSLE{year}.tif'
}

subfolders = {
    'C': Path(base_path) / 'C',
    'K': Path(base_path) / 'K',
    'LS': Path(base_path) / 'LS',
    'P': Path(base_path) / 'P',
    'R': Path(base_path) / 'R',
    'SOC': Path(base_path) / '土壤有机碳流失量',
    'erosion': Path(base_path) / '土壤侵蚀模数'
}

# ================== 广东省边界 ==================
gdf = gpd.read_file(shp_path)
gd_geometry = gdf.geometry.values

# ================== 栅格裁剪 + 聚合函数 ==================
def process_raster(file_path, target_res, ref_transform=None, ref_shape=None, ref_crs=None):
    with rasterio.open(file_path) as src:
        out_image, out_transform = mask(
            src,
            gd_geometry,
            crop=True,
            filled=True,
            nodata=src.nodata
        )
        data = out_image[0]

        # 对齐到参考网格
        if ref_transform is not None:
            dst = np.empty(ref_shape, dtype=np.float32)
            reproject(
                source=data,
                destination=dst,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.average,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )
            return dst, ref_transform, ref_crs

        # 首次生成参考网格
        scale = target_res / src.res[0]
        new_h = int(data.shape[0] / scale)
        new_w = int(data.shape[1] / scale)

        dst = np.empty((new_h, new_w), dtype=np.float32)

        dst_transform = Affine(
            out_transform.a * scale, 0, out_transform.c,
            0, out_transform.e * scale, out_transform.f
        )

        reproject(
            source=data,
            destination=dst,
            src_transform=out_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.average,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )

        return dst, dst_transform, src.crs

# ================== 参考网格（1990 年 SOC） ==================
ref_year = years[0]
ref_file = subfolders['SOC'] / folders['SOC'].format(year=ref_year)
ref_data, ref_transform, ref_crs = process_raster(ref_file, agg_resolution)
ref_shape = ref_data.shape

rows, cols = np.indices(ref_shape)
xs = ref_transform.c + (cols + 0.5) * ref_transform.a
ys = ref_transform.f + (rows + 0.5) * ref_transform.e
coords_x = xs.ravel()
coords_y = ys.ravel()

# ================== 主循环（年度 + tqdm） ==================
all_years_data = []

static_k = None
static_ls = None

for year in tqdm(years, desc="Processing years"):
    year_dict = {
        'x': coords_x,
        'y': coords_y,
        't': (year - years[0]) / (len(years) - 1)
    }

    # SOC 与侵蚀
    soc, _, _ = process_raster(
        subfolders['SOC'] / folders['SOC'].format(year=year),
        agg_resolution,
        ref_transform,
        ref_shape,
        ref_crs
    )
    ero, _, _ = process_raster(
        subfolders['erosion'] / folders['erosion'].format(year=year),
        agg_resolution,
        ref_transform,
        ref_shape,
        ref_crs
    )

    year_dict['SOC_loss'] = soc.ravel()
    year_dict['erosion'] = ero.ravel()

    # 年变因子
    for fac in ['C', 'P', 'R']:
        f, _, _ = process_raster(
            subfolders[fac] / folders[fac].format(year=year),
            agg_resolution,
            ref_transform,
            ref_shape,
            ref_crs
        )
        year_dict[fac] = f.ravel()

    # 静态因子
    if static_k is None:
        k, _, _ = process_raster(
            subfolders['K'] / folders['K'],
            agg_resolution,
            ref_transform,
            ref_shape,
            ref_crs
        )
        ls, _, _ = process_raster(
            subfolders['LS'] / folders['LS'],
            agg_resolution,
            ref_transform,
            ref_shape,
            ref_crs
        )
        static_k = k.ravel()
        static_ls = ls.ravel()

    year_dict['K'] = static_k
    year_dict['LS'] = static_ls

    df_year = pd.DataFrame(year_dict)

    # ★ 最终筛选规则：SOC_loss ≠ NaN OR erosion ≠ NaN
    valid_mask = ~(df_year['SOC_loss'].isna() & df_year['erosion'].isna())
    df_year = df_year[valid_mask].reset_index(drop=True)

    # --------- 单年输出 ---------
    csv_path = out_root / f"Result_data{year}.csv"
    npz_path = out_root / f"Result_data{year}.npz"

    df_year.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        **{c: df_year[c].values for c in df_year.columns}
    )

    all_years_data.append(df_year)

# ================== 合并所有年份 ==================
full_df = pd.concat(all_years_data, ignore_index=True)

full_df.to_csv(out_root / "Result_data_all.csv", index=False)
np.savez_compressed(
    out_root / "Result_data_all.npz",
    **{c: full_df[c].values for c in full_df.columns}
)

print("全部处理完成")
print(f"结果已保存至：{out_root}")
print(f"总样本数：{len(full_df)}")
