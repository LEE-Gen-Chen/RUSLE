import os
import math
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import rasterize
from rasterio.transform import Affine
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =============================
# 参数（按需修改）
# =============================
input_folder = r"./data\NDVI\C_value"           # 原始 100m C 文件
output_folder = r"./data\NDVI\C_value_30m"     # 输出 30m 文件
mask_shp = r"./data\Geoscene\GD.shp"           # 掩膜矢量
target_resolution = 30                               # 目标分辨率（m）
target_crs = "YOUR_CRS_EPSG"                             # 输出投影
nodata_val = -9999
MAX_WORKERS = 1                                      # 并发线程数：建议 1 或 2（根据内存调整）
os.makedirs(output_folder, exist_ok=True)

# =============================
# 读取掩膜并投影到目标 CRS
# =============================
gdf = gpd.read_file(mask_shp).to_crs(target_crs)
mask_geoms = [geom.__geo_interface__ for geom in gdf.geometry]
minx, miny, maxx, maxy = gdf.total_bounds
print(f"掩膜边界 (投影 {target_crs})：{(minx, miny, maxx, maxy)}")

# =============================
# 列出输入文件
# =============================
c_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".tif") and f.startswith("C_NDVI_")])
print(f"检测到 {len(c_files)} 个 C 值文件。")

# =============================
# 处理单个文件的函数
# =============================
def process_file(file_name):
    try:
        src_path = os.path.join(input_folder, file_name)
        dst_path = os.path.join(output_folder, file_name)

        with rasterio.open(src_path) as src:
            # === 1) 计算目标整体栅格的仿射、宽高（如果整张重投影）
            transform_full, width_full, height_full = calculate_default_transform(
                src.crs, target_crs,
                src.width, src.height, *src.bounds,
                resolution=target_resolution
            )

            # === 2) 计算掩膜的包围盒在目标栅格上的列/行索引（窗口）
            # 将边界坐标（minx,miny,maxx,maxy）转为目标栅格的 col,row
            # rasterio.transform.rowcol 使用 (transform, xs, ys)
            # 注意 y 方向：rowcol(transform, x, y) 返回 (row, col)
            # 我们用 math.floor / math.ceil 做保守包含
            from rasterio.transform import rowcol

            # 左上点 (minx, maxy) -> row_min, col_min
            row_min, col_min = rowcol(transform_full, minx, maxy, op=math.floor)
            # 右下点 (maxx, miny) -> row_max, col_max
            row_max, col_max = rowcol(transform_full, maxx, miny, op=math.ceil)

            # 限制在范围内
            col_min = max(0, col_min)
            row_min = max(0, row_min)
            col_max = min(width_full - 1, col_max)
            row_max = min(height_full - 1, row_max)

            # 如果掩膜不与源影像重叠，跳过
            if (col_min >= col_max) or (row_min >= row_max):
                return f"{file_name} ✖ 掩膜在目标投影下与影像无重叠，已跳过。"

            dst_win_width = col_max - col_min + 1
            dst_win_height = row_max - row_min + 1

            # === 3) 计算这个窗口对应的仿射变换
            dst_transform = transform_full * Affine.translation(col_min, row_min)

            # === 4) 创建一个临时输出文件（仅为掩膜包围盒大小）并把 reproject 写入磁盘
            tmp_reproj_path = dst_path.replace(".tif", "_tmp_reproj.tif")
            dst_kwargs = {
                "driver": "GTiff",
                "height": dst_win_height,
                "width": dst_win_width,
                "count": 1,
                "dtype": "float32",
                "crs": target_crs,
                "transform": dst_transform,
                "nodata": nodata_val,
                "compress": "lzw"
            }

            # 使用 reproject 直接写入磁盘（不会建立整个大数组）
            with rasterio.open(tmp_reproj_path, "w", **dst_kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    dst_width=dst_win_width,
                    dst_height=dst_win_height,
                    resampling=Resampling.nearest,
                    num_threads=1
                )

        # === 5) 在小窗口上 rasterize 掩膜并应用掩膜（此处内存开销很小，因为是窗口大小）===
        with rasterio.open(tmp_reproj_path, "r+") as dst:
            out_meta = dst.meta.copy()
            # 创建掩膜：rasterize 返回 0/1 的 uint8 数组
            mask_arr = rasterize(
                [(geom, 1) for geom in mask_geoms],
                out_shape=(dst.height, dst.width),
                transform=dst.transform,
                fill=0,
                dtype='uint8'
            )
            # 读取当前数据块到内存（此时尺寸是窗口级别，远小于整图）
            data = dst.read(1)
            # 将掩膜外区域设为 nodata
            data[mask_arr == 0] = nodata_val
            dst.write(data, 1)
            dst.update_tags(1, HISTORY="Resampled to 30m, reprojected to EPSG:4547 and masked by GD.shp")

        # === 6) 把结果移动到最终路径（如果临时名不同），并删除中间文件 ===
        final_path = dst_path  # 保持同名输出
        os.replace(tmp_reproj_path, final_path)

        return f"{file_name} ✅ 完成（30m，EPSG:4547，按掩膜包围盒裁剪并掩膜）"

    except Exception as e:
        # 若存在临时文件则尝试删除以免残留
        try:
            if 'tmp_reproj_path' in locals() and os.path.exists(tmp_reproj_path):
                os.remove(tmp_reproj_path)
        except:
            pass
        return f"{file_name} ❌ 出错: {e}"

# =============================
# 并发执行（并发数小）
# =============================
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_file, f) for f in c_files]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
        print(fut.result())

print("全部处理结束。")
