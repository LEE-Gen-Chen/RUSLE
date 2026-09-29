import numpy as np
import rasterio
import pandas as pd
import os
import pyodbc
from tqdm import tqdm


def read_hwsd_mdb(mdb_path):
    """
    从HWSD MDB文件中读取属性数据

    参数:
    mdb_path: MDB文件路径

    返回:
    DataFrame: 包含土壤属性数据的DataFrame
    """
    print("正在读取MDB文件...")

    # 构建连接字符串
    conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + mdb_path + ';'
    )

    try:
        # 连接数据库
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # 查询HWSD_DATA表
        query = """
        SELECT MU_GLOBAL, T_SAND, T_SILT, T_CLAY, T_OC, 
               S_SAND, S_SILT, S_CLAY, S_OC
        FROM HWSD_DATA
        """

        # 执行查询
        cursor.execute(query)

        # 获取列名
        columns = [column[0] for column in cursor.description]

        # 获取所有数据
        data = cursor.fetchall()

        # 创建DataFrame
        attributes = pd.DataFrame.from_records(data, columns=columns)

        # 关闭连接
        cursor.close()
        conn.close()

        print(f"成功读取 {len(attributes)} 条土壤记录")
        return attributes

    except Exception as e:
        print(f"读取MDB文件失败: {str(e)}")
        # 尝试使用其他方法
        return read_hwsd_mdb_alternative(mdb_path)


def read_hwsd_mdb_alternative(mdb_path):
    """
    备选方法读取MDB文件
    """
    try:
        # 尝试使用pandas的access支持（需要安装accessdb）
        import accessdb
        attributes = accessdb.read_table(mdb_path, 'HWSD_DATA')
        return attributes
    except:
        # 如果所有方法都失败，提示用户手动导出
        print("无法直接读取MDB文件，请手动将HWSD_DATA表导出为CSV格式")
        print("导出步骤:")
        print("1. 使用Microsoft Access打开HWSD.mdb")
        print("2. 找到HWSD_DATA表")
        print("3. 右键点击表，选择'导出' -> '文本文件'或'Excel'")
        print("4. 保存为CSV格式")
        print("5. 将CSV文件路径替换到脚本中")
        exit(1)


def calculate_k_factor(hwsd_raster_path, hwsd_mdb_path, output_path):
    """
    使用已经投影和裁剪的Study Area HWSD栅格数据计算土壤可蚀性因子K值

    参数:
    hwsd_raster_path: 已经投影和裁剪的Study Area HWSD栅格数据路径
    hwsd_mdb_path: HWSD MDB文件路径
    output_path: 输出K值栅格路径
    """

    print("正在读取Study Area HWSD栅格数据...")

    # 读取已经投影和裁剪的Study Area HWSD栅格数据
    with rasterio.open(hwsd_raster_path) as src:
        hwsd_data = src.read(1)
        src_profile = src.profile
        src_crs = src.crs
        src_transform = src.transform
        src_bounds = src.bounds

        print(f"Study Area HWSD数据信息: CRS={src_crs}, 分辨率={src.res}, 尺寸={src.width}x{src.height}")

    # 从MDB文件读取属性数据
    attributes = read_hwsd_mdb(hwsd_mdb_path)

    # 创建从MU_GLOBAL到土壤参数的映射
    # 使用顶层数据 (0-30cm)
    mu_to_sand = dict(zip(attributes['MU_GLOBAL'], attributes['T_SAND']))
    mu_to_silt = dict(zip(attributes['MU_GLOBAL'], attributes['T_SILT']))
    mu_to_clay = dict(zip(attributes['MU_GLOBAL'], attributes['T_CLAY']))
    mu_to_oc = dict(zip(attributes['MU_GLOBAL'], attributes['T_OC']))

    print("正在计算土壤参数栅格...")

    # 将MU_GLOBAL映射为土壤参数
    def create_parameter_raster(mu_data, mapping):
        """创建土壤参数栅格"""
        param_raster = np.zeros_like(mu_data, dtype=np.float32)
        unique_mu = np.unique(mu_data)

        # 使用tqdm显示进度
        for mu_value in tqdm(unique_mu, desc="处理土壤单元"):
            if mu_value in mapping and not np.isnan(mapping[mu_value]):
                mask = mu_data == mu_value
                param_raster[mask] = mapping[mu_value]
            else:
                mask = mu_data == mu_value
                param_raster[mask] = np.nan

        return param_raster

    # 创建各土壤参数栅格
    print("处理沙粒含量...")
    sand_raster = create_parameter_raster(hwsd_data, mu_to_sand)

    print("处理粉粒含量...")
    silt_raster = create_parameter_raster(hwsd_data, mu_to_silt)

    print("处理粘粒含量...")
    clay_raster = create_parameter_raster(hwsd_data, mu_to_clay)

    print("处理有机碳含量...")
    oc_raster = create_parameter_raster(hwsd_data, mu_to_oc)

    print("正在计算K值...")

    # RUSLE K值计算公式
    def calculate_k(sand, silt, clay, oc):
        """计算土壤可蚀性因子K值"""

        # 避免除以零和无效值
        valid_mask = (~np.isnan(sand)) & (~np.isnan(silt)) & (~np.isnan(clay)) & (~np.isnan(oc))
        k_values = np.full(sand.shape, np.nan, dtype=np.float32)

        # 只在有效数据位置计算
        sand_valid = sand[valid_mask]
        silt_valid = silt[valid_mask]
        clay_valid = clay[valid_mask]
        oc_valid = oc[valid_mask]

        # 计算SN1
        SN1 = 1 - sand_valid / 100

        # RUSLE K值公式
        term1 = 0.2 + 0.3 * np.exp(-0.0256 * sand_valid * (1 - silt_valid / 100))

        # 避免除以零
        silt_clay_sum = silt_valid + clay_valid
        silt_clay_sum[silt_clay_sum == 0] = 1e-6  # 避免除以零
        term2 = (silt_valid / silt_clay_sum) ** 0.3

        # 有机碳项
        oc_term = oc_valid + np.exp(3.72 - 2.95 * oc_valid)
        oc_term[oc_term == 0] = 1e-6
        term3 = 1 - (0.25 * oc_valid) / oc_term

        # SN1项
        sn1_term = SN1 + np.exp(-5.51 + 22.9 * SN1)
        sn1_term[sn1_term == 0] = 1e-6
        term4 = 1 - (0.7 * SN1) / sn1_term

        # 计算K值
        k_valid = term1 * term2 * term3 * term4

        # 将有效值放回原数组
        k_values[valid_mask] = k_valid

        return k_values

    # 计算K值
    k_raster = calculate_k(sand_raster, silt_raster, clay_raster, oc_raster)

    print("正在保存K值结果...")

    # 更新元数据
    src_profile.update({
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw'
    })

    # 保存K值栅格
    with rasterio.open(output_path, 'w', **src_profile) as dst:
        dst.write(k_raster, 1)

    print(f"K值计算完成！结果已保存至: {output_path}")

    # 输出统计信息
    valid_k = k_raster[~np.isnan(k_raster)]
    if len(valid_k) > 0:
        print(f"\n广东省K值统计信息:")
        print(f"  最小值: {np.min(valid_k):.4f}")
        print(f"  最大值: {np.max(valid_k):.4f}")
        print(f"  平均值: {np.mean(valid_k):.4f}")
        print(f"  标准差: {np.std(valid_k):.4f}")
        print(f"  有效像元数: {len(valid_k)}")

    return k_raster


def main():
    """主函数"""

    # 输入文件路径
    hwsd_raster_path = r"./data/soil/hwsd_raster.tif"  # 已经投影和裁剪的Study Area HWSD数据
    hwsd_mdb_path = r"./data/soil/HWSD.mdb"

    # 输出文件路径
    output_path = r"./data/soil/K_factor.tif"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 计算K值
        k_result = calculate_k_factor(
            hwsd_raster_path,
            hwsd_mdb_path,
            output_path
        )

        print("\n处理完成！")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()