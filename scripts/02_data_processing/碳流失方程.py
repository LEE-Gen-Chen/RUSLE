import numpy as np
import rasterio
import pandas as pd
import os
import pyodbc
from tqdm import tqdm


def read_hwsd_soc(mdb_path):
    """
    从HWSD MDB文件中仅读取表层土壤有机碳数据

    参数:
    mdb_path: MDB文件路径

    返回:
    DataFrame: 包含MU_GLOBAL和T_OC的DataFrame
    """
    print("正在读取MDB文件中的有机碳数据...")

    # 构建连接字符串
    conn_str = (
            r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
            r'DBQ=' + mdb_path + ';'
    )

    try:
        # 连接数据库
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # 精简查询：仅提取计算所需的必要字段
        # T_OC: 表层有机碳含量(%)
        # SEQ=1: 提取主要土壤单元（占比最高的）
        query = """
        SELECT MU_GLOBAL, T_OC
        FROM HWSD_DATA
        WHERE SEQ = 1
        """

        # 执行查询
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
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
        return read_hwsd_soc_alternative(mdb_path)


def read_hwsd_soc_alternative(mdb_path):
    """
    备选方法读取MDB文件中的有机碳数据
    """
    try:
        import accessdb
        attributes = accessdb.read_table(mdb_path, 'HWSD_DATA')
        # 筛选主要土壤单元
        attributes = attributes[attributes['SEQ'] == 1][['MU_GLOBAL', 'T_OC']]
        return attributes
    except:
        print("无法直接读取MDB文件，请手动将HWSD_DATA表导出为CSV格式")
        print("导出步骤:")
        print("1. 使用Microsoft Access打开HWSD.mdb")
        print("2. 找到HWSD_DATA表")
        print("3. 右键点击表，选择'导出' -> '文本文件'或'Excel'")
        print("4. 保存为CSV格式")
        print("5. 将CSV文件路径替换到脚本中")
        exit(1)


def calculate_soil_carbon_loss_gd(hwsd_raster_path, hwsd_mdb_path,
                                  erosion_raster_path, output_path, Er=0.855):
    """
    使用HWSD数据和RUSLE侵蚀模数计算土壤有机carbon量

    参数:
    hwsd_raster_path: 已投影和裁剪的HWSD栅格数据路径（广东省）
    hwsd_mdb_path: HWSD MDB文件路径
    erosion_raster_path: RUSLE侵蚀模数栅格路径（单位：t/(km²·a)）
    output_path: 输出carbon量栅格路径（单位：t/(km²·a)）
    Er: 侵蚀泥沙富集系数（默认0.855）
    """
    print("=" * 50)
    print("土壤有机carbon量计算")
    print("=" * 50)

    # 1. 读取HWSD栅格数据（获取MU_GLOBAL空间分布）
    print("\n[1/4] 正在读取广东省HWSD栅格数据...")
    with rasterio.open(hwsd_raster_path) as src:
        hwsd_data = src.read(1)
        src_profile = src.profile
        src_crs = src.crs
        src_transform = src.transform

        print(f"   数据信息: CRS={src_crs}")
        print(f"   分辨率: {src.res[0]:.6f}° × {src.res[1]:.6f}°")
        print(f"   尺寸: {src.width} × {src.height} 像元")

    # 2. 读取RUSLE侵蚀模数栅格
    print("\n[2/4] 正在读取RUSLE侵蚀模数栅格...")
    with rasterio.open(erosion_raster_path) as erosion_src:
        erosion_data = erosion_src.read(1)
        # 验证空间一致性
        if erosion_src.shape != hwsd_data.shape:
            raise ValueError("侵蚀模数栅格与HWSD栅格尺寸不匹配！")
        if erosion_src.crs != src_crs:
            print("   警告：CRS不完全一致，请确保数据已正确配准")

        print(f"   侵蚀模数范围: {np.nanmin(erosion_data):.2f} - {np.nanmax(erosion_data):.2f} t/(km²·a)")

    # 3. 从MDB文件读取有机碳数据
    print("\n[3/4] 正在提取表层土壤有机碳含量...")
    attributes = read_hwsd_soc(hwsd_mdb_path)

    # 关键步骤：单位转换和创建映射
    # T_OC单位: % → 转换为g/kg (乘以10)
    attributes['C_g_kg'] = attributes['T_OC'] * 10.0

    # 创建MU_GLOBAL到有机碳的映射字典
    mu_to_soc = dict(zip(attributes['MU_GLOBAL'], attributes['C_g_kg']))

    # 统计信息
    valid_oc = attributes['C_g_kg'].dropna()
    print(f"   有机碳含量范围: {valid_oc.min():.2f} - {valid_oc.max():.2f} g/kg")
    print(f"   平均值: {valid_oc.mean():.2f} g/kg")

    # 4. 创建有机碳含量空间栅格
    print("\n[4/4] 正在创建有机碳空间分布栅格...")

    def create_soc_raster(mu_array, soc_mapping):
        """将MU_GLOBAL数组转换为有机碳含量数组"""
        soc_raster = np.full(mu_array.shape, np.nan, dtype=np.float32)
        unique_mu = np.unique(mu_array)

        # 使用tqdm显示进度
        for mu_value in tqdm(unique_mu, desc="   处理土壤单元"):
            if mu_value in soc_mapping:
                mask = mu_array == mu_value
                soc_raster[mask] = soc_mapping[mu_value]

        return soc_raster

    # 生成有机碳栅格
    soc_raster = create_soc_raster(hwsd_data, mu_to_soc)

    # 5. 计算土壤有机carbon量
    print("\n正在计算carbon量...")
    print(f"   使用富集系数 Er = {Er}")

    # 创建有效数据掩膜
    valid_mask = (~np.isnan(soc_raster)) & (~np.isnan(erosion_data))

    # 初始化结果数组
    carbon_loss_raster = np.full(hwsd_data.shape, np.nan, dtype=np.float32)

    # 应用计算公式: N = (A × C × E_r) / 1000
    # A: 侵蚀模数 [t/(km²·a)] - 已有
    # C: 有机碳 [g/kg] - 已转换
    # Er: 富集系数 - 给定
    # 无需额外单位转换，结果直接为 t/(km²·a)

    carbon_loss_raster[valid_mask] = (
            erosion_data[valid_mask] *
            soc_raster[valid_mask] *
            Er / 1000.0
    )

    # 6. 保存结果栅格
    print("\n正在保存结果...")

    # 更新元数据
    src_profile.update({
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw',
        'interleave': 'band'
    })

    # 写入GeoTIFF
    with rasterio.open(output_path, 'w', **src_profile) as dst:
        dst.write(carbon_loss_raster, 1)

    print(f"   ✓ 结果已保存至: {output_path}")

    # 7. 输出统计信息
    valid_loss = carbon_loss_raster[~np.isnan(carbon_loss_raster)]
    if len(valid_loss) > 0:
        print(f"\n{'=' * 50}")
        print("广东省土壤有机carbon量统计")
        print(f"{'=' * 50}")
        print(f"  有效像元数: {len(valid_loss):,}")
        print(f"  最小值: {np.min(valid_loss):.4f} t/(km²·a)")
        print(f"  最大值: {np.max(valid_loss):.4f} t/(km²·a)")
        print(f"  平均值: {np.mean(valid_loss):.4f} t/(km²·a)")
        print(f"  标准差: {np.std(valid_loss):.4f} t/(km²·a)")
        print(f"  总量: {np.sum(valid_loss):,.2f} t/a")

        # 分级统计
        bins = [0, 0.1, 0.5, 1.0, 5.0, np.inf]
        labels = ['极低', '低', '中等', '高', '极高']
        hist, _ = np.histogram(valid_loss, bins=bins)
        print(f"\n  分级统计:")
        for i, label in enumerate(labels):
            print(f"    {label}: {hist[i]:,} 像元 ({hist[i] / len(valid_loss) * 100:.1f}%)")

    return carbon_loss_raster


def main():
    """主函数"""

    # ==================== 输入参数设置 ====================
    # 已投影和裁剪的广东省HWSD栅格数据（MU_GLOBAL）
    hwsd_raster_path = r"./data\soil\HWSD_RASTER\hwsd_region.tif"

    # HWSD属性数据库
    hwsd_mdb_path = r"./data\soil\HWSD.mdb"

    # RUSLE侵蚀模数栅格（单位：t/(km²·a)）
    erosion_raster_path = r"./data\RUSLE\erosion_modulus_GD.tif"

    # 输出路径
    output_path = r"./data\carbon\soil_carbon_loss_GD.tif"

    # 富集系数（可根据研究调整）
    ER_COEFFICIENT = 0.855

    # =====================================================

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"""
    计算配置:
    - 研究区域: 广东省
    - 侵蚀模数: {erosion_raster_path}
    - 富集系数: {ER_COEFFICIENT}
    - 输出路径: {output_path}
    """)

    try:
        # 执行计算
        result = calculate_soil_carbon_loss_gd(
            hwsd_raster_path=hwsd_raster_path,
            hwsd_mdb_path=hwsd_mdb_path,
            erosion_raster_path=erosion_raster_path,
            output_path=output_path,
            Er=ER_COEFFICIENT
        )

        print("\n" + "=" * 50)
        print("✓ 土壤有机carbon量计算完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()