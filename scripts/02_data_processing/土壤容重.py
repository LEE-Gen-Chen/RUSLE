import numpy as np
import rasterio
import pandas as pd
import os
import pyodbc
from tqdm import tqdm


def read_hwsd_bd_weighted(mdb_path):
    """
    从HWSD MDB文件中提取表层土壤容重（加权平均版本）
    根据HWSD文档第13-14页：T_REF_BULK_DENSITY字段提供表土(0-30cm)参考容重
    单位：kg/dm³ (等价于g/cm³)
    """
    print("正在读取MDB文件中的容重数据（加权平均模式）...")
    conn_str = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + mdb_path

    try:
        conn = pyodbc.connect(conn_str)
        # 提取所有土壤单元（SEQ=1-9）的容重和占比
        query = """
        SELECT MU_GLOBAL, T_REF_BULK_DENSITY, SHARE 
        FROM HWSD_DATA 
        WHERE SEQ >= 1 AND SEQ <= 9
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # 数据验证：检查SHARE总和是否为100%
        share_sum = df.groupby('MU_GLOBAL')['SHARE'].sum()
        problematic_units = share_sum[(share_sum < 99.9) | (share_sum > 100.1)]
        if len(problematic_units) > 0:
            print(f"警告：发现{len(problematic_units)}个制图单元SHARE总和不等于100%")

        # 单位说明：kg/dm³ = g/cm³，无需转换
        # 验证数据范围合理性（HWSD文档：典型值0.9-1.8 g/cm³）
        bd_values = df['T_REF_BULK_DENSITY'].dropna()
        if np.any(bd_values < 0.5) or np.any(bd_values > 2.5):
            print("警告：部分BD值超出合理范围[0.5, 2.5] g/cm³，请检查数据质量")

        print(f"   容重范围: {bd_values.min():.3f} - {bd_values.max():.3f} g/cm³")
        print(f"   平均值: {bd_values.mean():.3f} g/cm³")

        # 按MU_GLOBAL进行加权平均计算
        weighted_bd = df.groupby('MU_GLOBAL').apply(
            lambda x: np.average(x['T_REF_BULK_DENSITY'], weights=x['SHARE'])
        ).reset_index(name='BD_g_cm3')

        print(f"✓ 成功读取并加权平均 {len(weighted_bd)} 个制图单元")
        return dict(zip(weighted_bd['MU_GLOBAL'], weighted_bd['BD_g_cm3']))

    except Exception as e:
        print(f"✗ 读取MDB文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


def extract_bd_raster(hwsd_raster_path, hwsd_mdb_path, output_bd_path):
    """
    从HWSD数据中提取表层土壤容重栅格（0-30cm）

    参数:
    hwsd_raster_path: 已投影裁剪的广东省HWSD栅格（存储MU_GLOBAL）
    hwsd_mdb_path: HWSD MDB文件路径
    output_bd_path: 输出的BD栅格路径（单位：g/cm³）
    """
    print("=" * 60)
    print("HWSD表层土壤容重(BD)提取")
    print("=" * 60)

    # 1. 读取HWSD栅格
    print("\n[步骤1] 读取HWSD栅格...")
    with rasterio.open(hwsd_raster_path) as src:
        hwsd_data = src.read(1)
        src_profile = src.profile
        print(f"   栅格尺寸: {src.width} × {src.height} 像元")
        print(f"   坐标系统: {src.crs}")
        print(f"   分辨率: {src.res[0]:.6f}° × {src.res[1]:.6f}°")

    # 2. 读取并处理容重属性数据（加权平均版本）
    print("\n[步骤2] 提取容重属性（加权平均）...")
    mu_to_bd = read_hwsd_bd_weighted(hwsd_mdb_path)

    # 验证数据范围
    bd_values = np.array(list(mu_to_bd.values()))
    if np.any(bd_values < 0) or np.any(bd_values > 3):
        print("警告：部分BD值超出物理合理范围[0, 3] g/cm³")

    print(f"   容重范围: {np.min(bd_values):.3f} - {np.max(bd_values):.3f} g/cm³")
    print(f"   平均值: {np.mean(bd_values):.3f} g/cm³")

    # 3. 创建BD空间分布栅格（性能优化）
    print("\n[步骤3] 生成BD空间栅格...")
    # 使用np.vectorize替代循环，提升性能
    vectorized_map = np.vectorize(lambda x: mu_to_bd.get(x, np.nan))
    bd_raster = vectorized_map(hwsd_data)

    # 转换为float32格式
    bd_raster = bd_raster.astype(np.float32)

    # 4. 保存BD栅格
    print("\n[步骤4] 保存BD栅格...")

    # 更新元数据
    src_profile.update({
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw',
        'interleave': 'band',
        'count': 1,
        # 添加描述信息
        'units': 'g/cm³',
        'description': 'HWSD topsoil bulk density (0-30cm), weighted by SHARE'
    })

    # 写入GeoTIFF
    with rasterio.open(output_bd_path, 'w', **src_profile) as dst:
        dst.write(bd_raster, 1)

    print(f"   ✓ BD栅格已保存至: {output_bd_path}")

    # 5. 输出统计信息
    valid_bd = bd_raster[~np.isnan(bd_raster)]
    if len(valid_bd) > 0:
        print(f"\n广东省表层土壤容重(0-30cm)统计:")
        print(f"  有效像元数: {len(valid_bd):,}")
        print(f"  最小值: {np.min(valid_bd):.3f} g/cm³")
        print(f"  最大值: {np.max(valid_bd):.3f} g/cm³")
        print(f"  平均值: {np.mean(valid_bd):.3f} g/cm³")
        print(f"  标准差: {np.std(valid_bd):.3f} g/cm³")

        # 分级统计（FAO推荐范围）
        bins = [0, 1.0, 1.4, 1.6, 2.0, np.inf]
        labels = ['极低 (<1.0)', '低 (1.0-1.4)', '中等 (1.4-1.6)', '高 (1.6-2.0)', '极高 (>2.0)']
        hist, _ = np.histogram(valid_bd, bins=bins)
        print(f"\n  分级统计:")
        for i, label in enumerate(labels):
            print(f"    {label}: {hist[i]:,} 像元 ({hist[i] / len(valid_bd) * 100:.1f}%)")

    return bd_raster


def main():
    """主函数"""
    # ==================== 参数设置 ====================
    # HWSD栅格数据（MU_GLOBAL）
    hwsd_raster_path = r"./data\soil\HWSD_RASTER\hwsd_region.tif"

    # HWSD属性数据库
    hwsd_mdb_path = r"./data\soil\HWSD.mdb"

    # 输出BD栅格路径
    output_bd_path = r"./data\土壤属性\BD_30_weighted.tif"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_bd_path), exist_ok=True)
    # ==================================================

    print(f"""
    HWSD容重提取配置（加权平均版本）
    ================================
    研究区域: 广东省
    输出路径: {output_bd_path}
    数据深度: 0-30 cm
    单位: g/cm³ (kg/dm³)
    计算方法: 按SHARE加权平均所有土壤单元(SEQ=1-9)
    数据来源: T_REF_BULK_DENSITY字段
    ================================
    """)

    try:
        extract_bd_raster(
            hwsd_raster_path=hwsd_raster_path,
            hwsd_mdb_path=hwsd_mdb_path,
            output_bd_path=output_bd_path
        )

        print("\n" + "=" * 60)
        print("✓ BD栅格提取完成！")
        print("=" * 60)
        print("\n【地表carbon计算准备】")
        print("1. 在ArcGIS/Python中加载以下数据：")
        print(f"   - SOC栅格: SOC_30_weighted.tif")
        print(f"   - BD栅格: {os.path.basename(output_bd_path)}")
        print("   - 侵蚀模数栅格: A_YYYY.tif (单位: t/ha/yr)")
        print("\n2. 使用以下方程计算有机carbon量：")
        print("   SOC_loss = SOC × BD × 30 × (A × 10) × 0.855")
        print("   单位: g C m⁻² yr⁻¹")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()