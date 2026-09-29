import numpy as np
import rasterio
import pandas as pd
import os
import pyodbc
from tqdm import tqdm


def read_hwsd_soc_weighted(mdb_path):
    """
    从HWSD MDB文件中提取表层土壤有机碳含量（加权平均版本）
    根据HWSD文档：每个制图单元(MU_GLOBAL)可包含1-9个土壤单元(SEQ)，
    需按SHARE字段进行加权平均，以反映真实土壤组成。
    """
    print("正在读取MDB文件中的有机碳数据（加权平均模式）...")
    conn_str = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + mdb_path

    try:
        conn = pyodbc.connect(conn_str)
        # 提取所有土壤单元（SEQ=1-9）的有机碳和占比
        query = """
        SELECT MU_GLOBAL, T_OC, SHARE 
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

        # 单位转换：% → g/kg (乘以10)
        df['C_g_kg'] = df['T_OC'] * 10.0

        # 按MU_GLOBAL进行加权平均计算
        # 权重为SHARE字段，反映各土壤单元在制图单元中的面积占比
        weighted_soc = df.groupby('MU_GLOBAL').apply(
            lambda x: np.average(x['C_g_kg'], weights=x['SHARE'])
        ).reset_index(name='C_g_kg')

        print(f"✓ 成功读取并加权平均 {len(weighted_soc)} 个制图单元")
        return dict(zip(weighted_soc['MU_GLOBAL'], weighted_soc['C_g_kg']))

    except Exception as e:
        print(f"✗ 读取MDB文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


def extract_soc_raster(hwsd_gd_raster_path, hwsd_mdb_path, output_soc_path):
    """
    从HWSD数据中提取表层土壤有机碳含量栅格（0-30cm）

    参数:
    hwsd_gd_raster_path: 已投影裁剪的广东省HWSD栅格（存储MU_GLOBAL）
    hwsd_mdb_path: HWSD MDB文件路径
    output_soc_path: 输出的SOC栅格路径（单位：g/kg）
    """
    print("=" * 60)
    print("HWSD表层土壤有机碳(SOC)提取")
    print("=" * 60)

    # 1. 读取HWSD栅格
    print("\n[步骤1] 读取HWSD栅格...")
    with rasterio.open(hwsd_gd_raster_path) as src:
        hwsd_data = src.read(1)
        src_profile = src.profile
        print(f"   栅格尺寸: {src.width} × {src.height} 像元")
        print(f"   坐标系统: {src.crs}")
        print(f"   分辨率: {src.res[0]:.6f}° × {src.res[1]:.6f}°")

    # 2. 读取并处理有机碳属性数据（加权平均版本）
    print("\n[步骤2] 提取有机碳属性（加权平均）...")
    mu_to_soc = read_hwsd_soc_weighted(hwsd_mdb_path)

    # 验证数据范围合理性（HWSD文档第14页：SOC通常<10%即<100 g/kg）
    soc_values = np.array(list(mu_to_soc.values()))
    if np.any(soc_values < 0) or np.any(soc_values > 100):
        print("警告：部分SOC值超出合理范围[0, 100] g/kg，请检查数据质量")

    print(f"   有机碳范围: {np.min(soc_values):.2f} - {np.max(soc_values):.2f} g/kg")
    print(f"   平均值: {np.mean(soc_values):.2f} g/kg")

    # 3. 创建SOC空间分布栅格（性能优化）
    print("\n[步骤3] 生成SOC空间栅格...")
    # 使用np.vectorize替代循环，提升性能
    vectorized_map = np.vectorize(lambda x: mu_to_soc.get(x, np.nan))
    soc_raster = vectorized_map(hwsd_data)

    # 转换为float32格式
    soc_raster = soc_raster.astype(np.float32)

    # 4. 保存SOC栅格
    print("\n[步骤4] 保存SOC栅格...")

    # 更新元数据
    src_profile.update({
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'lzw',
        'interleave': 'band',
        'count': 1,
        # 添加描述信息
        'units': 'g/kg',
        'description': 'HWSD topsoil organic carbon (0-30cm), weighted by SHARE'
    })

    # 写入GeoTIFF
    with rasterio.open(output_soc_path, 'w', **src_profile) as dst:
        dst.write(soc_raster, 1)

    print(f"   ✓ SOC栅格已保存至: {output_soc_path}")

    # 5. 输出统计信息
    valid_soc = soc_raster[~np.isnan(soc_raster)]
    if len(valid_soc) > 0:
        print(f"\n广东省表层土壤有机碳(0-30cm)统计:")
        print(f"  有效像元数: {len(valid_soc):,}")
        print(f"  最小值: {np.min(valid_soc):.2f} g/kg")
        print(f"  最大值: {np.max(valid_soc):.2f} g/kg")
        print(f"  平均值: {np.mean(valid_soc):.2f} g/kg")
        print(f"  标准差: {np.std(valid_soc):.2f} g/kg")

        # 分级统计（基于HWSD文档第14页的分类标准）
        bins = [0, 6, 12, 24, 48, np.inf]
        labels = ['极低 (<6)', '低 (6-12)', '中等 (12-24)', '高 (24-48)', '极高 (>48)']
        hist, _ = np.histogram(valid_soc, bins=bins)
        print(f"\n  分级统计:")
        for i, label in enumerate(labels):
            print(f"    {label}: {hist[i]:,} 像元 ({hist[i] / len(valid_soc) * 100:.1f}%)")

    return soc_raster


def main():
    """主函数"""
    # ==================== 参数设置 ====================
    # HWSD栅格数据（MU_GLOBAL）
    hwsd_gd_raster_path = r"./data\中国土壤数据集\HWSD_RASTER\hwsd_GD.tif"

    # HWSD属性数据库
    hwsd_mdb_path = r"./data\中国土壤数据集\HWSD.mdb"

    # 输出SOC栅格路径
    output_soc_path = r"./data\土壤属性\SOC_30_weighted.tif"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_soc_path), exist_ok=True)
    # ==================================================

    print(f"""
    HWSD有机碳提取配置（加权平均版本）
    ================================
    研究区域: 广东省
    输出路径: {output_soc_path}
    数据深度: 0-30 cm
    单位: g/kg
    计算方法: 按SHARE加权平均所有土壤单元(SEQ=1-9)
    ================================
    """)

    try:
        extract_soc_raster(
            hwsd_gd_raster_path=hwsd_gd_raster_path,
            hwsd_mdb_path=hwsd_mdb_path,
            output_soc_path=output_soc_path
        )

        print("\n" + "=" * 60)
        print("✓ SOC栅格提取完成！")
        print("=" * 60)
        print("\n【下一步：地表碳流失计算】")
        print("1. 在ArcGIS/Python中加载以下数据：")
        print(f"   - SOC栅格: {os.path.basename(output_soc_path)}")
        print("   - 侵蚀模数栅格: A_YYYY.tif (单位: t/ha/yr)")
        print("   - 容重栅格: BD.tif (单位: g/cm³)")
        print("\n2. 使用以下正确方程计算有机碳流失量：")
        print("   SOC_loss = SOC × BD × 30 × (A × 10) × 0.855")
        print("   单位: g C m⁻² yr⁻¹")
        print("\n3. Python实现示例：")
        print("   soc_loss = Raster(soc_path) * Raster(bd_path) * 30 * (Raster(erosion_path) * 10) * 0.855")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()