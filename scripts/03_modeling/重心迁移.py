"""
SOC 横向流失空间重心迁移图
1990-2024 年，每5年一个重心点
t 归一化解算：year = round(1990 + t × 34)
"""

import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════
NPZ_PATH  = r'./data\Data\GTNNWR\Result_data_all.npz'
SHP_PATH  = r'./data\Data\GD_SHP\GD.shp'
FONT_PATH = r'H:\安装包\Simei.ttf'
OUT_PATH  = r'./data\SOC_centroid_migration.png'

YEAR_MIN  = 1990
YEAR_MAX  = 2024
YEARS     = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]

# ═══════════════════════════════════════════════════════════
# 1. 读取数据，将 t 还原为整数年份
# ═══════════════════════════════════════════════════════════
print("正在读取数据（约 6500 万条记录）…")
raw     = np.load(NPZ_PATH, allow_pickle=True)
x_all   = raw['x']
y_all   = raw['y']
t_all   = raw['t']
soc_all = raw['SOC_loss']

# t → year：线性反归一化，四舍五入到整数年
year_all = np.round(YEAR_MIN + t_all * (YEAR_MAX - YEAR_MIN)).astype(np.int16)

# 验证解算结果
unique_years, counts = np.unique(year_all, return_counts=True)
print("\n【t 解算后的年份分布】")
for yr, cnt in zip(unique_years, counts):
    print(f"  {yr} 年：{cnt:>10,} 条记录")

# ═══════════════════════════════════════════════════════════
# 2. 对每个目标年份计算加权空间重心
# ═══════════════════════════════════════════════════════════
def calc_weighted_centroid(x, y, w):
    """以 SOC_loss 为权重计算加权质心"""
    w_d   = w.astype(np.float64)
    w_sum = w_d.sum()
    cx    = np.dot(w_d, x.astype(np.float64)) / w_sum
    cy    = np.dot(w_d, y.astype(np.float64)) / w_sum
    return cx, cy, w_sum


print("\n【各年份空间重心（投影坐标 m）】")
centroids = {}
for yr in YEARS:
    mask = (year_all == yr) & (~np.isnan(soc_all)) & (soc_all > 0)
    n    = int(mask.sum())
    if n == 0:
        print(f"  {yr} 年：无有效数据，跳过")
        continue
    cx, cy, w_sum = calc_weighted_centroid(
        x_all[mask], y_all[mask], soc_all[mask])
    centroids[yr] = (cx, cy, w_sum)
    print(f"  {yr} 年  {n:>9,} pts  重心=({cx:,.0f}, {cy:,.0f})  总流失={w_sum:.3e}")

years_ok = sorted(centroids.keys())

# ═══════════════════════════════════════════════════════════
# 3. 绘图
# ═══════════════════════════════════════════════════════════
fp = FontProperties(fname=FONT_PATH)
matplotlib.rcParams['axes.unicode_minus'] = False

gdf    = gpd.read_file(SHP_PATH)    # EPSG:4547
bounds = gdf.total_bounds           # (minx, miny, maxx, maxy)

CMAP = plt.get_cmap('RdYlBu_r')
norm = Normalize(vmin=YEAR_MIN, vmax=YEAR_MAX)

fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(False)  # 取消格网

# ── 底图 ──────────────────────────────────────────────────
gdf.plot(ax=ax,
         facecolor='#EFEFEF',
         edgecolor='#888888',
         linewidth=0.45,
         zorder=1)

# ── 重心点 ────────────────────────────────────────────────
DOT_SIZE  = 28
HALO_SIZE = 80
LABEL_FS  = 6.2

# 每年独立偏移 (dx_m, dy_m, ha, va)
# 根据图中实际点位分布，将标注向四周散开，避免压盖
# 正方向：dx>0 向东，dy>0 向北
LABEL_OFFSET = {
    1990: ( 28000,  22000, 'left',  'bottom'),  # 右上
    1995: ( 32000,   2000, 'left',  'center'),  # 右
    2000: ( 12000,  30000, 'center','bottom'),  # 正上
    2005: ( 32000, -22000, 'left',  'top'   ),  # 右下
    2010: (-20000,  18000, 'right', 'bottom'),  # 左上（孤立点，引线短）
    2015: (-32000,   8000, 'right', 'center'),  # 左
    2020: (-28000, -22000, 'right', 'top'   ),  # 左下
    2024: (10000, -52000, 'center','top'   ),  # 正下
}

for yr in years_ok:
    cx, cy, _ = centroids[yr]
    color = CMAP(norm(yr))
    dx, dy, ha, va = LABEL_OFFSET.get(yr, (15000, 10000, 'left', 'bottom'))

    # 半透明光晕
    ax.scatter(cx, cy, s=HALO_SIZE, zorder=4, linewidths=0,
               color=(*color[:3], 0.20))
    # 主点
    ax.scatter(cx, cy, s=DOT_SIZE, zorder=5,
               color=color, edgecolors='white', linewidths=0.6)

    # 引导细线（从点到文字起点，长度为偏移量的 70%）
    lx = cx + dx * 0.78
    ly = cy + dy * 0.78
    ax.plot([cx, lx], [cy, ly],
            color='#777777', lw=0.55, zorder=8,
            solid_capstyle='round')

    # 年份标注
    txt = ax.text(cx + dx, cy + dy, str(yr),
                  fontsize=LABEL_FS, fontproperties=fp,
                  color='#111111', ha=ha, va=va, zorder=7)
    txt.set_path_effects([
        pe.withStroke(linewidth=2.2, foreground='white')
    ])

# ── 图例：彩色圆点 + 年份，放置在右下角 ──────────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0],
           marker='o', color='none',
           markerfacecolor=CMAP(norm(yr)),
           markeredgecolor='white',
           markeredgewidth=0.5,
           markersize=5,
           label=str(yr))
    for yr in years_ok
]
legend = ax.legend(
    handles=legend_handles,
    title='重心年份',
    title_fontproperties=fp,
    prop=fp,

    fontsize=4,  # ↓ 字体大幅缩小
    loc='lower right',

    frameon=True,
    framealpha=0.9,
    edgecolor='#AAAAAA',
    facecolor='white',

    borderpad=0.1,  # ↓ 外边距极小
    labelspacing=0.1,  # ↓ 行距极小
    handletextpad=0.2,  # ↓ 点-字间距
    borderaxespad=0.1,

    handlelength=0.6,  # ↓ 横向长度
    handleheight=0.4,

    markerscale=0.6,  # ⭐ 核心：点缩小到一半
)

legend.get_title().set_fontsize(4.5)   # 标题正常
for txt in legend.get_texts():
    txt.set_fontsize(4)             # 年份更小
legend.get_frame().set_linewidth(0.4)

# # ── 比例尺（左下角）────────────────────────────────────
# # ── 比例尺（底部正中）────────────────────────────
# map_width = bounds[2] - bounds[0]
#
# sb_len = 100_000  # 100 km
#
# # ⭐ 关键：居中
# sb_x0 = bounds[0] + map_width / 2 - sb_len / 2
#
# # ⭐ 底部（稍微抬一点，避免贴边）
# sb_y0 = bounds[1] + 30000
#
# # 主线
# ax.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
#         color='#333333', lw=1.5, zorder=10)
#
# # 刻度
# for tx in [sb_x0, sb_x0 + sb_len/2, sb_x0 + sb_len]:
#     ax.plot([tx, tx], [sb_y0 - 4000, sb_y0 + 4000],
#             color='#333333', lw=1.2, zorder=10)
#
# # 标注（放下面更规范）
# ax.text(sb_x0, sb_y0 - 9000, '0',
#         ha='center', fontsize=6.5, fontproperties=fp)
#
# ax.text(sb_x0 + sb_len/2, sb_y0 - 9000, '50',
#         ha='center', fontsize=6.5, fontproperties=fp)
#
# ax.text(sb_x0 + sb_len, sb_y0 - 9000, '100 km',
#         ha='center', fontsize=6.5, fontproperties=fp)

# ── 指北针（右上角）────────────────────────────────────
na_x = bounds[2] - 35000
na_y = bounds[3] - 10000
ax.annotate('',
            xy=(na_x, na_y + 35000), xytext=(na_x, na_y),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333',
                            mutation_scale=14),
            zorder=10)
ax.text(na_x, na_y + 43000, 'N',
        ha='center', va='bottom', fontsize=15,
        fontproperties=fp, color='#333333',
        fontweight='bold', zorder=10)

# # ── 坐标轴（km）─────────────────────────────────────────
# ax.xaxis.set_major_formatter(
#     mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}'))
# ax.yaxis.set_major_formatter(
#     mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}'))
# ax.set_xlabel('东向距 / Easting (km)',
#               fontproperties=fp, fontsize=9, labelpad=5)
# ax.set_ylabel('北向距 / Northing (km)',
#               fontproperties=fp, fontsize=9, labelpad=5)

# for lbl in ax.get_xticklabels() + ax.get_yticklabels():
#     lbl.set_fontproperties(fp)
#     lbl.set_fontsize(7.5)
#
# ax.tick_params(direction='in', length=3.5, width=0.6, top=True, right=True)
# for spine in ax.spines.values():
#     spine.set_linewidth(0.6)

# ── 标题 ──────────────────────────────────────────────────
ax.set_title(
    '1990–2024 年广东省 SOC 横向流失空间重心迁移\n'
    'Spatial Centroid Migration of SOC Lateral Loss in Guangdong (1990–2024)',
    fontproperties=fp, fontsize=10, pad=10, color='#111111')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'\n[OK] 已保存: {OUT_PATH}')
plt.show()