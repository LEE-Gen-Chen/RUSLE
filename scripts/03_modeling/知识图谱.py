import plotly.graph_objects as go
import networkx as nx
import numpy as np
import random

# 设置随机种子，确保布局可复现
random.seed(42)
np.random.seed(42)

# 1. 创建有向图
G = nx.DiGraph()

# 2. 节点分类（与之前一致）
nodes_by_category = {
    '研究区域': ['云南省', '湟水流域', '淮河流域', '元阳县', '三江源', '巢湖流域', '黄土高原', '拉萨河流域',
                 '皇甫川流域', '朝阳市', '南方丘陵山区', '长汀县河田盆地区', '澜沧江流域', '延河流域', '红枫湖流域',
                 '丽江县'],
    '核心模型与方法': ['RUSLE模型', 'GIS技术', '遥感(RS)', '137Cs示踪法', '情景模拟', '改进RUSLE模型', '空间信息技术',
                       '土壤侵蚀控制度'],
    '核心分析要素': ['erosion', '土壤侵蚀强度', '养分流失', '土壤有机碳(SOC)', '全氮(TN)', '全磷(TP)', '全钾(TK)',
                     '土壤有机质(SOM)', '土壤侵蚀量', '土壤可蚀性(K)', '治理潜力', '经济损失'],
    '影响/驱动因子': ['降雨侵蚀力(R)', '坡度坡长(LS)', '植被覆盖与管理(C)', '水土保持措施(P)', '土地利用变化', '城市化',
                      '梯田建设', '过度放牧', '海拔', '坡度', '植被覆盖度'],
    '研究结论/建议': ['侵蚀强度减小', '梯田效益显著', '旱地是主要策源地', '重点防治陡坡地', '侵蚀得到改善',
                      'SOC流失严重', '需加强重点区域治理', '林草措施有效'],
    '文献': ['[陈正发, 2021]', '[陈朝良, 2021]', '[陈红, 2021]', '[陈峰, 2021]', '[林慧龙, 2017]', '[查良松, 2015]',
             '[高海东, 2015]', '[赵明松, 2016]']
}

# 节点颜色
category_colors = {
    '研究区域': '#4A90E2',
    '核心模型与方法': '#50E3C2',
    '核心分析要素': '#F5A623',
    '影响/驱动因子': '#E74C3C',
    '研究结论/建议': '#9B59B6',
    '文献': '#95A5A6'
}

# 添加节点并记录类别
for category, nodes in nodes_by_category.items():
    for node in nodes:
        G.add_node(node, category=category)

# 3. 扩展边（连接更密集，逻辑完整）
edges = [
    # 文献 → 研究区域
    ('[陈正发, 2021]', '云南省', {'label': '聚焦于'}),
    ('[陈朝良, 2021]', '湟水流域', {'label': '聚焦于'}),
    ('[陈红, 2021]', '淮河流域', {'label': '聚焦于'}),
    ('[陈峰, 2021]', '元阳县', {'label': '聚焦于'}),
    ('[林慧龙, 2017]', '三江源', {'label': '聚焦于'}),
    ('[高海东, 2015]', '黄土高原', {'label': '聚焦于'}),
    ('[查良松, 2015]', '巢湖流域', {'label': '聚焦于'}),
    ('[赵明松, 2016]', '云南省', {'label': '聚焦于'}),

    # 更多区域 → RUSLE模型（强化中心辐射）
    ('巢湖流域', 'RUSLE模型', {'label': '采用'}),
    ('拉萨河流域', 'RUSLE模型', {'label': '采用'}),
    ('皇甫川流域', 'RUSLE模型', {'label': '采用'}),
    ('延河流域', 'RUSLE模型', {'label': '采用'}),
    ('南方丘陵山区', 'RUSLE模型', {'label': '采用'}),
    ('长汀县河田盆地区', 'RUSLE模型', {'label': '采用'}),
    ('澜沧江流域', 'RUSLE模型', {'label': '采用'}),

    # RUSLE模型 → 核心分析要素
    ('RUSLE模型', 'erosion', {'label': '计算'}),
    ('RUSLE模型', '土壤侵蚀强度', {'label': '评估'}),
    ('RUSLE模型', '土壤可蚀性(K)', {'label': '包含因子'}),
    ('RUSLE模型', '养分流失', {'label': '评估'}),
    ('RUSLE模型', '治理潜力', {'label': '评估'}),

    # 其他方法支持
    ('GIS技术', '空间信息技术', {'label': '支持'}),
    ('遥感(RS)', '植被覆盖度', {'label': '监测'}),
    ('137Cs示踪法', '土壤侵蚀量', {'label': '验证'}),

    # 研究区域 → 核心分析要素
    ('三江源', '经济损失', {'label': '估算'}),
    ('黄土高原', '治理潜力', {'label': '评估'}),
    ('云南省', '土壤有机碳(SOC)', {'label': '评估'}),

    # 核心分析要素 → 驱动因子（密集连接）
    ('erosion', '降雨侵蚀力(R)', {'label': '受影响'}),
    ('erosion', '坡度坡长(LS)', {'label': '受影响'}),
    ('erosion', '植被覆盖与管理(C)', {'label': '受影响'}),
    ('erosion', '水土保持措施(P)', {'label': '受影响'}),
    ('erosion', '土地利用变化', {'label': '受影响'}),
    ('erosion', '坡度', {'label': '受影响'}),
    ('erosion', '海拔', {'label': '受影响'}),
    ('养分流失', '过度放牧', {'label': '加剧'}),

    # 驱动因子 → 结论/建议
    ('梯田建设', '侵蚀强度减小', {'label': '导致'}),
    ('梯田建设', '梯田效益显著', {'label': '表明'}),
    ('水土保持措施(P)', '侵蚀得到改善', {'label': '促进'}),
    ('林草措施有效', '需加强重点区域治理', {'label': '支持'}),
    ('旱地是主要策源地', '重点防治陡坡地', {'label': '建议'}),
]

for u, v, data in edges:
    G.add_edge(u, v, **data)

# 4. 同心圆布局（中心为 RUSLE模型）
center_node = 'RUSLE模型'
pos = {center_node: (0, 0)}

# 计算节点度，用于分层
degrees = dict(G.degree())
other_nodes = [n for n in G.nodes() if n != center_node]
sorted_nodes = sorted(other_nodes, key=lambda n: degrees[n], reverse=True)

layers = 4  # 4层环
layer_dict = {}
for i, node in enumerate(sorted_nodes):
    layer = min(layers - 1, i // (len(sorted_nodes) // layers + 1))
    layer_dict[node] = layer

# 生成坐标
for node, layer in layer_dict.items():
    radius = 1.2 + layer * 1.4  # 层间距适中
    nodes_in_layer = [n for n, l in layer_dict.items() if l == layer]
    idx = nodes_in_layer.index(node)
    angle = idx * 2 * np.pi / len(nodes_in_layer) + random.uniform(-0.25, 0.25)
    pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

# 5. 准备 Plotly 数据
# 边
edge_x = []
edge_y = []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y,
                        line=dict(width=1.5, color='#888'),
                        hoverinfo='none',
                        mode='lines')

# 节点
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_color = [category_colors[G.nodes[n]['category']] for n in G.nodes()]
node_size = [40 if n == center_node else 22 for n in G.nodes()]
node_text = [f"<b>{n}</b><br>类别：{G.nodes[n]['category']}" for n in G.nodes()]

node_trace = go.Scatter(x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition='middle center',
                        textfont=dict(size=9, color='white'),
                        marker=dict(color=node_color,
                                    size=node_size,
                                    line=dict(width=2, color='black')),
                        hovertemplate='<b>%{text}</b><extra></extra>')

# 6. 创建图形
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='<b>RUSLE模型与土壤侵蚀研究知识图谱（Plotly交互版）</b><br>基于20篇中文文献分析',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=18)
                    ),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=20, r=20, t=100),
                    annotations=[],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    height=900,  # 更大一点更清晰
                    width=1200
                ))

# 7. 添加图例（手动创建分类图例）
for cat, color in category_colors.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=15, color=color),
        name=cat
    ))

fig.update_layout(legend_title_text='节点类别')

# 8. 保存并显示
fig.write_html('RUSLE_知识图谱_Plotly交互版.html')
fig.show()

print(f"总节点数: {G.number_of_nodes()}")
print(f"总边数: {G.number_of_edges()}")
print("已生成交互HTML文件：RUSLE_知识图谱_Plotly交互版.html")