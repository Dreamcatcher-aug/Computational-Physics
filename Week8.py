import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

#区域离散化
nodes = (np.array
([
    [0.0, 0.0],  [0.25, 0.0], [0.5, 0.0], [0.75, 0.0],  [1.0, 0.0], [0.0, 0.25],
    [0.25, 0.25], [0.5, 0.25], [0.75, 0.25], [0.0, 0.5], [0.25, 0.5], [0.5, 0.5],
    [0.0, 0.75],  [0.25, 0.75],  [0.0, 1.0],
]))

elements = (np.array
([
    [0, 1, 6],   [1, 2, 7],   [2, 3, 7],   [3, 4, 8],   [0, 5, 6],  [1, 6,7 ],
    [3,7,8],  [5,9,10], [5, 6, 10],  [6, 7, 10], [7,10,11], [7,8,11],
    [9,10, 13],   [9,12,13],   [10,11,13],  [12,13,14]
]))


n_nodes = len(nodes)
K = np.zeros((n_nodes, n_nodes))
F = np.zeros(n_nodes)
for e_idx, e in enumerate(elements):
    i, j, k = e
    x_i, y_i = nodes[i]
    x_j, y_j = nodes[j]
    x_k, y_k = nodes[k]
    area = 0.5 * abs((x_j - x_i) * (y_k - y_i) - (x_k - x_i) * (y_j - y_i))
    if area < 1e-10:
        raise ValueError(f"单元{e}（索引{e_idx}）为畸形单元！面积={area:.2e}，节点：{nodes[i]}, {nodes[j]}, {nodes[k]}")

    b_i = y_j - y_k
    b_j = y_k - y_i
    b_k = y_i - y_j
    c_i = x_k - x_j
    c_j = x_i - x_k
    c_k = x_j - x_i

#计算Ke
    k_e = (np.array
    ([
        [b_i * b_i + c_i * c_i, b_i * b_j + c_i * c_j, b_i * b_k + c_i * c_k],
        [b_j * b_i + c_j * c_i, b_j * b_j + c_j * c_j, b_j * b_k + c_j * c_k],
        [b_k * b_i + c_k * c_i, b_k * b_j + c_k * c_j, b_k * b_k + c_k * c_k]
    ]) / (4 * area))

    K[i, i] += k_e[0, 0]
    K[i, j] += k_e[0, 1]
    K[i, k] += k_e[0, 2]
    K[j, i] += k_e[1, 0]
    K[j, j] += k_e[1, 1]
    K[j, k] += k_e[1, 2]
    K[k, i] += k_e[2, 0]
    K[k, j] += k_e[2, 1]
    K[k, k] += k_e[2, 2]


boundary_conditions = {}
for node in [0, 5, 9, 12]:
    boundary_conditions[node] = 1.0
for node in [1, 2, 3]:
    boundary_conditions[node] = 0.0
for node in [4, 8, 11, 13, 14]:
    boundary_conditions[node] = 1.0
for node_idx, phi_val in boundary_conditions.items():
    K[node_idx, :] = 0.0
    K[node_idx, node_idx] = 1.0
    F[node_idx] = phi_val

phi = np.linalg.solve(K, F)
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
tri = Triangulation(nodes[:, 0], nodes[:, 1], elements)
tripcolor = plt.tripcolor(tri, phi, shading='gouraud', cmap='viridis')
plt.colorbar(tripcolor, label='Electric potential φ(x,y)')
plt.triplot(tri, color='black', linewidth=0.2)  # 绘制单元边界
plt.scatter(nodes[:, 0], nodes[:, 1], color='red', s=15, zorder=5)  # 标记节点
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('result', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
