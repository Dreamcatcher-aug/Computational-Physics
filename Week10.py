import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#初始化条件
L = 10.0
N = 64
m = 1.0
dt = 0.02
steps = 200
sigma = 1.0
epsilon = 1.0
np.random.seed(42)
pos = np.zeros((N, 3))
grid_size = int(np.ceil(N**(1/3)))
index = 0
for i in range(grid_size):
    for j in range(grid_size):
        for k in range(grid_size):
            if index < N:
                pos[index] = [
                    i * L / grid_size + np.random.uniform(-0.1, 0.1),
                    j * L / grid_size + np.random.uniform(-0.1, 0.1),
                    k * L / grid_size + np.random.uniform(-0.1, 0.1)
                ]
                index += 1
vel = np.random.uniform(-1.0, 1.0, (N, 3))
acc = np.zeros((N, 3))

#定义 Lennard-Jones 势的力计算函数
def compute_forces(pos, L, sigma, epsilon):
    forces = np.zeros((N, 3))
    for i in range(N):
        for j in range(i + 1, N):
            r = pos[i] - pos[j]
            r = r - np.round(r / L) * L
            dist = np.linalg.norm(r)
            if dist < 1e-10:
                continue
            f = 24 * epsilon / dist * (2 * (sigma / dist)**12 - (sigma / dist)**6)
            forces[i] += f * r / dist
            forces[j] -= f * r / dist
    return forces


final_pos = np.zeros((N, 3))
final_vel = np.zeros((N, 3))

# 初始力计算
forces = compute_forces(pos, L, sigma, epsilon)
acc = forces / m

for step in range(steps):
    pos += vel * dt + 0.5 * acc * dt ** 2
    pos = pos % L
    new_forces = compute_forces(pos, L, sigma, epsilon)
    new_acc = new_forces / m
    vel += 0.5 * (acc + new_acc) * dt
    acc = new_acc
    if step == steps - 1:
        final_pos = pos.copy()
        final_vel = vel.copy()
speed = np.linalg.norm(final_vel, axis=1)

# 绘制速度分布直方图
plt.figure(figsize=(8, 6))
plt.hist(speed, bins=15, density=True, alpha=0.7, color='blue')
plt.xlabel('Speed')
plt.ylabel('Probability Density')
plt.title('Speed Distribution of Atoms (After {} Steps)'.format(steps))
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 绘制位置散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], s=30, c='red', alpha=0.6)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Position Distribution of Atoms (After {} Steps)'.format(steps))
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
plt.show()