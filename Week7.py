import numpy as np
import matplotlib.pyplot as plt

# 参数
Lx = np.pi
Ly = np.pi
Nx = 90
Ny = 90
hx = Lx / Nx
hy = Ly / Ny
omega = 7/4
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')


phi = np.zeros((Nx+1, Ny+1))
phi[0, :] = 0.0  # 左边界 x=0：φ(0, y)=0
phi[Nx, :] = 0.0  # 右边界 x=π：φ(π, y)=0
phi[:, 0] = 0.0  # 下边界 y=0：φ(x, 0)=0
phi[:, Ny] = np.sin(x) #上边界x=π：φ(π, y)=sin(x)
Ax = 1.0 / hx**2
Ay = 1.0 / hy**2
A0 = 2.0 * (Ax + Ay)

# SOR迭代
max_iter = 10000
tol = 1e-6
for it in range(max_iter):
    max_diff = 0.0
    for i in range(1, Nx):
        for j in range(1, Ny):
            phi_new = ((Ax * (phi[i+1, j] + phi[i-1, j]) +Ay * (phi[i, j+1] + phi[i, j-1])) / A0)
            phi[i, j] = (1 - omega) * phi[i, j] + omega * phi_new
            max_diff = max(max_diff, abs(phi[i, j] - phi_new))
    if max_diff < tol:
        print(f"SOR 收敛于 {it} 步")
        break

# 精确解
phi_exact = np.sin(X) * np.sinh(Y) / np.sinh(Lx)
error = np.abs(phi - phi_exact)
print("最大绝对误差:", np.max(error))

#绘图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("SOR numerical")
plt.contourf(X, Y, phi, 50, cmap='viridis')
plt.colorbar()
plt.subplot(1,2,2)
plt.title("error")
plt.contourf(X, Y, error, 50, cmap='inferno')
plt.colorbar()
plt.show()
