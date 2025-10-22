import numpy as np
import matplotlib.pyplot as plt

#理论分布
def target_distribution(v, alpha):
    C=4*np.sqrt(alpha**3 / np.pi)  # 计算得到归一化常数
    return C * v**2 * np.exp(-alpha * v**2)   #形成目标分布函数

#Metropolis方法
def metropolis_sampling(alpha, n_samples, step_size=0.5):
    samples = np.zeros(n_samples)  #用于存储采样结果
    current_v = np.random.uniform(0, 2)  # 初始点（随机选）
    for i in range(n_samples):
        proposed_v = current_v + np.random.normal(0, step_size)
        if proposed_v < 0:
            proposed_v = current_v
        f_current = target_distribution(current_v, alpha)
        f_proposed = target_distribution(proposed_v, alpha)
        acceptance = min(1, f_proposed / f_current)
        if np.random.rand() < acceptance:
            current_v = proposed_v
        samples[i] = current_v
    return samples

# 设定参数并采样，可根据需要更改
alpha = 1.0
n_samples = 1000000
samples = metropolis_sampling(alpha, n_samples)

# 绘制采样结果的直方图与理论曲线
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=100, density=True, alpha=0.6, label='Sampled Distribution')
v_range = np.linspace(0, 5, 1000)
theoretical = target_distribution(v_range, alpha)
plt.plot(v_range, theoretical, 'r-', linewidth=2, label='Theoretical Distribution')
plt.xlabel('Velocity')
plt.ylabel('Probability Density')
plt.title('Metropolis Sampling of Fermion Velocity Distribution $f(v) = Cv^2e^{-v^2}$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()