import numpy as np
import matplotlib.pyplot as plt

#目标分布函数
def target_distribution(x):
    return np.exp(-x ** 2/2)

#Metropolis方法的核心函数
def metropolis(n_samples, delta):
    samples = np.zeros(n_samples)
    current_x = 0
    accept_count = 0

    for i in range(n_samples):
        proposed_x = current_x + delta * (np.random.rand() - 0.5) * 2     #在x+delta and x-delta 生成试探点
        r = target_distribution(proposed_x) / target_distribution(current_x)    #生成 try/n 的数值
        prob_accept = min(1, r)     #判断是否满足试探接纳条件
        if np.random.rand() < prob_accept:
            current_x = proposed_x
            accept_count += 1
        samples[i] = current_x

    accept_ratio = accept_count / n_samples    #接受点与试探点步数之比
    return samples, accept_ratio

#测试不同的最大试探步长 delta
delta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
n_samples = 10000
results = {}

for delta in delta_values:
    samples, accept_ratio = metropolis(n_samples, delta)
    x_squared_mean = np.mean(samples ** 2)
    results[delta] = (samples, accept_ratio, x_squared_mean)

#绘图部分
for i, delta in enumerate(delta_values):
    samples, accept_ratio, x_squared_mean = results[delta]
    plt.subplot(2, 3, i + 1)
    plt.hist(samples, bins=50, density=True, alpha=0.6, label=f'all samples')
    x_theory = np.linspace(-5, 5, 1000)
    pdf_theory = np.exp(-x_theory ** 2 / 2) / np.sqrt(2 * np.pi)
    plt.plot(x_theory, pdf_theory, 'r-', linewidth=2, label='gauss')
    plt.title(f'delta={delta}\naccept_ratios={accept_ratio:.2f} | all_samples<x²>={x_squared_mean:.4f}', fontsize=10)
    plt.legend()
    plt.xlim(-5, 5)

plt.tight_layout()
plt.show()

# 分析平衡时间与 delta 的关系以及合理的 delta
print("分析不同 delta 的情况：")
for delta in delta_values:
    samples, accept_ratio, x_squared_mean = results[delta]
    print(f"delta = {delta}:")
    print(f"  接受率 = {accept_ratio:.2f}")
    print(f"  <x²> = {x_squared_mean:.4f} (目标值约为 1)")
    x_squared_early = np.mean(samples[:1000] ** 2)
    print(f"  前 1000 个样本的 <x²> = {x_squared_early:.4f}，可大致判断平衡时间")