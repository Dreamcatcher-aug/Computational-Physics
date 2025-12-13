import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

def generate_spin_samples(L, sample_size):
    return np.random.choice([-1, 1], size=(sample_size, L))

def compute_matrix_elements(samples):
    sample_size, L = samples.shape
    matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            matrix[i, j] = np.mean(samples[:, i] * samples[:, j])
    return matrix

L = 10
sample_sizes = [50, 200, 1000, 5000]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, size in enumerate(sample_sizes):
    samples = generate_spin_samples(L, size)
    mat = compute_matrix_elements(samples)
    sns.heatmap(mat, ax=axes[idx], cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f")
    axes[idx].set_title(f"Sample Size = {size}")
    axes[idx].set_xlabel("Spin Index")
    axes[idx].set_ylabel("Spin Index")

plt.tight_layout()
plt.show()

def compute_ground_truth_energy(samples, J=1):
    sample_size, L = samples.shape
    energies = []
    for s in samples:
        energy = -J * np.sum(s * np.roll(s, -1))
        energies.append(energy)
    return np.array(energies)

L = 10
sample_size = 1000
samples = generate_spin_samples(L, sample_size)
y_true = compute_ground_truth_energy(samples)
X = samples

models =  \
{
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X, y_true)
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    results[name] = {"mse": mse, "coefficients": model.coef_}

for name, res in results.items():
    print(f"{name}: 均方误差 = {res['mse']:.4f}")

print("三个拟合的均方误差都很大，是因为待拟合模型为一阶单自旋线性模型，仅能捕捉单个自旋的独立贡献，"
      "而真实模型是二阶自旋耦合模型，能量由相邻自旋的关联作用决定，二者函数形式不匹配。")