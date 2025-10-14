import numpy as np
import matplotlib.pyplot as plt

N_list = [1000,5000,10000,20000,50000,100000]
true_value = 3.32335  #真实积分值
estimates = []
errors = []

for N in N_list:
    x_samples = np.random.exponential(scale=1, size=N)  #g(x)=exp(-x)
    h = x_samples ** (5/2)                              #h(x)=x^2.5
    I_estimate = np.mean(h)                             #重要抽样法核心：I≈E[h(x)]
    estimates.append(I_estimate)
    error = abs(I_estimate - true_value)
    errors.append(error)
    print(f"投点数N={N}，积分估计值={I_estimate:.6f}，误差={error:.4f}")

print("说明误差随着投点数的增多而减少")