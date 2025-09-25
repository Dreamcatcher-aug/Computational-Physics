import numpy as np
import matplotlib.pyplot as plt

def generate_samples(lambda_input, n):
    xi = np.random.uniform(0, 1, n)
    sample = -np.log(xi) / lambda_input
    return sample

lambda_test= 2   #指数分布为2，可按需要更改
n_test= 10000    #产生随机数数量10000，可按需要更改
samples = generate_samples(lambda_test, n_test)
x = np.linspace(0, 5, 100)
theoretical = lambda_test * np.exp(-lambda_test * x)

plt.figure(figsize=(8, 6))
plt.hist(samples, bins=500, density=True, alpha=0.7, label='Sample Histogram')     #样本分为500个区间，可以按照需要更改
plt.plot(x, theoretical, 'r-', linewidth=2, label=f'f(x)={lambda_test}e^(-{lambda_test}x)')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Verification')
plt.legend()
plt.show()