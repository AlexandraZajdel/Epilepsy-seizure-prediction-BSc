'''Visualize ReLU and sigmoid activation functions. (Figure 3.3 and 3.4).'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=1.2)  

def ReLU(z):
    return max(z,0)

def sigmoid(z):
    return 1/(1+ np.exp(-z))

def plot_activ_funcs(x, y, label):
    plt.figure(figsize=(5,3))
    plt.plot(x, y, label=label)
    plt.xlabel('input value')
    plt.ylabel('output value')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == '__main__':
    x = np.arange(-10, 10, step=0.1)
    y_relu = [ReLU(val) for val in x]
    y_sigm = [sigmoid(val) for val in x]

    plot_activ_funcs(x, y_relu, r'$f(z) = max(z, 0)$')
    plot_activ_funcs(x, y_sigm, r'$f(z) = \frac{1}{1 + exp(-z)}$')
