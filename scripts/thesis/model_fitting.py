'''Illustrate underfitting, overfitting and good fitting on synthetic data. 
(Figure 2.10)
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=1.5)  

def generate_data():
    np.random.seed(seed=42)
    n_points = 20

    x = np.random.uniform(0, 11, n_points)
    x = np.sort(x)
    y = (-x+2) * (x-9) + np.random.normal(0, 3, n_points)

    return x,y

def fit(x, y, degree):
    poly_coeff = np.polyfit(x, y, degree)
    return poly_coeff

def generate_plot(x, y):
    x_continuous = np.linspace(0, max(x), num=300)

    plt.figure()
    plt.scatter(x, y, label='training examples', c='k')
    plt.plot(x_continuous, np.polyval(fit(x, y, 2), x_continuous),
            label='good model (deg=2)', c='g')
    plt.plot(x_continuous, np.polyval(fit(x, y, 1), x_continuous),
            label='underfit model (deg=1)', c='b')
    plt.plot(x_continuous, np.polyval(fit(x, y, 14), x_continuous),
            label='overfit model (deg=14)', c='r')
    plt.ylim([-20, 20])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == '__main__':
    x, y = generate_data()
    generate_plot(x, y)