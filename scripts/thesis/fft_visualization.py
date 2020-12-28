''' 
Fast Fourier Transform visualization

Based on the code:
Taspinar, A (2018) Machine Learning with Signal Processing techniques 
[Source code] http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
Licence: MIT
'''

import numpy as np
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

font = {
    'size': 15
}
plt.rc('font', **font)

def get_fft_values(y_values, T, N):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    return f_values, fft_values

def generate_3D_plot(x_values, y_values, f_values, fft_values,
                     composite_y_value, frequencies):
    colors = ['g'] + ['gold', 'darkkhaki', 'khaki']
    labels = ['original signal'] + [
        f'{freq}Hz component' for freq in list(reversed(frequencies))
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("\nTime [s]", fontsize=16)
    ax.set_ylabel("\nFrequency [Hz]", fontsize=16)
    ax.set_zlabel("\nAmplitude", fontsize=16)

    y_values_list = [composite_y_value] + list(reversed(y_values))
    frequencies = [0] + list(reversed(frequencies))

    for idx in range(len(y_values_list)):
        signal = y_values_list[idx]
        length = signal.shape[0]
        x = x_values
        y = np.array([frequencies[idx]] * length)
        z = signal

        if idx == 0:
            linewidth = 4
        else:
            linewidth = 2
        ax.plot(list(x),
                list(y),
                zs=list(z),
                linewidth=linewidth,
                color=colors[idx],
                label=labels[idx])

    x = [10] * 30
    y = f_values[:30]
    z = fft_values[:30] * 3
    ax.plot(list(x),
            list(y),
            zs=list(z),
            linewidth=2,
            color='red',
            label='FFT spectrum')

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    signal_len_sec = 10
    num_points = 1000
    T = signal_len_sec / num_points
    amplitudes = [0.5, 1, 2]
    frequencies = [2.5, 1.5, 1]

    x_values = np.linspace(0, signal_len_sec, num_points)
    y_values = [
        amplitudes[idx] * np.sin(2 * np.pi * frequencies[idx] * x_values)
        for idx in range(0, len(amplitudes))
    ]
    # sum all sine waves
    composite_y_value = np.sum(y_values, axis=0)

    f_values, fft_values = get_fft_values(composite_y_value, T, num_points)

    generate_3D_plot(x_values, y_values, f_values, fft_values,
                     composite_y_value, frequencies)
