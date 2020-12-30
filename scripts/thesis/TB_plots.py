import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

font = {
    'size': 15
}
plt.rc('font', **font)

def generate_plot(train, val):
    plt.figure()
    plt.plot(train, label='train', c='r')
    plt.plot(val, label='validation', c='g')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Binary cross-entropy loss')
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    loss_train = pd.read_csv('../../results/loss_train_TB.csv', header=None)
    loss_val = pd.read_csv('../../results/loss_val_TB.csv', header=None)

    generate_plot(loss_train[2], loss_val[2])
