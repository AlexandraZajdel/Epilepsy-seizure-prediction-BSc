'''Generate examples of ROC curves (Figure 4.1).'''

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set(font_scale=1.3)  

def plot_ROC(fpr=None, tpr=None):
    plt.figure(figsize=(9,7), dpi=600)
    lw = 2
    # perfect ROC
    plt.plot([0, 0, 1], [0, 1, 1],
             color='green',
             lw=lw,
             label='perfect classifier (ROC AUC = 1.0)')
    # random ROC 
    plt.plot([0, 1], [0, 1],
             color='navy',
             lw=lw,
             linestyle='--',
             label='random classifier (ROC AUC = 0.5)')
    # others 
    if fpr and tpr is not None:
        plt.plot(fpr, tpr, color='yellow', lw=lw)

    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_ROC()
