import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(font="Times New Roman", font_scale=1.5)
plt.rcParams["font.family"] = "Times New Roman"

data = pd.read_csv('../../results/KNN_results/KNN_all_results.csv', 
                   header=0, 
                   index_col=0)

plt.figure(figsize=(9,3))
ax = sns.heatmap(data, annot=True, cbar=False, cmap='Reds')
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(left=False, top=False)
plt.tight_layout(pad=0)
plt.show()