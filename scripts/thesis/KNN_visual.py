'''K-nearest neighbors classification and elbow method visualization. 
(Figure 2.4 and 2.5).
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

plt.style.use('seaborn')

def generate_dataset():
    X, y = make_classification(n_samples=2000, n_features=20, n_redundant=8, 
                                n_informative=4,
                                n_clusters_per_class=1, n_classes=2)    
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    return X, y

def generate_mesh(X):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    return xx, yy

def generate_plot(X_train, X_test, y_train, y_test, xx, yy):
    figure = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1 = plot_dataset(ax1, X_train, X_test, y_train, y_test, xx, yy)
    ax2 = plt.subplot(1, 2, 2)
    ax2 = plot_decision_boundary(ax2, X_train, X_test, y_train, y_test, xx, yy)

    plt.tight_layout()
    plt.show()

def plot_dataset(ax, X_train, X_test, y_train, y_test, xx, yy):
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return ax

def plot_decision_boundary(ax, X_train, X_test, y_train, y_test, xx, yy):
    clf, _ = classify(K_NEIGHBORS, X_train, y_train, X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    cm_bright = ListedColormap(['#99ccff', '#ff9999'])
    cm_dark = ListedColormap(['#FF0000', '#0000FF'])

    ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_dark,
                edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_dark,
                edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return ax

def classify(K, X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(K)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error = np.mean(y_pred != y_test)
    return clf, error

def optim_K(X_train, X_test, y_train, y_test):
    K_lst = np.arange(1, 60)
    error_buffer = []
    for K in K_lst:
        _, error = classify(K, X_train, y_train, X_test, y_test)
        error_buffer.append(error)

    fig = plt.figure(figsize=(8,6))
    plt.plot(K_lst, error_buffer, color='b', linestyle='dashed', marker='o',
            markerfacecolor='r', markersize=5)
    plt.xlabel('K value')
    plt.ylabel('error rate')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    K_NEIGHBORS = 15

    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    xx, yy = generate_mesh(X)
    generate_plot(X_train, X_test, y_train, y_test, xx, yy)

    optim_K(X_train, X_test, y_train, y_test)







