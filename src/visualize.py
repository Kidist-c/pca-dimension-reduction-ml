import matplotlib.pyplot as plt

def plot_pca(X_pca, y):
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization")
    plt.show()