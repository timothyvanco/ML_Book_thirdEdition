import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pandas as pd

class LogisticRegressionGD(object):
    """ Logistic Regression Classifier using gradient descent

    Parameters
    eta : float
        Leraning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset
    random_state : init
        Random number generator seed for random weight initialization

    Attributes
    w_ : 1d-array
        weights after fitting
    cost_ : list
        Logistic cost function value in each epoch
    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data, labels):
        """ Fit training data
        Parameters
        Data : { array-like }, shape = [n-examples, n_features]
                Training vectors, where n_examples is the number of examples and
                n_features is the number of features
        Labels : array-like, shape = [n_examples]
                Target values

        Return
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + data.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(data)
            output = self.activation(net_input)
            errors = (y - output)

            self.w_[1:] += self.eta * data.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # note that compute the logistic 'cost' now instead of the sum of squared errors cost
            cost = (-labels.dot(np.log(output)) - ((1 - labels).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, data):
        """ Calculate net input """
        return np.dot(data, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """ Compute logistic sigmoid activation """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, data):
        """ return class label after unit step """
        return np.where(self.net_input(data) >= 0.0, 1, 0)


def plot_decision_regions(data, labels, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    # plot decision surface
    data1_min, data1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    data2_min, data2_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    datax1, datax2 = np.meshgrid(np.arange(data1_min, data1_max, resolution),
                           np.arange(data2_min, data2_max, resolution))

    Z = classifier.predict(np.array([datax1.ravel(), datax2.ravel()]).T)
    Z = Z.reshape(datax1.shape)

    plt.contourf(datax1, datax2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(datax1.min(), datax1.max())
    plt.ylim(datax2.min(), datax2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')



s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

# select setosa and versicolor
labels = df.iloc[0:100, 4].values
labels = np.where(labels == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
data = df.iloc[0:100, [0, 2]].values

# APPLING STANDARDIZATION
data_std = np.copy(data)
data_std[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
data_std[:, 1] = (data[:, 1] - data[:, 1].mean()) / data[:, 1].std()


### CODE IS NOT FINISHED!