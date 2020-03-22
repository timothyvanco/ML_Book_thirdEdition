from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# return 3 class labels - Iris-setosa (0), Iris-versicolor (1), Iris-virginica (2)
print("Class Labels: ", np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print("Labels counts in y: ", np.bincount(y))
print("Labels counts in y_train: ", np.bincount(y_train))
print("Labels counts in y_test: ", np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

"""
StandardScaler.fit() estimated the parameters, 
𝜇 (sample mean) and 𝜎 (standard deviation), 
for each feature dimension from the training data

calling transform function we standardized the training 
data using those estimated parameters, 𝜇 and 𝜎

after standardization -> train perceptron
"""
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
# perceptron misclassifies 1 out of the 45 flower examples. The misclassification
# error on the test dataset is approximately 2.2 percent
# Classification accuracy : 1 - error = 1 - 0.022 = 0.978 = 97.8%
print('Misclassified examples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
# y_test - true classes
# y_pred - predicted classes
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# prediction accuracy by combining the predict call with accuracy_score
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))