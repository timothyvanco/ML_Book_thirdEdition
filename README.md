## MACHINE LEARNING
I am coding examples from a book - Python Machine Learning third editin

# PERCEPTRON
Basic code for Perceptron from scratch.
This is how basic Perceptron looks like:

![single_layer_perceptron](perceptron/single_layer_perceptron.png)

As a data I used IRIS dataset. I divide data based on labels 'setosa' amd 'cersicolor'.
Then perceptron learn - adjust his weights, to divide these 2 classes.

#### Result:

Perceptron with Scikit-learn library on Iris flower dataset:

![perceptron_skicit-learn](perceptron_skicit-learn/plot.jpg) 

# SVM

Powerful and widely used learning algorithm is the support vector machine (SVM), 
which can be considered an extension of the perceptron. Using the perceptron algorithm, 
we minimized misclassification errors. However, in SVMs our optimization objective is to maximize the margin. 
The margin is defined as the distance between the separating hyperplane (decision boundary) and the training examples 
that are closest to this hyperplane, which are the so-called support vectors. This is illustrated in the following figure:

![svm](SVM/svm.png) 

#### Kernel trick

How to visualize and model non-linear data with SVM?

1 - transform it to higher dimension

2 - train linear SVM model to classify data in a new feature space

In my code, I tried it with this type of dataset and the result:

![kerneltrick](SVM/kerneltrick.jpeg)

#### GAMMA - cut off parameter
Gamma is used to control overfitting.

svm = SVC(kernel='rbf', random_state=1, gamma=xxx, C=1.0)

Bigger gammga = possible overfitting

GAMMA = 0.2 | GAMMA = 100

![svmgamma](SVM/svmgamma.jpeg)

# DECISION TREE

A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, 
including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only 
contains conditional control statements.

![decision_tree_theory](decision_tree/decision_tree_theory.png)

Decision trees are commonly used in operations research, specifically in decision analysis, 
to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.

I used Decision Tree model on Iris flower dataset:

![tree](decision_tree/tree.png)

# RANDOM FOREST

Random forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.

Random forest of 25 decision trees:

![random_forest](decision_tree/random_forest_25DT.jpg)


# KNN - K-Nearest neighbors
KNN is a typical example of a lazy learner. 
It is called "lazy" not because of its apparent simplicity, 
but because it doesn't learn a discriminative function from 
the training data but memorizes the training dataset instead

![KNN_theory](KNN/KNN_theory.png)

I used KNN on Iris flower dataset:

![KNN](KNN/knn.jpg)

# DATA PRE-PROCESSING

The quality of the data and the amount of useful information that it contains are key factors 
that determine how well a machine learning algorithm can learn. 
Therefore, it is absolutely critical to ensure that we examine and preprocess a dataset 
before we feed it to a learning algorithm. In this file, 
I was working on the essential data preprocessing techniques that will help to build good machine learning models.

Main topics in data pre-processing:
- Removing and imputing missing values from the dataset
- Getting categorical data into shape for machine learning algorithms
- Selecting relevant features for the model construction

I was working with wine dataset. This is picture of Feature importance in that dataset, 
which can help in getting rid of some part of data in future - make dataset smaller.

![feature_importance_wines](data_preprocessing/feature_importance_wines.jpg)

# COMPRESSING DATA via DIMENSIONALITY REDUCTION

I learned about three different, fundamental dimensionality reduction techniques 
for feature extraction: standard PCA, LDA, and KPCA. 

- Using PCA, I projected data onto a lower-dimensional subspace to maximize the variance along the orthogonal feature axes, 
while ignoring the class labels. 

- LDA, in contrast to PCA, is a technique for supervised dimensionality reduction, 
which means that it considers class information in the training dataset to attempt to maximize the 
class-separability in a linear feature space

- Lastly nonlinear feature extractor, KPCA. 
Using the kernel trick and a temporary projection into a higher-dimensional feature space, 
I was ultimately able to compress datasets consisting of nonlinear features onto a lower-dimensional 
subspace where the classes became linearly separable.

# MODEL EVALUATION & HYPERPARAMETER TUNIG

My code in this example is all based on Breast Cancer Wisconsin dataset.
(https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic) )

#### Pipeline

I worked with Pipeline -  fit a model including an arbitrary number of 
transformation steps and apply it to make predictions about new data

#### holdout method

Split the dataste into 3 categories:

- Train dataset
- Validation dataset
- Test dataset

It is very good to have test data which hasn't been seen yet! Model will be good on predicting in real world on new data.

#### K-fold cross-validation

In k-fold cross-validation, we randomly split the training dataset into k folds without replacement, where k – 1 folds are 
used for the model training, and one fold is used for performance evaluation. This procedure is repeated k 
times so that we obtain k models and performance estimates

-> then calculate average performace

-> after finding satisfactory hyperparameters values => retrain model on complete training dataset -> obtain final performance

-> advantage => each example will be used for training & for validation exactly once

![kfoldcrossvalidation](Model_evaluation/kfoldcrossvalidation.png)

#### Grid search

it's a brute-force exhaustive search paradigm where we specify a list of values for different 
hyperparameters, and the computer evaluates the model performance for each combination 
to obtain the optimal combination of values from this set

#### Nested cross-validation

![nestedcrossvalidation](Model_evaluation/nestedcrossvalidation.png)

#### Confusion matrix

A confusion matrix is simply a square matrix that reports the counts of the true positive (TP), true negative (TN), 
false positive (FP), and false negative (FN) predictions of a classifier, as shown in the following figure

![confusionmatrix](Model_evaluation/confusionmatrix.png)

### ROC AUC
Receiver operating characteristic (ROC) graphs are useful tools to select models for 
classification based on their performance with respect to the FPR and TPR, which are computed by shifting 
the decision threshold of the classifier. The diagonal of
a ROC graph can be interpreted as random guessing, and classification models that fall 
below the diagonal are considered as worse than random guessing. A perfect classifier would 
fall into the top-left corner of the graph with a TPR of 1 and an FPR of 0. Based on the ROC curve, 
we can then compute the so-called ROC area under the curve (ROC AUC) to characterize the performance 
of a classification model

### CLASS IMBALANCE

When class1 has 80% and class2 has 20% of dataset

SOLVE class imbalance:
- assign a larger penalty to wrong predictions on the minority class
- upsampling the minority class, downsampling the majority class
- generation of synthetic training examples

