## MACHINE LEARNING
I am coding examples from a book - Python Machine Learning third editin

# PERCEPTRON
Basic code for Perceptron from scratch.
This is how basic Perceptron looks like:

![single_layer_perceptron](perceptron/single_layer_perceptron.png)

As a data I used IRIS dataset. I divide data based on labels 'setosa' amd 'cersicolor'.
Then perceptron learn - adjust his weights, to divide these 2 classes.

#### Result:

![Sepal_Petal_length](perceptron/Sepal_Petal_length.png)

Perceptron with Scikit-learn library on Iris flower dataset:

![perceptron_skicit-learn](perceptron_skicit-learn/plot.jpg) 

# SVM

Powerful and widely used learning algorithm is the support vector machine (SVM), 
which can be considered an extension of the perceptron. Using the perceptron algorithm, 
we minimized misclassification errors. However, in SVMs our optimization objective is to maximize the margin. 
The margin is defined as the distance between the separating hyperplane (decision boundary) and the training examples 
that are closest to this hyperplane, which are the so-called support vectors. This is illustrated in the following figure:

![svm](SVM/svm.png) 
