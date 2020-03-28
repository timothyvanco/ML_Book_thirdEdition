Powerful and widely used learning algorithm is the support vector machine (SVM), 
which can be considered an extension of the perceptron. Using the perceptron algorithm, 
we minimized misclassification errors. However, in SVMs our optimization objective is to maximize the margin. 
The margin is defined as the distance between the separating hyperplane (decision boundary) and the training examples 
that are closest to this hyperplane, which are the so-called support vectors. This is illustrated in the following figure:

![svm](svm.png) 

### result on Iris flower dataset

![svm_plot](plot.jpg)



## KERNEL TRICK
How to visualize and model non-linear data with SVM?

1 - transform it to higher dimension

2 - train linear SVM model to classify data in a new feature space

![kernel_trick](kernel_trick.png)

In my code, I tried it with this type of dataset:

![dataset_kernel_trick](datasetSVM.png)

and the result is here:

![kerneltrickSVM](kerneltrickSVM.png)

## GAMMA - cut off parameter
Gamma is used to control overfitting.

Bigger gammga = possible overfitting

GAMMA = 0.2 

![SVM_gamma_02](SVM_gamma_02.png)

GAMMA = 100

![SVM_gamma_100](SVM_gamma_100.png)
