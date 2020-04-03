# Regression analysis

Regression models are used to predict target variables on a continuous scale, which makes 
them attractive for addressing many questions in science. They also have applications in industry, 
such as understanding relationships between variables, evaluating trends, or making forecasts. 

One example is predicting the sales of a company in future months.
In this chapter I worked on main concepts of regression models and cover the following topics:
- Exploring and visualizing datasets
- Looking at different approaches to implement linear regression models
- Training regression models that are robust to outliers
- Evaluating regression models and diagnosing common problems
- Fitting regression models to nonlinear data

In this Chapter I am working with Housing dataset which can be downloaded from - 

https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/data/boston_house_prices.csv

## Linear Regression

The goal of linear regression is to model the relationship between 
one or multiple features and a continuous target variable. In contrast to 
classification—a different subcategory of supervised learning—regression analysis aims 
to predict outputs on a continuous scale rather than categorical class labels

### Simple linear regression

The goal of simple (univariate) linear regression is to model the 
relationship between a single feature (explanatory variable, x) and 
a continuous-valued target (response variable, y).

y = w0 + w1 * x 

(y = ax + b)

Goal is to learn the weights (w0, w1)

linear regression can be understood as finding the best-fitting straight line through the training examples
This best-fitting line is also called the regression line, and the vertical lines from the regression line to 
the training examples are the so-called offsets or residuals—the errors of our prediction

### Multiple linear regression

y = w0x0 + w1x1 +... = wT x

