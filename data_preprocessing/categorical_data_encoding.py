# CATEGORICAL DATA ENCODING with pandas
import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']
                   ])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)


# we have to manually define values to ordinal features
size_mapping = {'XL': 3,
                'L' : 2,
                'M' : 1}

df['size'] = df['size'].map(size_mapping)   # change sizes of T-shirts to numbers
print(df)

import numpy as np
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
#print(class_mapping)

# use the mapping dictionary to transform the class labels into integers
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# other option is with LabelEncoder from scikit-learn
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)  # fit_transform - shortcut for calling fit & transform separately
#print(y)


# ONE HOT ENCODING on Nominal features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = class_le.fit_transform(X[:, 0])
print(X) # blue = 0, green = 1, red = 2 - but it doesnt mean red > green > blue!! => ONE HOT ENCODING

from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())    # apply OneHotEncoding just to only single column

# selectively transform columms
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
print(c_transf.fit_transform(X).astype(float))
"""
colors - size - price
[0.0, 1.0, 0.0, 1, 10.1],
[0.0, 0.0, 1.0, 2, 13.5],
[1.0, 0.0, 0.0, 3, 15.3]
"""

# best and easiest option - from pandas - get_dummies - only convert string columns
print(pd.get_dummies(df[['price', 'color', 'size']]))