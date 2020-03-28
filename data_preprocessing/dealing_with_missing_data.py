import pandas as pd
from io import StringIO  # to read string as csv_data

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# df = Data Frame from pandas
df = pd.read_csv(StringIO(csv_data))
print(df)
"""
    A       B       C       D
0   1.0     2.0     3.0     4.0
1   5.0     6.0     NaN     8.0
2   10.0    11.0    12.0    NaN
"""

# find out if cell contains NaN value, with sum() - count number of missing values per column
print(df.isnull().sum())

# Eliminate training examples/features with missing values
# drop out whole column or row where is something missing
print(df.dropna(axis=0)) # output is only rows where are all features = drop rows
print(df.dropna(axis=1)) # output is only columns where are all features = drop columns

print(df.dropna(how='all'))     # drop rows where all columns are NaN
print(df.dropna(thresh=4))      # drop rows that have fewer then 4 real values
print(df.dropna(subset=['C']))  # drop rows where NaN apper in specific column - C



from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

# same with pandas
print(df.fillna(df.mean()))