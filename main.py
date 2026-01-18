import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = './data'
csv_path = os.path.join(path, 'spam.csv')
data = pd.read_csv(csv_path,encoding='latin-1')
data = data[['v1', 'v2']]
data.loc[data['v1'] == 'spam', 'v1',] = 0
data.loc[data['v1'] == 'ham', 'v1',] = 1
X_train, X_test, Y_train, Y_test = train_test_split(data['v2'],data['v1'], test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(data['v1'])
print(Y_test)