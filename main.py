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
X_train, X_test, Y_train, Y_test = train_test_split(data['v2'],data['v1'], test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
print(data['v1'])
print(Y_test)
model = LogisticRegression()
model.fit(X_train_features, Y_train)
prediction = model.predict(X_test_features)
accuracy = accuracy_score(Y_test, prediction)
print('accuracy on test',accuracy*100,'%')
input = ["Hello mom, are we having dinner tonight?", 
    "URGENT! You have won a FREE iPhone. Click here", 
    "Call me back later",                             
    "Congratulations, you won 1000$ prize money",     
    "Hey bro what's up"]
input_features = feature_extraction.transform(input)
prediction = model.predict(input_features)
print(prediction)