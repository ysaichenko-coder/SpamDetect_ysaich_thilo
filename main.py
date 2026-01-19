import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from data_load import DataLoad

path = './data'
csv_path = os.path.join(path, 'spam.csv')
loader = DataLoad()
X_train_features, X_test_features, Y_train, Y_test, feature_extraction = loader.load_data()
print(Y_test)
model = SVC(kernel='sigmoid', C=1.0)
model.fit(X_train_features, Y_train)
prediction = model.predict(X_test_features)
accuracy = accuracy_score(Y_test, prediction)
print('accuracy on test',accuracy*100,'%')
joblib.dump(model,'model.joblib')
input = ["Hello mom, are we having dinner tonight?",
    "URGENT! You have won a FREE iPhone. Click here",
    "Call me back later",
    "Congratulations, you won 1000$ prize money",
    "Hey bro what's up"]
input_features = feature_extraction.transform(input)
prediction = model.predict(input_features)
print(prediction)