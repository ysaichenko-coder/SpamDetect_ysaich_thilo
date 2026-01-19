import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data_load import DataLoad

path = './data'
csv_path = os.path.join(path, 'spam.csv')
loader = DataLoad()
X_train_features, X_test_features, Y_train, Y_test, feature_extraction = loader.load_data()
print(Y_test)
modelSVC = SVC(kernel='sigmoid', C=1.0)
modelLR = LogisticRegression()
for i in [modelSVC, modelLR]:
    i.fit(X_train_features, Y_train)
    prediction = i.predict(X_test_features)
    accuracy = accuracy_score(Y_test, prediction)
    print(i, 'accuracy on test', accuracy * 100, '%')
joblib.dump(modelSVC, 'model.joblib')
input = ["Hello mom, are we having dinner tonight?",
         "URGENT! You have won a FREE iPhone. Click here",
         "Call me back later",
         "Congratulations, you won 1000$ prize money",
         "Hey bro what's up"]
input_features = feature_extraction.transform(input)
prediction = modelSVC.predict(input_features)
print(prediction)
