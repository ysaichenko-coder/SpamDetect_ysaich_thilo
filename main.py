from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data_load import DataLoad


class ModelTrainer:
    # handles model training and evaluation.

    def __init__(self, X_train, X_test, y_train, y_test, feature_extractor):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_extractor = feature_extractor

    def train_and_evaluate(self, model):
        # train model and return accuracy.
        model.fit(self.X_train, self.y_train)
        prediction = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, prediction)
        return accuracy

    def predict(self, model, texts):
        # predict spam/ham for text messages.
        features = self.feature_extractor.transform(texts)
        return model.predict(features)

    def save_model(self, model, filepath='model.joblib'):
        # save trained model.
        joblib.dump(model, filepath)


if __name__ == '__main__':
    # load data
    loader = DataLoad()
    X_train_features, X_test_features, Y_train, Y_test = loader.load_data()

    # train and evaluate models
    trainer = ModelTrainer(
        X_train_features, X_test_features, Y_train, Y_test, loader.feature_extractor
    )

    modelSVC = SVC(kernel='sigmoid', C=1.0)
    modelLR = LogisticRegression()

    for model in [modelSVC, modelLR]:
        accuracy = trainer.train_and_evaluate(model)
        print(f'{model.__class__.__name__} accuracy on test: {accuracy * 100:.2f}%')

    # save best model
    trainer.save_model(modelSVC, 'modelSVC.joblib')
    trainer.save_model(modelLR, 'modelLR.joblib')

    # test predictions
    test_messages = [
        "Hello mom, are we having dinner tonight?",
        "URGENT! You have won a FREE iPhone. Click here",
        "Call me back later",
        "Congratulations, you won 1000$ prize money",
        "Hey bro what's up"
    ]
    predictions = trainer.predict(modelSVC, test_messages)
    print(predictions)
