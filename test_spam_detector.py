import unittest
import os
import tempfile
import shutil
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data_load import DataLoad
from main import ModelTrainer


class TestDataLoad(unittest.TestCase):
    # core tests for DataLoad class.

    def setUp(self):
        # set up test fixtures.
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, 'test_spam.csv')

        # create sample test data
        test_data = pd.DataFrame({
            'v1': ['ham', 'spam', 'ham', 'spam', 'ham'] * 20,  # 100 rows
            'v2': [
                      'Hello, how are you?',
                      'URGENT! Win a free iPhone now!',
                      'Call me later',
                      'Congratulations! You won $1000!',
                      'Hey, what\'s up?'
                  ] * 20
        })
        test_data.to_csv(self.test_csv, index=False, encoding='latin-1')

    def tearDown(self):
        # clean up.
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        # test DataLoad initialization.
        loader = DataLoad()
        self.assertEqual(loader.data_path, './data')
        self.assertEqual(loader.csv_filename, 'spam.csv')
        self.assertIsNone(loader.feature_extractor)

        custom_loader = DataLoad(data_path=self.test_dir, csv_filename='test_spam.csv')
        self.assertEqual(custom_loader.data_path, self.test_dir)
        self.assertEqual(custom_loader.csv_filename, 'test_spam.csv')

    def test_load_data(self):
        # test data loading and preprocessing.
        loader = DataLoad(data_path=self.test_dir, csv_filename='test_spam.csv')
        X_train, X_test, y_train, y_test = loader.load_data()

        # check return values
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        # check feature extractor was created
        self.assertIsNotNone(loader.feature_extractor)
        self.assertIsInstance(loader.feature_extractor, TfidfVectorizer)

        # check data shapes
        self.assertEqual(X_train.shape[0], len(y_train))
        self.assertEqual(X_test.shape[0], len(y_test))


class TestModelTrainer(unittest.TestCase):
    # core tests for ModelTrainer class.

    def setUp(self):
        # set up test fixtures. create sample data
        self.feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

        sample_texts = [
                           'Hello, how are you?',
                           'URGENT! Win a free iPhone now!',
                           'Call me later',
                           'Congratulations! You won $1000!',
                           'Hey, what\'s up?'
                       ] * 20  # 100 samples

        sample_labels = ['ham', 'spam', 'ham', 'spam', 'ham'] * 20

        X_features = self.feature_extractor.fit_transform(sample_texts)
        y = pd.Series(sample_labels)

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

    def test_initialization(self):
        # test ModelTrainer initialization.
        trainer = ModelTrainer(
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_extractor
        )
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.X_test)
        self.assertIsNotNone(trainer.feature_extractor)

    def test_train_and_evaluate(self):
        # test model training and evaluation.
        trainer = ModelTrainer(
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_extractor
        )

        model = SVC(kernel='sigmoid', C=1.0)
        accuracy = trainer.train_and_evaluate(model)

        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_predict(self):
        # test prediction on text messages.
        trainer = ModelTrainer(
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_extractor
        )

        model = SVC(kernel='sigmoid', C=1.0)
        model.fit(self.X_train, self.y_train)

        test_messages = ['Hello, how are you?', 'URGENT! Win a free iPhone!']
        predictions = trainer.predict(model, test_messages)

        self.assertEqual(len(predictions), len(test_messages))
        self.assertTrue(all(pred in ['ham', 'spam'] for pred in predictions))

    def test_save_model(self):
        # test model saving.
        trainer = ModelTrainer(
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_extractor
        )

        model = SVC(kernel='sigmoid', C=1.0)
        model.fit(self.X_train, self.y_train)

        test_file = os.path.join(tempfile.gettempdir(), 'test_model.joblib')
        trainer.save_model(model, test_file)

        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)


class TestIntegration(unittest.TestCase):
    # integration test for complete workflow.

    def setUp(self):
        # set up test fixtures.
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.test_dir, 'test_spam.csv')

        test_data = pd.DataFrame({
            'v1': ['ham', 'spam', 'ham', 'spam', 'ham'] * 40,  # 200 rows
            'v2': [
                      'Hello, how are you doing?',
                      'URGENT! Win a free iPhone now!',
                      'Call me later when free',
                      'Congratulations! You won $1000!',
                      'Hey, what\'s up?'
                  ] * 40
        })
        test_data.to_csv(self.test_csv, index=False, encoding='latin-1')

    def tearDown(self):
        # clean up.
        shutil.rmtree(self.test_dir)

    def test_full_pipeline(self):
        # test complete pipeline from data loading to prediction. load data
        loader = DataLoad(data_path=self.test_dir, csv_filename='test_spam.csv')
        X_train, X_test, y_train, y_test = loader.load_data()

        # train and evaluate
        trainer = ModelTrainer(
            X_train, X_test, y_train, y_test, loader.feature_extractor
        )

        model = SVC(kernel='sigmoid', C=1.0)
        accuracy = trainer.train_and_evaluate(model)

        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

        # test prediction
        test_messages = ['Hello, how are you?', 'URGENT! Win a free iPhone!']
        predictions = trainer.predict(model, test_messages)

        self.assertEqual(len(predictions), len(test_messages))

if __name__ == '__main__':
    unittest.main(verbosity=2)
