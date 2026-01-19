import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class DataLoad:
    # handles data loading and preprocessing.
    
    def __init__(self, data_path='./data', csv_filename='spam.csv'):
        self.data_path = data_path
        self.csv_filename = csv_filename
        self.feature_extractor = None
    
    def load_data(self):
        # load and preprocess data."""
        csv_path = os.path.join(self.data_path, self.csv_filename)
        data = pd.read_csv(csv_path, encoding='latin-1')
        data = data[['v1', 'v2']]
        print(data['v1'].value_counts())
        print(f"total rows: {len(data)}")
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            data['v2'], data['v1'], test_size=0.2, random_state=3
        )
        
        self.feature_extractor = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
        X_train_features = self.feature_extractor.fit_transform(X_train)
        X_test_features = self.feature_extractor.transform(X_test)
        
        return X_train_features, X_test_features, Y_train, Y_test