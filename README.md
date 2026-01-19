# SMS Spam Filter

A simple Machine Learning project to detect Spam SMS using **Support Vector Machine (SVC)** and **Logistic Regression**.

## Features

* **Data Preprocessing**: Automatic loading, cleaning, and splitting of the dataset.
* **Text Vectorization**: Uses `TfidfVectorizer` to convert text into numerical features.
* **Multi-Model Training**: Compares Support Vector Machine (SVC) and Logistic Regression.
* **OOP Design**: Modular code separated into `DataLoad` and `ModelTrainer` classes.
* **Model Persistence**: Saves trained models (`.joblib`) for future use.
* **Comprehensive Testing**: Includes unit tests and integration tests using `unittest`.

## Quick Start

1.  **Clone and Install:**
    ```bash
    git clone https://github.com/ysaichenko-coder/SpamDetect_ysaich_thilo.git
    cd SpamDetect_ysaich_thilo
    pip install pandas scikit-learn joblib
    ```

2.  **Run the Project:**
    ```bash
    python main.py
    ```
    *This will train the models, show accuracy, and save them.*

3.  **Run Tests:**
    ```bash
    python test_spam_detector.py
    ```
## Project Structure

    ```text
    ├── data/
    │   └── spam.csv              # The raw dataset (SMS Spam Collection)
    ├── data_load.py              # Class for data loading and preprocessing
    ├── main.py                   # Main script for training and evaluation
    ├── test_spam_detector.py     # Unit and Integration tests
    ├── modelSVC.joblib           # Saved SVC model (generated after run)
    ├── modelLR.joblib            # Saved Logistic Regression model (generated after run)
    ├── README.md                 # Project documentation
    └── requirements.txt          # List of required Python libraries 