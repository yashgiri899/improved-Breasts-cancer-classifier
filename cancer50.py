import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set dataset filename (assuming it's in the same repo)
DATASET_PATH = "Data (2).csv"

def load_data(file_path):
    """
    Loads dataset from CSV file.
    """
    logging.info("Loading dataset...")
    dataset = pd.read_csv(file_path)
    
    # Drop ID column if necessary
    if 'Sample code number' in dataset.columns:
        dataset = dataset.drop(columns=['Sample code number'])

    # Separate features and target variable
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Convert target labels if necessary (2 → 0, 4 → 1)
    if set(y) == {2, 4}:
        y = np.where(y == 2, 0, 1)
    
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    logging.info("Splitting dataset into train and test sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=300, random_state=42):
    """
    Trains a Random Forest Classifier.
    """
    logging.info(f"Training Random Forest Classifier with {n_estimators} trees...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints results.
    """
    logging.info("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Print results
    print("\n🔹 Model Performance 🔹")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n🔹 Confusion Matrix 🔹")
    print(confusion_matrix(y_test, y_pred))
    print("\n🔹 Classification Report 🔹")
    print(classification_report(y_test, y_pred))

def feature_importance(model, feature_names=None):
    """
    Displays feature importance from the trained Random Forest model.
    """
    logging.info("Extracting feature importance...")
    importances = model.feature_importances_
    feature_names = feature_names if feature_names else [f"Feature {i}" for i in range(len(importances))]
    
    print("\n🔹 Feature Importance 🔹")
    for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")

def main():
    """
    Main function to load data, train the model, and evaluate performance.
    """
    # Load dataset
    X, y = load_data(DATASET_PATH)

    # Split data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Show feature importance
    feature_importance(model)

if __name__ == "__main__":
    main()
