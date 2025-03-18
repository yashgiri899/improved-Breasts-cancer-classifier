import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(file_path):
    """
    Loads dataset from CSV file.

    Parameters:
        file_path (str): Path to the dataset CSV file.

    Returns:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Target variable
    """
    logging.info("Loading dataset...")
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 1:-1].values  # Assuming first column is an index and last column is the target
    y = dataset.iloc[:, -1].values
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target array.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logging.info("Splitting dataset into train and test sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=300, random_state=42):
    """
    Trains a Random Forest Classifier.

    Parameters:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target array.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed.

    Returns:
        trained_model (RandomForestClassifier)
    """
    logging.info(f"Training Random Forest Classifier with {n_estimators} trees...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and prints results.

    Parameters:
        model (RandomForestClassifier): Trained classifier.
        X_test (numpy.ndarray): Test feature matrix.
        y_test (numpy.ndarray): True test labels.

    Returns:
        None
    """
    logging.info("Evaluating model performance...")
    y_pred = model.predict(X_test)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print("\n🔹 Model Performance 🔹")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\n🔹 Confusion Matrix 🔹")
    print(conf_matrix)
    print("\n🔹 Classification Report 🔹")
    print(class_report)


def feature_importance(model, feature_names=None):
    """
    Displays feature importance from the trained Random Forest model.

    Parameters:
        model (RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names (optional).

    Returns:
        None
    """
    logging.info("Extracting feature importance...")
    importances = model.feature_importances_
    feature_names = feature_names if feature_names else [f"Feature {i}" for i in range(len(importances))]

    # Sort features by importance
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n🔹 Feature Importance 🔹")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")


def main():
    """
    Main function to load data, train the model, and evaluate performance.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest Classifier.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV file.")
    args = parser.parse_args()

    # Load dataset
    X, y = load_data(args.data)

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
