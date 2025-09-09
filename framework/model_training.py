import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def load_data(file="data/simulated/telemetry_faulty.csv"):
    df = pd.read_csv(file)
    X = df[["voltage", "temperature", "cpu_load", "signal_strength"]]
    y = df["fault_label"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Fault"]
    )
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Feature importance
    feature_importances = model.feature_importances_
    features = X_test.columns

    plt.figure()
    plt.barh(features, feature_importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.show()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
