import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"\n Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Fault"]
    )
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.show()


def plot_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        plt.figure()
        plt.barh(features, importances)
        plt.xlabel("Importance")
        plt.title("Random Forest - Feature Importance")
        plt.show()


if __name__ == "__main__":
    # Load and split data
    X_train, X_test, y_train, y_test = load_data()
    features = X_train.columns

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test)
    plot_feature_importance(rf, features)

    # Train SVM
    svm = SVC(kernel="rbf", gamma="scale", C=1.0)
    svm.fit(X_train, y_train)
    evaluate_model("Support Vector Machine", svm, X_test, y_test)
