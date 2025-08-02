from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def split_data(X, y, test_size=0.2, shuffle=False):
    """
    Splits the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


def train_model(X_train, y_train, base_model=None):
    """
    Trains a multi-output classifier.

    Parameters:
        X_train: features for training
        y_train: binary classification targets (multi-step)
        base_model: base classifier (default is RandomForest)

    Returns:
        Fitted MultiOutputClassifier model
    """
    if base_model is None:
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Uses trained model to predict on X_test.
    """
    return model.predict(X_test)


def evaluate(y_true, y_pred, verbose=True):
    """
    Evaluates prediction accuracy.

    Returns a dict of overall and per-step accuracy.
    """
    per_step_accuracy = (y_true == y_pred).mean(axis=0)
    overall_accuracy = (y_true == y_pred).mean()

    if verbose:
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        for i, acc in enumerate(per_step_accuracy, 1):
            print(f"t+{i} accuracy: {acc:.4f}")
        print("\nSample prediction:")
        print("Predicted:", y_pred[0])
        print("Actual:   ", y_true[0])

    return {
        "overall_accuracy": overall_accuracy,
        "per_step_accuracy": per_step_accuracy.tolist()
    }
