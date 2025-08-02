import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt

def run_pipeline(
    spark,
    table_name="AAPL_market_price",
    n_lags=10,
    n_steps=3,
    as_direction=True,
    test_size=0.2,
    shuffle=False,
    return_type="dataframe",
    verbose=True,
    experiment_name="/Users/feldmanngreg@gmail.com/AAPL_Forecaster",
    register_model_name="main.default.updown_forecaster_model"
):
    """
    End-to-end pipeline for loading, training, evaluating, and logging a forecasting model.
    Returns:
        model: trained classifier
        metrics: dict of accuracy metrics
    """
    from data_loader import load_adj_close_from_spark_table
    from features import create_multistep_lagged_numpy
    from modelling import split_data, train_model, predict, evaluate

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Step 1: Load data
    historical_data = load_adj_close_from_spark_table(spark, table_name=table_name, return_type=return_type)

    # Step 2: Create lagged features and labels
    X, y = create_multistep_lagged_numpy(
        historical_data["adj_close"].values,
        n_lags=n_lags,
        n_steps=n_steps,
        as_direction=as_direction
    )

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, shuffle=shuffle)

    # Step 4: Train the model
    model = train_model(X_train, y_train)

    # Step 5: Predict
    y_pred = predict(model, X_test)

    # Step 6: Evaluate
    metrics = evaluate(y_test, y_pred, verbose=verbose)

    # Step 7: Prepare MLflow inputs
    X_test_df = pd.DataFrame(X_test, columns=[f"lag_{i}" for i in range(X_test.shape[1])])
    evaluation_data = X_test_df.copy()
    evaluation_data["label"] = y_test[:, 0]  # only evaluating t+1

    signature = infer_signature(X_test_df, y_pred)

    # Step 8: Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("n_lags", n_lags)
        mlflow.log_param("n_steps", n_steps)
        mlflow.log_param("model_type", "RandomForestClassifier")

        mlflow.log_metric("overall_accuracy", metrics["overall_accuracy"])
        for i, acc in enumerate(metrics["per_step_accuracy"], 1):
            mlflow.log_metric(f"accuracy_t+{i}", acc)

        model_info = mlflow.sklearn.log_model(model,
            name="sk_models",
            input_example=X_test_df.iloc[[0]],
            signature=signature,
            registered_model_name=register_model_name
        )

        # Evaluate using MLflow's evaluator (only on t+1 step)
        for i in range(y_test.shape[1]):
            mlflow.log_metric(f"accuracy_step_{i+1}", np.mean(y_test[:, i] == y_pred[:, i]))

        print(f"✅ Model logged and registered: {model_info.model_uri}")

    return model, metrics