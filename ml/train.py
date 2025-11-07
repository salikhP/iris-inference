import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature


def main() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        params = {"max_iter": 300, "C": 1.0, "penalty": "l2", "solver": "lbfgs"}
        model = LogisticRegression(**params).fit(Xtr, ytr)

        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")

        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})

        signature = infer_signature(Xte.head(2), model.predict(Xte.head(2)))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=Xte.head(2),
        )

        run_id = mlflow.active_run().info.run_id
        print(f"OK. Run logged. run_id={run_id}")

if __name__ == "__main__":
    main()
