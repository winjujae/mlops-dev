# src/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.data import load_dataset

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # 로컬 sqlite
#MLFLOW_ARTIFACTS = "./artifacts"
EXPERIMENT_NAME = "iris_rf"

def _ensure_experiment(experiment_name: str, artifact_dir: str):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_location = f"file://{os.path.abspath(artifact_dir)}"
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)

def main(n_estimators:int=300, max_depth:int=5, test_size:float=0.2, random_state:int=42):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ensure_experiment(EXPERIMENT_NAME, artifact_dir="artifacts")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")

        # 파라미터/메트릭 로깅
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": random_state
        })
        mlflow.log_metric("f1_macro", f1)

        # 모델 저장 (MLflow 형식)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 로컬 복사본도 저장 (서빙용)
        os.makedirs("models", exist_ok=True)
        mlflow.sklearn.save_model(model, path="models/current")

        print(f"Run ID: {run.info.run_id}, f1_macro={f1:.4f}")

if __name__ == "__main__":
    main()
