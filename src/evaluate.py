# src/evaluate.py
import sys
import mlflow
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.data import load_dataset
import mlflow.sklearn

THRESHOLD = 0.80  # 배포 허용 최소 성능(예시)
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "iris_rf"

def evaluate_latest():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("Experiment not found.")
        sys.exit(1)

    # 최신 run 하나 가져오기 (start_time desc)
    runs = client.search_runs([exp.experiment_id],
                              order_by=["attributes.start_time DESC"],
                              max_results=1)
    if not runs:
        print("No runs found.")
        sys.exit(1)

    run = runs[0]
    f1 = run.data.metrics.get("f1_macro", None)
    print(f"Latest run: {run.info.run_id}, f1_macro={f1}")
    if f1 is None or f1 < THRESHOLD:
        print(f"❌ Failed gate: f1_macro < {THRESHOLD}")
        sys.exit(2)
    print("✅ Passed gate.")
    sys.exit(0)

if __name__ == "__main__":
    evaluate_latest()
