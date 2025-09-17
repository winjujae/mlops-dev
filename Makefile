.PHONY: all data train eval serve ui clean

PY=python -m src

all: train eval

data:
	@$(PY).data

train:
	@$(PY).train

eval:
	@$(PY).evaluate

serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000

clean:
	rm -rf artifacts models mlflow.db __pycache__ .pytest_cache