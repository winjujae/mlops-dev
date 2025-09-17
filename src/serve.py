# src/serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
from typing import List

# 학습 단계에서 저장한 로컬 모델
MODEL_PATH = "models/current"

app = FastAPI()
model = mlflow.sklearn.load_model(MODEL_PATH)

class PredictRequest(BaseModel):
    features: List[List[float]]  # 2D: 배치 예측

@app.post("/predict")
def predict(req: PredictRequest):
    preds = model.predict(req.features).tolist()
    return {"predictions": preds}
