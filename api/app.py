from fastapi import FastAPI
from pydantic import BaseModel
import os, pandas as pd, mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
MODEL_URI = os.getenv("MODEL_URI", "models:/iris_cls@production")
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="Iris Inference API")

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    class_id: int

@app.post("/predict")
def predictHandler(request: IrisRequest) -> IrisResponse:
    df = pd.DataFrame(
        [[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
    )

    prediction = int(model.predict(df)[0])
    return IrisResponse(class_id=prediction)
