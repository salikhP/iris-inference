from fastapi import FastAPI
from pydantic import BaseModel
import os, pandas as pd, mlflow

app = FastAPI(title="Iris Inference API")

RUN_ID = os.getenv("RUN_ID")
MODEL_URI = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(MODEL_URI)

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
