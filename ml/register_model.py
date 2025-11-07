import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")

client = MlflowClient()
run_id = "986993762816158947"
model_name = "iris_cls"

try:
    client.create_registered_model(model_name)
except Exception:
    pass

mv = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id,
)

client.set_registered_model_alias(
    name="iris_cls",
    alias="production",
    version=mv.version,
)

print (f"Registered iris_cls v{mv.version} and set alias @production")

client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Production",
)

print(f"Registered {model_name} v{mv.version} as Production")
