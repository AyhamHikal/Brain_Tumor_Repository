import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
RUN_ID = "143c9585426c4639b67b9e8cd1723d35"

model_uri = f"runs:/{RUN_ID}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name="BrainTumorMRIClassifier"
)

print("Registered model:", result.name)
print("Version:", result.version)