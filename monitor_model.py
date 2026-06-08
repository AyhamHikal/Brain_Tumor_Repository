import mlflow
import random
from datetime import datetime

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Brain Tumor MRI Monitoring")

with mlflow.start_run(run_name=f"monitoring_{datetime.now().date()}"):

    weekly_accuracy = random.uniform(0.80, 0.95)
    average_confidence = random.uniform(0.75, 0.95)
    misclassification_rate = 1 - weekly_accuracy
    number_of_predictions = random.randint(20, 100)

    mlflow.log_metric("weekly_accuracy", weekly_accuracy)
    mlflow.log_metric("average_confidence", average_confidence)
    mlflow.log_metric("misclassification_rate", misclassification_rate)
    mlflow.log_metric("number_of_predictions", number_of_predictions)

    mlflow.log_param("model_name", "BrainTumorMRIClassifier")
    mlflow.log_param("model_uri", "runs:/143c9585426c4639b67b9e8cd1723d35/model")
    mlflow.log_param("monitoring_type", "simulated_new_mri_data")

print("Monitoring metrics logged successfully.")