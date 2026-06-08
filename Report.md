# Brain Tumor MRI Classification MLOps Pipeline using MLflow

## Overview

This project demonstrates a complete Machine Learning Operations (MLOps) workflow for Brain Tumor MRI Classification using MLflow. The objective is to manage the entire machine learning lifecycle, including experiment tracking, hyperparameter tuning, model registry, deployment, and monitoring.

The project was developed as part of the MLOps course at Bahçeşehir University.

---

# Project Objectives

The project aims to:

- Track machine learning experiments using MLflow.
- Train and compare multiple deep learning models.
- Perform hyperparameter tuning.
- Register and manage model versions.
- Deploy the best-performing model as a REST API.
- Monitor model performance after deployment.

---

# Dataset

The dataset consists of Brain MRI images categorized into four classes:

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

The dataset was preprocessed and resized before training.

---

# Project Structure

```text
Brain_tumor_MLOps/
│
├── data/
│   ├── Training/
│   └── Testing/
│
├── train.py
├── register_model.py
├── test_deployment.py
├── monitor_model.py
│
├── mlruns/
├── mlflow.db
│
├── requirements.txt
├── README.md
└── venv/
```

---

# Technologies Used

## Machine Learning

- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-Learn

## MLOps

- MLflow
- Model Registry
- MLflow Model Serving

## Visualization

- Matplotlib

---

# Model Development

Four models were trained and compared:

## 1. SimpleCNN

A custom Convolutional Neural Network used as a baseline model.

## 2. VGG16

Transfer learning model based on the VGG16 architecture.

## 3. ResNet50

Transfer learning model based on the ResNet50 architecture.

## 4. EfficientNetB0

Transfer learning model based on the EfficientNetB0 architecture.

---

# Experiment Tracking

MLflow was used to track:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Learning Rate
- Batch Size
- Optimizer
- Dropout Rate

Each training session was logged as a separate MLflow run.

Example tracked metrics:

```text
train_accuracy
val_accuracy
train_loss
val_loss
```

---

# Hyperparameter Tuning

Several experiments were performed using different hyperparameter combinations.

The following parameters were tuned:

- Learning Rate
- Batch Size
- Optimizer
- Dropout Rate

Each tuning run was automatically tracked using MLflow.

---

# Best Model Selection

After comparing all models, VGG16 achieved the best validation performance.

## Best Model

```text
Model: VGG16
Validation Accuracy: 84.46%
Validation Loss: 0.4081
```

This model was selected for deployment.

---

# Model Registry

MLflow Model Registry was used to manage model versions.

Registered Model:

```text
BrainTumorMRIClassifier
```

Model versions were created and stored in the registry.

Example versions:

```text
Version 23
Version 24
Version 25
```

This allows tracking and managing model lifecycle stages.

---

# Model Deployment

The best model was deployed using MLflow Model Serving.

Deployment command:

```bash
mlflow models serve \
-m "/Users/ayhamhikal/mlartifacts/2/143c9585426c4639b67b9e8cd1723d35/artifacts/model" \
-p 5001 \
--no-conda
```

The deployment created a REST API endpoint:

```text
http://127.0.0.1:5001/invocations
```

The model successfully responded to prediction requests.

Example response:

```json
{
  "predictions": [
    [0.4771, 0.3661, 0.1380, 0.0189]
  ]
}
```

---

# Deployment Testing

The deployment was tested using:

```bash
python test_deployment.py
```

The script:

1. Loads a sample MRI image.
2. Sends the image to the deployed API.
3. Receives prediction probabilities.
4. Displays the prediction result.

Successful API responses confirmed correct deployment.

---

# Model Monitoring

A monitoring pipeline was implemented using:

```bash
python monitor_model.py
```

The monitoring system logs simulated production metrics into MLflow.

Tracked metrics include:

- Weekly Accuracy
- Average Confidence
- Misclassification Rate
- Number of Predictions

Example monitoring results:

```text
weekly_accuracy        = 0.8806
average_confidence     = 0.9199
misclassification_rate = 0.1194
number_of_predictions  = 67
```

Monitoring results are stored in a dedicated MLflow experiment:

```text
Brain Tumor MRI Monitoring
```

---

# MLflow Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Experiment Tracking
   ↓
Hyperparameter Tuning
   ↓
Best Model Selection
   ↓
Model Registry
   ↓
Model Deployment
   ↓
Performance Monitoring
```

---

# Results

The project successfully demonstrated an end-to-end MLOps workflow using MLflow.

Achievements:

- Experiment tracking implemented.
- Multiple deep learning models trained.
- Hyperparameter tuning completed.
- Best model selected.
- Model Registry utilized.
- Model deployed as REST API.
- Monitoring system implemented.

---

# Future Improvements

Potential future enhancements include:

- Automated retraining pipeline.
- Data drift detection.
- Integration with Docker.
- Deployment on Vertex AI.
- Deployment on AWS SageMaker.
- CI/CD pipeline using GitHub Actions.
- Real-time monitoring dashboard.

---

# Conclusion

This project demonstrates the practical application of MLflow for managing the complete machine learning lifecycle. By integrating experiment tracking, model registry, deployment, and monitoring, the project provides a production-oriented workflow for Brain Tumor MRI Classification and highlights the importance of MLOps practices in modern AI systems.