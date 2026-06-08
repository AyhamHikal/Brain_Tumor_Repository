import mlflow
import mlflow.keras
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Brain Tumor MRI Classification")


# Constants
DATA_DIR = "data/Training"
IMG_SIZE = (224, 224)
NUM_CLASSES = 4
EPOCHS = 5


# Hyperparameter search space
search_space = {
    "learning_rate": hp.choice("learning_rate", [0.001, 0.0001]),
    "dropout_rate": hp.choice("dropout_rate", [0.3, 0.5]),
    "batch_size": hp.choice("batch_size", [16, 32]),
    "optimizer": hp.choice("optimizer", ["adam", "rmsprop"]),
    "model_name": hp.choice(
        "model_name",
        ["SimpleCNN", "VGG16", "ResNet50", "EfficientNetB0"]
    )
}

# Preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return image


# Data generator function
def create_generators(batch_size):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_data = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_data = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_data, val_data


# Model function
def build_model(model_name, dropout_rate):

    # -------------------------------
    # Simple CNN
    # -------------------------------
    if model_name == "SimpleCNN":

        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),

            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        return model


    # -------------------------------
    # VGG16
    # -------------------------------
    elif model_name == "VGG16":

        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        model = models.Sequential([
            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),

            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        return model


    # -------------------------------
    # ResNet50
    # -------------------------------
    elif model_name == "ResNet50":

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        model = models.Sequential([
            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),

            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        return model

    # -------------------------------
    # EfficientNetB0
    # -------------------------------
    elif model_name == "EfficientNetB0":

        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        base_model.trainable = False

        model = models.Sequential([
            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),

            layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        return model


    # -------------------------------
    # Error handling
    # -------------------------------
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Hyperopt objective function
def objective(params):
    learning_rate = params["learning_rate"]
    dropout_rate = params["dropout_rate"]
    batch_size = params["batch_size"]
    optimizer_name = params["optimizer"]
    model_name = params["model_name"]

    train_data, val_data = create_generators(batch_size)

    with mlflow.start_run(run_name=f"tuning_{model_name}"):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("preprocessing", "resize_normalize_margin_removal")

        model = build_model(model_name, dropout_rate)

        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            verbose=1
        )

        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_loss", val_loss)

        mlflow.keras.log_model(
    model,
    artifact_path="model",
    registered_model_name="BrainTumorMRIClassifier"
)

        return {
            "loss": val_loss,
            "status": STATUS_OK
        }


# Run hyperparameter tuning
trials = Trials()

best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)

print("Best hyperparameters:")
print(best_params)