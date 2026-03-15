import os
import json
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIGURATION SECTION
# -------------------------
CONFIG = {
    "TRAIN_IMAGES_PATH": "/kaggle/working/KBTC_npy/KBTC_training_images.npy",
    "TRAIN_LABELS_PATH": "/kaggle/working/KBTC_npy/KBTC_training_labels.npy",
    "TEST_IMAGES_PATH": "/kaggle/working/KBTC_npy/KBTC_testing_images.npy",
    "TEST_LABELS_PATH": "/kaggle/working/KBTC_npy/KBTC_testing_labels.npy",

    "OUTDIR": "./results",
    "EPOCHS": 50,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-3,

    "VAL_SPLIT": 0.2,
    "PATIENCE": 10,
    "SEED": 101,

    "UNFREEZE_BASE": True,
    "INPUT_SHAPE": (150, 150, 3)
}

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=101):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Plot utilities
# -------------------------
def plot_history(history, outdir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(9,4))
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(outdir, "accuracy.png"))
    plt.close()

    plt.figure(figsize=(9,4))
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(outdir, "loss.png"))
    plt.close()

def plot_confusion(y_true, y_pred, outpath, labels=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu')

    if labels:
        plt.xticks(np.arange(len(labels))+0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels))+0.5, labels)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------
# Model Builder
# -------------------------

def build_model(input_shape=(150,150,3), num_classes=4, base_trainable=False, lr=1e-3):

    inputs = tf.keras.Input(shape=input_shape)

    # -------------------------
    # Data Augmentation Layer
    # -------------------------
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    x = data_augmentation(inputs)

    # -------------------------
    # DenseNet Base
    # -------------------------
    base_model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    base_model.trainable = base_trainable

    x = base_model(x)

    # -------------------------
    # Classification Head
    # -------------------------
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------
# MAIN TRAINING PIPELINE
# -------------------------
def main(cfg):

    set_seed(cfg["SEED"])
    ensure_dir(cfg["OUTDIR"])

    print("Loading datasets...")

    X_train_full = np.load(cfg["TRAIN_IMAGES_PATH"])
    y_train_full = np.load(cfg["TRAIN_LABELS_PATH"])

    X_test = np.load(cfg["TEST_IMAGES_PATH"])
    y_test = np.load(cfg["TEST_LABELS_PATH"])

    print("Training data:", X_train_full.shape)
    print("Testing data:", X_test.shape)

    # Normalize
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # -------------------------
    # TRAIN / VALIDATION SPLIT
    # -------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=cfg["VAL_SPLIT"],
        stratify=y_train_full,
        random_state=cfg["SEED"]
    )

    print("Train:", x_train.shape)
    print("Validation:", x_val.shape)
    
    # One-hot encoding
    # -------------------------
    encoder = OneHotEncoder(sparse_output=False)

    y_train_ohe = encoder.fit_transform(y_train.reshape(-1,1))
    y_val_ohe = encoder.transform(y_val.reshape(-1,1))
    y_test_ohe = encoder.transform(y_test.reshape(-1,1))

    num_classes = y_train_ohe.shape[1]

    # -------------------------
    # Build model
    # -------------------------
    model = build_model(
        input_shape=cfg["INPUT_SHAPE"],
        num_classes=num_classes,
        base_trainable=cfg["UNFREEZE_BASE"],
        lr=cfg["LEARNING_RATE"]
    )

    model.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_model_path = os.path.join(cfg["OUTDIR"], f"best_model_{timestamp}.keras")

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["PATIENCE"],
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3
        )
    ]

    # -------------------------
    # TRAINING
    # -------------------------
    history = model.fit(
        x_train,
        y_train_ohe,
        validation_data=(x_val, y_val_ohe),
        epochs=cfg["EPOCHS"],
        batch_size=cfg["BATCH_SIZE"],
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------
    # FINAL MODEL SAVE
    # -------------------------
    final_model_path = os.path.join(cfg["OUTDIR"], f"final_model_{timestamp}.keras")
    model.save(final_model_path)

    # -------------------------
    # PLOTS
    # -------------------------
    plot_history(history, cfg["OUTDIR"])

    # -------------------------
    # TEST EVALUATION
    # -------------------------
    print("\nEvaluating on TEST dataset")

    eval_res = model.evaluate(X_test, y_test_ohe)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report["eval"] = {
        "loss": float(eval_res[0]),
        "accuracy": float(eval_res[1])
    }

    save_json(report, os.path.join(cfg["OUTDIR"], "classification_report.json"))

    plot_confusion(
        y_test,
        y_pred,
        os.path.join(cfg["OUTDIR"], "confusion_matrix.png"),
        labels=[str(c) for c in encoder.categories_[0]]
    )

    save_json(cfg, os.path.join(cfg["OUTDIR"], "used_config.json"))

    print("\nTraining Complete")
    print("Best model:", best_model_path)
    print("Final model:", final_model_path)


if __name__ == "__main__":
    main(CONFIG)