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
# Edit these defaults for your repository or experiment.
CONFIG = {
    "IMAGES_PATH": "/content/drive/MyDrive/KBTC_images.npy",
    "LABELS_PATH": "/content/drive/MyDrive/KBTC_labels.npy",
    "OUTDIR": "./results",
    "EPOCHS": 50,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 1e-3,
    "TEST_SIZE": 0.1,
    "PATIENCE": 10,
    "SEED": 101,
    "UNFREEZE_BASE": True,   # set True to unfreeze DenseNet base for fine-tuning
    "INPUT_SHAPE": (150, 150, 3)  # target image shape (height, width, channels)
}

# -------------------------
# Reproducibility & utils
# -------------------------
def set_seed(seed: int = 101):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def plot_history(history, outdir):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(9,4))
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.savefig(os.path.join(outdir, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(9,4))
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.ylim(0, max(max(loss or [0]), max(val_loss or [0]), 2))
    plt.legend(loc='upper left')
    plt.title('Loss')
    plt.savefig(os.path.join(outdir, 'loss.png'))
    plt.close()

def plot_confusion(y_true, y_pred, outpath, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu')
    if labels:
        plt.xticks(np.arange(len(labels))+0.5, labels, rotation=45)
        plt.yticks(np.arange(len(labels))+0.5, labels, rotation=0)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -------------------------
# Model builder
# -------------------------
def build_model(input_shape=(150,150,3), num_classes=4, base_trainable=False, lr=1e-3):
    base_model = tf.keras.applications.DenseNet201(include_top=False,
                                                   input_shape=input_shape,
                                                   weights='imagenet')
    base_model.trainable = bool(base_trainable)

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Main training flow
# -------------------------
def main(cfg):
    set_seed(cfg["SEED"])
    ensure_dir(cfg["OUTDIR"])

    # Load data
    print('Loading data...')
    X = np.load(cfg["IMAGES_PATH"], allow_pickle=True)
    y = np.load(cfg["LABELS_PATH"])
    print('Images shape', X.shape, 'Labels shape', y.shape)

    # Normalize images
    X = X.astype('float32') / 255.0

    # Ensure channel dimension and resize if needed
    if X.ndim == 3:
        X = np.expand_dims(X, -1)
    if X.shape[1:] != tuple(cfg["INPUT_SHAPE"]):
        print(f'Warning: dataset image shape {X.shape[1:]} differs from CONFIG INPUT_SHAPE {cfg["INPUT_SHAPE"]}. Resizing images.')
        X_resized = []
        for img in X:
            img_tf = tf.image.resize(img, cfg["INPUT_SHAPE"][:2]).numpy()
            if img_tf.shape[-1] == 1 and cfg["INPUT_SHAPE"][-1] == 3:
                img_tf = np.repeat(img_tf, 3, axis=-1)
            X_resized.append(img_tf)
        X = np.stack(X_resized).astype('float32')

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["TEST_SIZE"], random_state=cfg["SEED"], stratify=y)

    # One-hot encode labels (fit on train only)
    encoder = OneHotEncoder(sparse=False)
    y_train_ohe = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_ohe = encoder.transform(y_test.reshape(-1,1))

    num_classes = y_train_ohe.shape[1]
    input_shape = x_train.shape[1:]

    # Build and summarize model
    model = build_model(input_shape=input_shape, num_classes=num_classes,
                        base_trainable=cfg["UNFREEZE_BASE"], lr=cfg["LEARNING_RATE"])
    model.summary()

    # Callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(cfg["OUTDIR"], f'best_model_{timestamp}.keras')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                    save_best_only=True, mode='min', verbose=1)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg["PATIENCE"],
                                                 mode='min', verbose=1, restore_best_weights=True,
                                                 min_delta=1e-3)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=3, verbose=1, mode='min', min_delta=1e-4)
    callbacks = [checkpoint, earlystop, reduce_lr]

    # Train
    history = model.fit(x_train, y_train_ohe,
                        validation_data=(x_test, y_test_ohe),
                        epochs=cfg["EPOCHS"],
                        batch_size=cfg["BATCH_SIZE"],
                        callbacks=callbacks,
                        verbose=1)

    # Save final model
    final_model_path = os.path.join(cfg["OUTDIR"], f'final_model_{timestamp}.keras')
    model.save(final_model_path)

    # Plots and metrics
    plot_history(history, cfg["OUTDIR"])

    eval_res = model.evaluate(x_test, y_test_ohe, verbose=1)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_true = np.argmax(y_test_ohe, axis=-1)

    report = classification_report(y_true, y_pred, output_dict=True)
    report['eval'] = {'loss': float(eval_res[0]), 'accuracy': float(eval_res[1])}
    save_json(report, os.path.join(cfg["OUTDIR"], 'classification_report.json'))

    plot_confusion(y_true, y_pred, os.path.join(cfg["OUTDIR"], 'confusion_matrix.png'),
                   labels=[str(c) for c in encoder.categories_[0]])

    save_json(cfg, os.path.join(cfg["OUTDIR"], 'used_config.json'))

    print('Evaluation loss and accuracy:', eval_res)
    print('Classification report saved to', os.path.join(cfg["OUTDIR"], 'classification_report.json'))
    print('Best model saved to', model_path)
    print('Final model saved to', final_model_path)
    print('Used configuration saved to', os.path.join(cfg["OUTDIR"], 'used_config.json'))

if __name__ == '__main__':
    main(CONFIG)