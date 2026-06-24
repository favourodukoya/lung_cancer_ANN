import numpy as np
import tensorflow as tf
from model import get_callbacks


def train_model(model:      tf.keras.Model,
                X_train:    np.ndarray,
                y_train:    np.ndarray,
                X_val:      np.ndarray,
                y_val:      np.ndarray,
                epochs:     int = 150,
                batch_size: int = 32) -> tf.keras.callbacks.History:
    """
    Train the ANN and return the training history.
    Args:
        model:       Compiled Keras model from build_model().
        X_train:     Scaled training feature array.
        y_train:     One-hot training label array.
        X_val:       Scaled validation feature array.
        y_val:       One-hot validation label array.
        epochs:      Maximum number of training epochs.
        batch_size:  Samples per gradient update step.

    Returns:
        Keras History object with loss and accuracy per epoch,
        usable for plotting training curves.
    """
    callbacks = get_callbacks()

    print(f"\n[Training] Starting — max {epochs} epochs | batch {batch_size}.")
    print(f"[Training] EarlyStopping will halt after 15 non-improving epochs.")
    print(f"[Training] ReduceLROnPlateau will halve lr after 7 non-improving epochs.\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Report the best validation accuracy seen during training
    best_val_acc  = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    epochs_run    = len(history.history['loss'])

    print(f"\n[Training] Complete after {epochs_run} epochs.")
    print(f"[Training] Best val_accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"[Training] Best val_loss     : {best_val_loss:.4f}")

    return history