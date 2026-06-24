import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam


NUM_CLASSES  = 3                         # Low, Medium, High
CLASS_NAMES  = ['Low', 'Medium', 'High']
DEFAULT_LR   = 0.001                     # Adam learning rate starting point



def build_model(input_dim: int, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Args:
        input_dim:   Number of input features (23 for this dataset).
        num_classes: Number of output classes (3: Low/Medium/High).

    Returns:
        Compiled tf.keras.Model ready for training.
    """

    model = Sequential(name='cancer_risk_ann', layers=[
        Input(shape=(input_dim,), name='input'),

        #Hidden Layer 1
        Dense(64, activation='relu', name='dense_1'),
        BatchNormalization(name='batchnorm_1'),
        Dropout(0.3, name='dropout_1'),

        # Hidden Layer 2
        Dense(32, activation='relu',   name='dense_2'),
        BatchNormalization(name='batchnorm_2'),
        Dropout(0.2, name='dropout_2'),

        # Hidden Layer 3
        Dense(16, activation='relu', name='dense_3'),
        Dropout(0.1, name='dropout_3'),

        # Output Layer
        # Softmax produces a probability per class; argmax gives the
        # final prediction. 3 neurons = 3 risk levels (Low/Medium/High).
        Dense(num_classes, activation='softmax', name='output'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=DEFAULT_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("[Model] Architecture built and compiled.")
    model.summary()
    return model


# Callbacks

def get_callbacks(model_save_path: str = 'best_model.keras') -> list:
    """
    Args:
        model_save_path: Path to save the best model checkpoint.

    Returns:
        List of configured Keras callback objects.
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Halve the learning rate on plateau
        patience=7,
        min_lr=1e-6,       # Never go below 0.000001
        verbose=1
    )

    print(f"[Callbacks] EarlyStopping(patience=15) | "
          f"ModelCheckpoint('{model_save_path}') | "
          f"ReduceLROnPlateau(patience=7, factor=0.5) ready.")
    return [early_stop, checkpoint, reduce_lr]