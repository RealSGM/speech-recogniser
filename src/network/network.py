import numpy as np
import tensorflow as tf
import pickle
import keras
import file_reader as fr

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from utils.config import MainConfig, TrainingConfig


def encode_labels(labels: list, class_list: list) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def get_model_from_config(input_shape: tuple, num_classes: int, cfg: TrainingConfig) -> keras.Model:
    model = Sequential()
    model.add(InputLayer(shape=input_shape))

    filters = cfg.base_filters

    for _ in range(cfg.conv_blocks):
        model.add(Conv2D(filters, cfg.kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=cfg.pool_size))

        if cfg.use_batch_norm:
            model.add(BatchNormalization())

        filters *= 2

    model.add(Flatten())

    if cfg.dropout_rate > 0:
        model.add(Dropout(cfg.dropout_rate))

    model.add(Dense(cfg.dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def create_model(input_shape: tuple, num_classes: int, config: MainConfig) -> keras.Model:
    model = get_model_from_config(input_shape, num_classes, config.training)
    adam = Adam(learning_rate=config.training.learning_rate)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam) # type: ignore
    return model


def train_model(model, X_train, y_train, X_val, y_val, config: MainConfig):
    # Stop early if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    lr_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        min_lr=0.00001, 
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        verbose="auto",
        callbacks=[lr_reduction, early_stop] 
    )
    return history


# Save the model and encoder
def save_model(model, encoder, config: MainConfig) -> None:
    full_save_path = fr.get_config_path(config)
    
    model.save(f"{full_save_path}/{config.notes}.keras") # type: ignore
    
    with open(f"{full_save_path}/{config.notes}.pkl", "wb") as f:
        pickle.dump(encoder, f)


# Load the model and encoder
def load_model(config: MainConfig) -> tf.keras.Model: # type: ignore
    full_save_path = fr.get_config_path(config)

    model = tf.keras.models.load_model(f"{full_save_path}/{config.notes}.keras") # type: ignore
    
    with open(f"{full_save_path}/{config.notes}.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


## TODO Retrieve audio and visual specific results
def get_results_from_file(config: MainConfig) -> tuple[float, float]:
    audio_result = 0.0
    visual_result = 0.0
    
    results = fr.get_results_dict(config)
    
    if config.integration.audio_weight > 0:
        audio_result = results.get("test_accuracy_audio", 0.0)
    
    if config.integration.visual_weight > 0:
        visual_result = results.get("test_accuracy_visual", 0.0)
        
    return audio_result, visual_result
