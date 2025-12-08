import numpy as np
import tensorflow as tf
import pickle
import keras
import file_reader as fr

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import TimeDistributed, Bidirectional, LSTM, Reshape
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from utils.config import MainConfig, TrainingConfig


def encode_labels(labels: list, class_list: list) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def get_model_from_config(input_shape: tuple, num_classes: int, cfg: TrainingConfig) -> keras.Model:
    """
    Generic 2D-CNN model used for audio and early-integration super-vectors.
    Expects input_shape like (H, W, 1) or (T, F, 1).
    """
    model = Sequential()
    model.add(InputLayer(shape=input_shape))

    filters = cfg.base_filters

    for _ in range(cfg.conv_blocks):
        model.add(Conv2D(filters, cfg.kernel_size, activation='relu', padding='same'))
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


def get_visual_temporal_model(input_shape: tuple, num_classes: int, cfg: MainConfig) -> keras.Model:
    """
    Visual-only model: TimeDistributed CNN over (96x96) ROI + optical flow,
    followed by a BiLSTM over time.

    Expects input_shape = (T, F_flat),
    where F_flat = 96 * 96 * num_channels (num_channels should be 6 with base+delta+delta-delta).
    """
    model = Sequential()
    model.add(InputLayer(shape=input_shape))

    # Unpack shape
    time_steps = input_shape[0]
    feat_dim = input_shape[1]

    H, W = cfg.visual.roi_size
    spatial_dim = H * W
    num_channels = feat_dim // spatial_dim

    # (T, F_flat) -> (T, 96, 96, C)
    model.add(Reshape((time_steps, H, W, num_channels)))

    filters = cfg.training.base_filters

    # TimeDistributed CNN blocks
    for _ in range(cfg.training.conv_blocks):
        model.add(TimeDistributed(
            Conv2D(filters, cfg.training.kernel_size, activation='relu', padding='same')
        ))
        if cfg.training.use_batch_norm:
            model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D(pool_size=cfg.training.pool_size)))

        filters *= 2

    # Flatten per time step
    model.add(TimeDistributed(Flatten()))

    # Temporal modelling
    model.add(Bidirectional(LSTM(128, return_sequences=False)))

    if cfg.training.dropout_rate > 0:
        model.add(Dropout(cfg.training.dropout_rate))

    model.add(Dense(cfg.training.dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def create_model(input_shape: tuple, num_classes: int, config: MainConfig) -> keras.Model:
    """
    Chooses model architecture based on config.notes:
      - "visual": TimeDistributed CNN + BiLSTM (visual-only)
      - anything else: generic 2D-CNN (audio, early integration, etc.)
    """
    # Visual-only branch
    if config.notes == "visual":
        model = get_visual_temporal_model(input_shape, num_classes, config)
    else:
        # Audio / early / other branches
        model = get_model_from_config(input_shape, num_classes, config.training)

    adam = Adam(learning_rate=config.training.learning_rate)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)  # type: ignore
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


def get_results_from_file(config: MainConfig) -> tuple[float, float]:
    audio_result = 0.0
    visual_result = 0.0
    
    results = fr.get_results_dict(config)
    
    if config.integration.audio_weight > 0:
        audio_result = results.get("test_accuracy_audio", 0.0)
    
    if config.integration.visual_weight > 0:
        visual_result = results.get("test_accuracy_visual", 0.0)
        
    return audio_result, visual_result
