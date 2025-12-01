import numpy as np
import tensorflow as tf
import pickle
import os
import keras

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


SAVE_PATH = "data/networks"


def encode_labels(labels: list, class_list: list) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def create_model(input_shape: tuple, num_classes: int, use_complex=False) -> keras.Model:
    model = get_complex_model(input_shape, num_classes) if use_complex else get_simple_model(input_shape, num_classes)
    adam = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam) # type: ignore
    return model


def get_simple_model(input_shape: tuple, num_classes: int) -> tf.keras.Model: # type: ignore
    model = Sequential() 
    model.add(InputLayer(shape=input_shape)) 
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3))) 
    model.add(Flatten()) 
    model.add(Dense(256)) 
    model.add(Activation('relu')) 
    model.add(Dense(num_classes)) 
    model.add(Activation('softmax')) 
    return model


def get_complex_model(input_shape: tuple, num_classes: int) -> tf.keras.Model: # type: ignore
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
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
        batch_size=batch_size,
        epochs=epochs,
        verbose="auto",
        callbacks=[lr_reduction, early_stop] 
    )
    return history


# Save the model and encoder
def save_model(model, encoder, test_name="test") -> None:
    # Create save directory if it doesn't exist
    if not os.path.exists(f"{SAVE_PATH}/{test_name}"):
        os.makedirs(f"{SAVE_PATH}/{test_name}")
    
    model.save(f"{SAVE_PATH}/{test_name}/speech_model.keras") # type: ignore
    
    with open(f"{SAVE_PATH}/{test_name}/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)


# Load the model and encoder
def load_model(test_name="test") -> tf.keras.Model: # type: ignore
    model = tf.keras.models.load_model(f"{SAVE_PATH}/{test_name}/speech_model.keras") # type: ignore
    
    with open(f"{SAVE_PATH}/{test_name}/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder