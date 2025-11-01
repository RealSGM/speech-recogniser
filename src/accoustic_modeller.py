import numpy as np
import glob
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
import tensorflow as tf


def load_mfcc_data(mfcc_dir: str):
    mfcc_files = sorted(glob.glob(f'{mfcc_dir}/*.npy'))
    mfcc_data_list = [np.load(file) for file in mfcc_files]

    # Find maximum frame length
    max_frames = max(data.shape[1] for data in mfcc_data_list)

    # Pad all to equal length
    data = np.array([
        np.pad(d, ((0, 0), (0, max_frames - d.shape[1]))) for d in mfcc_data_list
    ])

    # Extract labels (e.g. from filenames like 'dave_001.npy')
    labels = np.array([Path(f).stem.split('_')[0] for f in mfcc_files])

    # Normalise
    data = data / np.max(data)

    print(f"Loaded {len(data)} samples from {mfcc_dir}")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

    return data, labels


def encode_labels(labels, class_list):
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def create_model(input_shape=(40, 21, 1), num_classes=10):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose="auto"
    )
    return history


def evaluate_model(model, X_test, y_test, encoder):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")

    cm = metrics.confusion_matrix(actual, predicted)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    print(metrics.classification_report(actual, predicted, target_names=encoder.classes_))


def predict_sample(model, X_sample, encoder):
    predicted_prob = model.predict(np.expand_dims(X_sample, axis=0), verbose=0)
    predicted_id = np.argmax(predicted_prob, axis=1)
    predicted_class = encoder.inverse_transform(predicted_id)
    print(f'Predicted class: {predicted_class[0]}')
    return predicted_class[0]