import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def encode_labels(labels, class_list):
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def create_model(input_shape=(40, 21, 1), num_classes=10):
    model = Sequential()
    model.add(InputLayer(shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
    # Stop early if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    learning_rate_reduction = ReduceLROnPlateau(
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
        callbacks=[early_stop]
    )
    return history


def evaluate_model(model, X_test, y_test, encoder):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)
    cm = metrics.confusion_matrix(actual, predicted)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    print("\nClassification Report:\n")
    print(
        metrics.classification_report(
            actual, 
            predicted, 
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )


def predict_sample(model, X_sample, encoder):
    predicted_prob = model.predict(np.expand_dims(X_sample, axis=0), verbose=0)
    predicted_id = np.argmax(predicted_prob, axis=1)
    predicted_class = encoder.inverse_transform(predicted_id)
    print(f'Predicted class: {predicted_class[0]}')
    return predicted_class[0]