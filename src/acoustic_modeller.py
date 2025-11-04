import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


SAVE_PATH = "data/networks"
MODEL_PATH = f"{SAVE_PATH}/speech_model.keras"
ENCODER_PATH = f"{SAVE_PATH}/label_encoder.pkl"


def encode_labels(labels, class_list):
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def create_model(input_shape, num_classes):
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

    adam = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam) # type: ignore
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
        callbacks=[learning_rate_reduction, early_stop]
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
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
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


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

# Save the model and encoder
def save_model(model, encoder):
    model.save(MODEL_PATH)
    
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)


# Load the model and encoder
def load_model() -> tf.keras.Model: # type: ignore
    model = tf.keras.models.load_model(MODEL_PATH) # type: ignore
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder