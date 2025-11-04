import numpy as np
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

SAVE_PATH = "data/networks"
GRAPH_PATH = "data/graphs"


def encode_labels(labels, class_list):
    encoder = LabelEncoder()
    encoder.fit(class_list)
    encoded = to_categorical(encoder.transform(labels))
    return encoded, encoder


def create_model(input_shape, num_classes, use_complex=False):
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


def evaluate_model(model, X_test, y_test, encoder, test_name: str = "", elapsed_time: float = 0.0):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)

    # cm = metrics.confusion_matrix(actual, predicted)
    # print("\nClassification Report:\n")
    # print(
    #     metrics.classification_report(
    #         actual, 
    #         predicted, 
    #         labels=range(len(encoder.classes_)),
    #         target_names=encoder.classes_,
    #         zero_division=0
    #     )
    # )
    
    # Save elapsed time and accuracy to file
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(f"{GRAPH_PATH}/{test_name}"):
        os.makedirs(f"{GRAPH_PATH}/{test_name}")
    
    with open(f"{GRAPH_PATH}/{test_name}/training.txt", "w") as f:
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


def predict_sample(model, X_sample, encoder):
    predicted_prob = model.predict(np.expand_dims(X_sample, axis=0), verbose=0)
    predicted_id = np.argmax(predicted_prob, axis=1)
    predicted_class = encoder.inverse_transform(predicted_id)
    print(f'Predicted class: {predicted_class[0]}')
    return predicted_class[0]


def plot_training_history(history, test_name):
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
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(f"{GRAPH_PATH}/{test_name}"):
        os.makedirs(f"{GRAPH_PATH}/{test_name}")
        
    plt.savefig(f"{GRAPH_PATH}/{test_name}/training_history.png")


def plot_confusion_matrix(cm, encoder, test_name, accuracy):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{GRAPH_PATH}/{test_name}/confusion_matrix.png")
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(f"{GRAPH_PATH}/{test_name}"):
        os.makedirs(f"{GRAPH_PATH}/{test_name}")
    
    # Save accuracy to text file
    with open(f"{GRAPH_PATH}/{test_name}/testing.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


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