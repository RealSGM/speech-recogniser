import network.network as network
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils.config import MainConfig

GRAPH_PATH = "data/graphs"


def evaluate_model(model, X_test, y_test, elapsed_time: float, config: MainConfig):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)
    
    full_graph_path = f"{GRAPH_PATH}/{config.test_name}"
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(full_graph_path):
        os.makedirs(full_graph_path)
    
    with open(f"{full_graph_path}/training.txt", "w") as f:
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


def plot_training_history(history, config: MainConfig):
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
    
    full_graph_path = f"{GRAPH_PATH}/{config.test_name}"
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(full_graph_path):
        os.makedirs(full_graph_path)
        
    plt.savefig(f"{full_graph_path}/training_history.png")


def train_network(super_vectors: np.ndarray, labels: list[str], config: MainConfig):
    print(f"\n=== Training network: {config.test_name} ===")
    
    encoded_labels, encoder = network.encode_labels(labels, sorted(set(labels)))
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        super_vectors,
        encoded_labels,
        test_size=0.2,
        stratify=encoded_labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        stratify=y_tmp
    )
    
    start_time = time.time()
    model = network.create_model(input_shape=super_vectors.shape[1:], num_classes=len(encoder.classes_), config=config)
    history = network.train_model(model, X_train, y_train, X_val, y_val, config)
    
    elapsed_time = time.time() - start_time
    evaluate_model(model, X_test, y_test, elapsed_time, config=config)
    
    network.save_model(model, encoder, config=config)
    plot_training_history(history, config=config)