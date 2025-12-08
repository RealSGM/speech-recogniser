import network.network as network
import matplotlib.pyplot as plt
import numpy as np
import time
import file_reader as fr

from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils.config import MainConfig


def save_results(model, X_test, y_test, elapsed_time: float, config: MainConfig):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    
    # Add to JSON results file
    results = {
        f"training_accuracy_{config.notes}": f"{accuracy}",
        f"elapsed_time_{config.notes}": f"{elapsed_time:.2f}"
    }
    fr.add_to_results_file(config, results)


def plot_training_history(history, config: MainConfig):
    config_path = fr.get_config_path(config)
    
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
    plt.savefig(f"{config_path}/training_history_{config.notes}.png")


def run_network(vectors: np.ndarray, labels: list[str], config: MainConfig):
    print(f"==============================")
    print(f"    TRAINING NETWORK: {config.test_name}")
    print(f"==============================\n")
    
    encoded_labels, encoder = network.encode_labels(labels, sorted(set(labels)))
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        vectors,
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
    
    model = network.create_model(input_shape=vectors.shape[1:], num_classes=len(encoder.classes_), config=config)
    history = network.train_model(model, X_train, y_train, X_val, y_val, config)
    
    elapsed_time = time.time() - config.start_time
    
    save_results(model, X_test, y_test, elapsed_time, config=config)
    plot_training_history(history, config=config)
    network.save_model(model, encoder, config=config)