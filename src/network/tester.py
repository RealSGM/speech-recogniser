import network.network as network
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import file_reader as fr

from sklearn import metrics
from utils.config import MainConfig


def plot_confusion_matrix(cm, encoder, accuracy, config: MainConfig):
    config_path = fr.get_config_path(config)

    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    
    plt.savefig(f"{config_path}/confusion_matrix.png")
    
    fr.add_to_results_file(config, {f"test_accuracy_{config.notes}:": f"{accuracy * 100:.2f}%"})


def test_network(super_vectors: np.ndarray, labels: list[str], config: MainConfig):
    print(f"\n=== Testing network: {config.test_name} ===")
    encoded_labels, encoder = network.encode_labels(labels, sorted(set(labels)))
    
    model, _ = network.load_model(config=config)
    predictions = model.predict(super_vectors, verbose=0)
    prediction_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(encoded_labels, axis=1)
    _ = encoder.inverse_transform(prediction_indices)
    accuracy = metrics.accuracy_score(actual_indices, prediction_indices)

    cm = metrics.confusion_matrix(actual_indices, prediction_indices)
    plot_confusion_matrix(cm, encoder, accuracy, config=config)