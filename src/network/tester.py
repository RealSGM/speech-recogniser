import network.network as network
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 

from sklearn import metrics
from utils.config import MainConfig


def plot_confusion_matrix(cm, encoder, accuracy, graph_path, config: MainConfig):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    full_graph_path = f"{graph_path}/{config.test_name}"
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(full_graph_path):
        os.makedirs(full_graph_path)
    
    plt.savefig(f"{full_graph_path}/confusion_matrix.png")
    
    # Save accuracy to text file
    with open(f"{full_graph_path}/testing.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


def test_network(super_vectors: np.ndarray, labels: list[str], graph_path: str, config: MainConfig):
    print(f"\n=== Testing network: {config.test_name} ===")
    encoded_labels, encoder = network.encode_labels(labels, sorted(set(labels)))
    
    model, _ = network.load_model(config=config)
    predictions = model.predict(super_vectors, verbose=0)
    prediction_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(encoded_labels, axis=1)
    _ = encoder.inverse_transform(prediction_indices)
    accuracy = metrics.accuracy_score(actual_indices, prediction_indices)

    cm = metrics.confusion_matrix(actual_indices, prediction_indices)
    plot_confusion_matrix(cm, encoder, accuracy, graph_path, config=config)