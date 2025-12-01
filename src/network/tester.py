import network.network as network
import matplotlib.pyplot as plt
import seaborn as sns
import os 


def plot_confusion_matrix(cm, encoder, test_name, accuracy, graph_path="data/graphs"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(f"{graph_path}/{test_name}"):
        os.makedirs(f"{graph_path}/{test_name}")
    
    plt.savefig(f"{graph_path}/{test_name}/confusion_matrix.png")
    
    # Save accuracy to text file
    with open(f"{graph_path}/{test_name}/testing.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


def test_network(testing_name: str, use_mel_filter_bank: bool, use_spectral_features: bool) -> None:
    print(f"\n=== Testing network: {testing_name} ===")
    # model, encoder = network.load_model(testing_name)
    # data, labels = fe.get_mfcc_dataset(fe.TESTING_PATH, fr.TESTING_PATH, include_noisy=True, use_spectral_features=use_spectral_features, use_mel_filter_bank=use_mel_filter_bank)
#     encoded_labels, _ = am.encode_labels(labels, sorted(set(labels)))
    
#     # Loop through all data and evaluate
#     predictions = model.predict(data, verbose=0)
#     prediction_indices = np.argmax(predictions, axis=1)
#     actual_indices = np.argmax(encoded_labels, axis=1)
#     predicted_labels = encoder.inverse_transform(prediction_indices)
#     accuracy = metrics.accuracy_score(actual_indices, prediction_indices)

#     cm = metrics.confusion_matrix(actual_indices, prediction_indices)
#     am.plot_confusion_matrix(cm, encoder, test_name, accuracy)
    pass