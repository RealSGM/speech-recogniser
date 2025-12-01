import network.network as network
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os


def evaluate_model(model, X_test, y_test, encoder, test_name: str = "", graph_path: str = "", elapsed_time: float = 0.0):
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)
    
    # Create graph directory if it doesn't exist
    if not os.path.exists(f"{graph_path}/{test_name}"):
        os.makedirs(f"{graph_path}/{test_name}")
    
    with open(f"{graph_path}/{test_name}/training.txt", "w") as f:
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")


def plot_training_history(history, graph_path, test_name):
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
    if not os.path.exists(f"{graph_path}/{test_name}"):
        os.makedirs(f"{graph_path}/{test_name}")
        
    plt.savefig(f"{graph_path}/{test_name}/training_history.png")


def train_network(test_name: str, use_complex_model: bool, use_mel_filter_bank: bool, use_spectral_features: bool, include_noisy_data: bool):
    print(f"\n=== Training network: {test_name} ===")
#     data, labels = fe.get_mfcc_dataset(fe.TRAINING_PATH, fr.TRAINING_PATH, include_noisy=include_noisy_data, use_spectral_features=use_spectral_features, use_mel_filter_bank=use_mel_filter_bank)
#     encoded_labels, encoder = am.encode_labels(labels, sorted(set(labels)))
    
#     X_train, X_tmp, y_train, y_tmp = train_test_split(
#         data,
#         encoded_labels,
#         test_size=0.2,
#         stratify=encoded_labels
#     )

#     X_val, X_test, y_val, y_test = train_test_split(
#         X_tmp,
#         y_tmp,
#         test_size=0.5,
#         stratify=y_tmp
#     )
    
#     start_time = time.time()
#     model = am.create_model(input_shape=data.shape[1:], num_classes=len(encoder.classes_), use_complex=use_complex_model) # type: ignore
#     history = am.train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    
#     elapsed_time = time.time() - start_time
#     am.evaluate_model(model, X_test, y_test, encoder, test_name, elapsed_time)
    
#     am.save_model(model, encoder, test_name)
#     plot_training_history(history, test_name)
    pass