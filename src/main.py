import time
import os
import shutil
import audio.file_reader as fr
import audio.feature_extractor as fe
import audio.acoustic_modeller as am
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

delete_mfccs_on_startup = True

test_configurations = [
    # {
    #     "test_name": "baseline",
    #     "use_complex_model": False,
    #     "use_mel_filter_bank": False,
    #     "use_spectral_features": False,
    #     "include_noisy_data": False
    # },
    # {
    #     "test_name": "complex_model",
    #     "use_complex_model": True,
    #     "use_mel_filter_bank": False,
    #     "use_spectral_features": False,
    #     "include_noisy_data": False
    # },
    # {
    #     "test_name": "mel_filter_bank",
    #     "use_complex_model": False,
    #     "use_mel_filter_bank": True,
    #     "use_spectral_features": False,
    #     "include_noisy_data": False
    # },
    # {
    #     "test_name": "spectral_features",
    #     "use_complex_model": False,
    #     "use_mel_filter_bank": False,
    #     "use_spectral_features": True,
    #     "include_noisy_data": False
    # },
    # {
    #     "test_name": "noisy_data",
    #     "use_complex_model": False,
    #     "use_mel_filter_bank": False,
    #     "use_spectral_features": False,
    #     "include_noisy_data": True
    # },
    {
        "test_name": "all_features_noisy_testing",
        "use_complex_model": True,
        "use_mel_filter_bank": True,
        "use_spectral_features": True,
        "include_noisy_data": True
    },
]


def delete_mfcc_files():
    print("Deleting existing MFCC files...")
    # Delete all folders in training and testing paths
    for path in [fe.TRAINING_PATH, fe.TESTING_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)


def test_network(testing_name: str, use_mel_filter_bank: bool, use_spectral_features: bool):
    print(f"\n=== Testing network: {testing_name} ===")
    
    model, encoder = am.load_model(testing_name)
    data, labels = fe.get_mfcc_dataset(fe.TESTING_PATH, fr.TESTING_PATH, include_noisy=True, use_spectral_features=use_spectral_features, use_mel_filter_bank=use_mel_filter_bank)
    encoded_labels, _ = am.encode_labels(labels, sorted(set(labels)))
    
    # Loop through all data and evaluate
    predictions = model.predict(data, verbose=0)
    prediction_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(encoded_labels, axis=1)
    predicted_labels = encoder.inverse_transform(prediction_indices)
    accuracy = metrics.accuracy_score(actual_indices, prediction_indices)

    cm = metrics.confusion_matrix(actual_indices, prediction_indices)
    am.plot_confusion_matrix(cm, encoder, test_name, accuracy)


def train_network(test_name: str, use_complex_model: bool, use_mel_filter_bank: bool, use_spectral_features: bool, include_noisy_data: bool):
    print(f"\n=== Training network: {test_name} ===")
    
    data, labels = fe.get_mfcc_dataset(fe.TRAINING_PATH, fr.TRAINING_PATH, include_noisy=include_noisy_data, use_spectral_features=use_spectral_features, use_mel_filter_bank=use_mel_filter_bank)
    encoded_labels, encoder = am.encode_labels(labels, sorted(set(labels)))
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data,
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
    model = am.create_model(input_shape=data.shape[1:], num_classes=len(encoder.classes_), use_complex=use_complex_model) # type: ignore
    history = am.train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    
    elapsed_time = time.time() - start_time
    am.evaluate_model(model, X_test, y_test, encoder, test_name, elapsed_time)
    
    am.save_model(model, encoder, test_name)
    am.plot_training_history(history, test_name)
    

if __name__ == "__main__":
    for config in test_configurations:
        # delete_mfcc_files()
        test_name = config['test_name']
        use_complex_model = config['use_complex_model']
        use_mel_filter_bank = config['use_mel_filter_bank']
        use_spectral_features = config['use_spectral_features']
        include_noisy_data = config['include_noisy_data']
        
        # train_network(test_name, use_complex_model, use_mel_filter_bank, use_spectral_features, include_noisy_data)
        test_network(test_name, use_mel_filter_bank, use_spectral_features)