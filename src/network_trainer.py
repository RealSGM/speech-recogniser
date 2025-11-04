from dotenv import load_dotenv
load_dotenv() 

import time
import file_reader as fr
import feature_extractor as fe
import acoustic_modeller as am
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data, labels = fe.get_mfcc_dataset(fe.TRAINING_PATH, fr.TRAINING_DIRECTORY, include_noisy=True)
    
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
    model = am.create_model(input_shape=data.shape[1:], num_classes=len(encoder.classes_)) # type: ignore
    history = am.train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    am.evaluate_model(model, X_test, y_test, encoder)
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    am.save_model(model, encoder)
    am.plot_training_history(history)
    