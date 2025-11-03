from dotenv import load_dotenv
load_dotenv() 

from feature_extractor import get_mfcc_dataset
from sklearn.model_selection import train_test_split
import accoustic_modeller as am


if __name__ == "__main__":
    data, labels = get_mfcc_dataset()
    print(f"Total MFCC samples loaded: {len(data)}")
    print(f"Data shape: {data.shape}, Labels shape: {len(labels)}")
    
    encoded_labels, encoder = am.encode_labels(labels, sorted(set(labels)))
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        data,
        encoded_labels,
        test_size=0.2,
        random_state=0,
        stratify=encoded_labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.5,
        random_state=0,
        stratify=y_tmp
    )
    
    model = am.create_model(input_shape=data.shape[1:], num_classes=len(encoder.classes_))
    history = am.train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=32)
    am.evaluate_model(model, X_test, y_test, encoder)