from dotenv import load_dotenv
load_dotenv() 

import file_reader as fr
import feature_extractor as fe
import acoustic_modeller as am
from sklearn import metrics


if __name__ == "__main__":
    model, encoder = am.load_model()
    
    data, labels = fe.get_mfcc_dataset(fe.TESTING_PATH, fr.TESTING_DIRECTORY)
    encoded_labels, _ = am.encode_labels(labels, sorted(set(labels)))
    
    # Loop through all data and evaluate
    predictions = model.predict(data, verbose=0)
    prediction_indices = np.argmax(predictions, axis=1)
    actual_indices = np.argmax(encoded_labels, axis=1)
    predicted_labels = encoder.inverse_transform(prediction_indices)
    
    accuracy = metrics.accuracy_score(actual_indices, prediction_indices)
    
    print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

    print("\n Classification Report:\n")
    print(
        metrics.classification_report(
            actual_indices,
            prediction_indices,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    cm = metrics.confusion_matrix(actual_indices, prediction_indices)
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_).plot()