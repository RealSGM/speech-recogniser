from dotenv import load_dotenv
load_dotenv() 

from feature_extractor import get_mfcc_dataset
from sklearn.model_selection import train_test_split
import acoustic_modeller as am
import numpy as np


if __name__ == "__main__":
    model, encoder = am.load_model()
    
    data, labels = get_mfcc_dataset()
    encoded_labels, _ = am.encode_labels(labels, sorted(set(labels)))
    
    X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2)
    am.evaluate_model(model, X_test, y_test, encoder)