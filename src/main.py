import file_reader 

TRAINING = "training"
TESTING = "testing"

def main() -> None:
    visual_data_train, audio_data_train, labels_train = file_reader.load_dataset(TRAINING)
    ## Process audio mfccs


if __name__ == "__main__":
    main()