import file_reader

if __name__ == "__main__":
    audio_data, labels = file_reader.load_dataset()
    print("Number of audio files loaded:", len(audio_data))