import file_reader 
import numpy as np

TRAINING = "training"
TESTING = "testing"


def build_super_vectors(audio: np.ndarray, visual: np.ndarray) -> np.ndarray:
	super = []

	for i in range(len(audio)):
		a = audio[i, :, :, 0]
		v = visual[i, :, :, 0]

		T_a = a.shape[0]
		T_v = v.shape[0]

		index_map = np.linspace(0, T_v - 1, T_a).astype(np.int32)
		v_aligned = v[index_map]

		fused = np.concatenate((a, v_aligned), axis=1)
		super.append(fused)

	super = np.array(super)
	super = np.expand_dims(super, -1)

	return super


def main() -> None:
    visual_data_train, audio_data_train, labels_train = file_reader.load_dataset(TRAINING)
    super_vectors_train = build_super_vectors(audio_data_train, visual_data_train)
    
    print(f"Training Visual Data Shape: {visual_data_train.shape}")
    print(f"Training Audio Data Shape: {audio_data_train.shape}")
    print(f"Training Super Vectors Shape: {super_vectors_train.shape}")
    

if __name__ == "__main__":
    main()