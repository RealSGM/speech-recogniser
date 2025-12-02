import file_reader 
import numpy as np
import os
import shutil

import network.network as network
import network.trainer as trainer
import network.tester as tester

TRAINING = "training"
TESTING = "testing"
VECTOR_PATH = "data/vectors"

delete_vectors = True

# test_configurations = [
#     # {
#     #     "test_name": "baseline",
#     #     "use_complex_model": False,
#     #     "use_mel_filter_bank": False,
#     #     "use_spectral_features": False,
#     #     "include_noisy_data": False
#     # },
#     # {
#     #     "test_name": "complex_model",
#     #     "use_complex_model": True,
#     #     "use_mel_filter_bank": False,
#     #     "use_spectral_features": False,
#     #     "include_noisy_data": False
#     # },
#     # {
#     #     "test_name": "mel_filter_bank",
#     #     "use_complex_model": False,
#     #     "use_mel_filter_bank": True,
#     #     "use_spectral_features": False,
#     #     "include_noisy_data": False
#     # },
#     # {
#     #     "test_name": "spectral_features",
#     #     "use_complex_model": False,
#     #     "use_mel_filter_bank": False,
#     #     "use_spectral_features": True,
#     #     "include_noisy_data": False
#     # },
#     # {
#     #     "test_name": "noisy_data",
#     #     "use_complex_model": False,
#     #     "use_mel_filter_bank": False,
#     #     "use_spectral_features": False,
#     #     "include_noisy_data": True
#     # },
#     {
#         "test_name": "all_features_noisy_testing",
#         "use_complex_model": True,
#         "use_mel_filter_bank": True,
#         "use_spectral_features": True,
#         "include_noisy_data": True
#     },
# ]

def delete_vector_files() -> None:
    if os.path.exists(VECTOR_PATH):
        shutil.rmtree(VECTOR_PATH)
        print(f"Deleted existing vector files in {VECTOR_PATH}")


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
    if delete_vectors:
        delete_vector_files()
        
    visual_data_train, audio_data_train, labels_train = file_reader.load_dataset(TRAINING, use_mel_filter_bank=True, use_spectral_features=False)
    super_vectors_train = build_super_vectors(audio_data_train, visual_data_train)
    
    print(f"Training Visual Data Shape: {visual_data_train.shape}")
    print(f"Training Audio Data Shape: {audio_data_train.shape}")
    print(f"Training Super Vectors Shape: {super_vectors_train.shape}")
    print(f"Training Labels Shape: {len(labels_train)}")
    
    trainer.train_network(super_vectors_train, labels_train, "test_run", use_complex_model=True)
    

if __name__ == "__main__":
    main()