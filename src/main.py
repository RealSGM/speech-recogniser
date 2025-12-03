import file_reader 
import numpy as np
import os
import shutil
import network.trainer as trainer
import network.tester as tester
from utils.config import get_configs

TRAINING = "training"
TESTING = "testing"
VECTOR_PATH = "data/vectors"
GRAPH_PATH = "data/graphs"
is_training   = True


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
    for config in get_configs():
        if is_training:
            delete_vector_files()
            
        visual_data_train, audio_data_train, labels_train = file_reader.load_dataset(TRAINING, config)
        super_vectors_train = build_super_vectors(audio_data_train, visual_data_train)
        
        print(f"Training Visual Data Shape: {visual_data_train.shape}")
        print(f"Training Audio Data Shape: {audio_data_train.shape}")
        print(f"Training Super Vectors Shape: {super_vectors_train.shape}")
        print(f"Training Labels Shape: {len(labels_train)}")
        
        if is_training:
            trainer.train_network(super_vectors_train, labels_train, config=config)
        else:
            tester.test_network(super_vectors_train, labels_train, GRAPH_PATH, config=config)


if __name__ == "__main__":
    main()