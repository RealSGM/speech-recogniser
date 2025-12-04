import file_reader as fr
import numpy as np
import os
import network.trainer as trainer
import network.tester as tester
import time
from utils.config import get_configs, MainConfig

TRAINING = "training"
TESTING = "testing"
VECTOR_PATH = "data/vectors"

LATE_RESULTS = "Late Integration Results:"

is_training   = True


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


def calculate_late_integration(config: MainConfig) -> None:
    audio_result, visual_result = tester.network.get_results_from_file(config)
    
    weighted_score = (config.integration.audio_weight * audio_result + 
        config.integration.visual_weight * visual_result) / \
        (config.integration.audio_weight + config.integration.visual_weight)
    fr.add_to_results_file(config, {"late_results:": f"{weighted_score * 100:.2f}%"})


def run_early_integration(config: MainConfig, audio_data_train: np.ndarray, visual_data_train: np.ndarray, labels_train: list[str], audio_data_test: np.ndarray, visual_data_test: np.ndarray, labels_test: list[str]) -> None:
    super_vectors_train = build_super_vectors(audio_data_train, visual_data_train)
    trainer.train_network(super_vectors_train, labels_train, config=config)
    # super_vectors_test = build_super_vectors(audio_data_test, visual_data_test)
    # tester.test_network(super_vectors_test, labels_test, config=config)


def run_late_integration(config: MainConfig, audio_data_train: np.ndarray, visual_data_train: np.ndarray, labels_train: list[str], audio_data_test: np.ndarray, visual_data_test: np.ndarray, labels_test: list[str]) -> None:
    if config.integration.audio_weight > 0:
        config.notes = "audio"
        trainer.train_network(audio_data_train, labels_train, config=config)
        # tester.test_network(audio_data_test, labels_test, config=duplicated_config)
        
    if config.integration.visual_weight > 0:
        config.notes = "visual"
        trainer.train_network(visual_data_train, labels_train, config=config)
        # tester.test_network(visual_data_test, labels_test, config=duplicated_config)
    
    calculate_late_integration(config)


def run_full_session() -> None:
    for config in get_configs():
        fr.create_config_path(config) ## Comment out if you want to keep previous results
        config.start_time = time.time()
        
        visual_data_train, audio_data_train, labels_train = fr.load_dataset(TRAINING, config)
        visual_data_test, audio_data_test, labels_test = fr.load_dataset(TESTING, config)
        
        match config.integration.integration_type:
            case "early":
                config.notes = "early"
                run_early_integration(config, audio_data_train, visual_data_train, labels_train, audio_data_test, visual_data_test, labels_test)
                
            case "late":
                run_late_integration(config, audio_data_train, visual_data_train, labels_train, audio_data_test, visual_data_test, labels_test)
            case _:
                print(f"Unknown integration type: {config.integration.integration_type}")


def main() -> None:
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    run_full_session()
    

if __name__ == "__main__":
    main()