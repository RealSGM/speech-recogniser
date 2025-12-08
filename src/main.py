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

def build_super_vectors(audio: np.ndarray, visual: np.ndarray) -> np.ndarray:
    super_vectors = []

    for i in range(len(audio)):
        a = audio[i, :, :, 0]
        v = visual[i, :, :, 0]

        T_a = a.shape[0]
        T_v = v.shape[0]

        # Align visual sequence to audio length
        index_map = np.linspace(0, T_v - 1, T_a).astype(np.int32)
        v_aligned = v[index_map]

        fused = np.concatenate((a, v_aligned), axis=1)
        super_vectors.append(fused)

    super_vectors = np.array(super_vectors)
    super_vectors = np.expand_dims(super_vectors, -1)

    return super_vectors


def calculate_late_integration(config: MainConfig) -> None:
    # Read last saved results for audio/visual models
    audio_result, visual_result = tester.network.get_results_from_file(config)

    if (config.integration.audio_weight + config.integration.visual_weight) == 0:
        weighted_score = 0.0
    else:
        weighted_score = (
            config.integration.audio_weight * audio_result +
            config.integration.visual_weight * visual_result
        ) / (config.integration.audio_weight + config.integration.visual_weight)

    fr.add_to_results_file(config, {"late_results:": f"{weighted_score * 100:.2f}%"})


def run_early_integration(
    config: MainConfig,
    audio_train, visual_train, labels_train,
    audio_test, visual_test, labels_test
) -> None:

    print("\n=== EARLY INTEGRATION ===")
    config.notes = "early"

    super_train = build_super_vectors(audio_train, visual_train)
    trainer.train_network(super_train, labels_train, config=config)

    super_test = build_super_vectors(audio_test, visual_test)
    tester.test_network(super_test, labels_test, config=config)


def run_late_integration(
    config: MainConfig,
    audio_train, visual_train, labels_train,
    audio_test, visual_test, labels_test
) -> None:

    print("\n=== LATE INTEGRATION ===")

    # AUDIO BRANCH
    if config.integration.audio_weight > 0:
        print("\n--- Training AUDIO model ---")
        config.notes = "audio"
        trainer.train_network(audio_train, labels_train, config=config)
        tester.test_network(audio_test, labels_test, config=config)

    # VISUAL BRANCH
    if config.integration.visual_weight > 0:
        print("\n--- Training VISUAL model ---")
        config.notes = "visual"
        trainer.train_network(visual_train, labels_train, config=config)
        tester.test_network(visual_test, labels_test, config=config)

    # Compute weighted late-integration score
    calculate_late_integration(config)


def run_full_session() -> None:

    for config in get_configs():

        # Show test name clearly
        print(f"\n==============================")
        print(f" RUNNING CONFIG: {config.test_name}")
        print(f"==============================\n")

        # Create config folder
        fr.create_config_path(config)
        config.start_time = time.time()

        # Load datasets
        visual_data_train, audio_data_train, labels_train = fr.load_dataset(TRAINING, config)
        visual_data_test,  audio_data_test,  labels_test  = fr.load_dataset(TESTING, config)

        # Decide integration strategy
        match config.integration.integration_type:
            case "early":
                run_early_integration(
                    config,
                    audio_data_train, visual_data_train, labels_train,
                    audio_data_test,  visual_data_test,  labels_test,
                )

            case "late":
                run_late_integration(
                    config,
                    audio_data_train, visual_data_train, labels_train,
                    audio_data_test,  visual_data_test,  labels_test,
                )

            case _:
                print(f"Unknown integration type: {config.integration.integration_type}")


def main() -> None:
    # optional: clear terminal output
    os.system('cls' if os.name == 'nt' else 'clear')
    run_full_session()


if __name__ == "__main__":
    main()
    
