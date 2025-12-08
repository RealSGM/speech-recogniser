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


def calculate_late_integration(config: MainConfig, key: str) -> None:
    # Read last saved results for audio/visual models
    audio_result, visual_result = tester.network.get_results_from_file(config, key)
    
    audio_result = float(audio_result)
    visual_result = float(visual_result)

    if (config.integration.audio_weight + config.integration.visual_weight) == 0:
        weighted_score = 0.0
    else:
        weighted_score = (
            config.integration.audio_weight * audio_result +
            config.integration.visual_weight * visual_result
        ) / (config.integration.audio_weight + config.integration.visual_weight)

    fr.add_to_results_file(config, {f"late_integration_{key}_results:": f"{weighted_score * 100:.2f}%"})


def run_network(caller, cfg: MainConfig, PATH: str) -> None:
    visual_data, audio_data, labels = fr.load_dataset(PATH, cfg)
    
    match cfg.integration.integration_type:
        case "early":
            super_train = build_super_vectors(audio_data, visual_data)
            caller.run_network(super_train, labels, config=cfg)
        case "late":
            if cfg.integration.audio_weight > 0:
                cfg.notes = "audio"
                caller.run_network(audio_data, labels, config=cfg)
            if cfg.integration.visual_weight > 0:
                cfg.notes = "visual"
                caller.run_network(visual_data, labels, config=cfg)
            calculate_late_integration(cfg, PATH.lower())
        case _:
            print(f"Unknown integration type: {cfg.integration.integration_type}")


def run_full_session() -> None:
    for config in get_configs():
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"==============================")
        print(f" RUNNING CONFIG: {config.test_name}")
        print(f"==============================\n")

        # Create config folder
        # fr.create_config_path(config)
        config.start_time = time.time()

        # run_network(trainer, config, TRAINING)
        run_network(tester, config, TESTING)


def main() -> None:
    run_full_session()


if __name__ == "__main__":
    main()
