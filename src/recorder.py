import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from time import sleep

SAMPLE_RATE = 16000        
DURATION = 2.0             
NUM_SAMPLES = 10        
SAVE_DIR = "data/test_recordings" 


def record_audio(name: str):
    # Make name directory
    name_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(name_dir, exist_ok=True)

    print(f"\nRecording {NUM_SAMPLES} samples for: {name}")
    print(f"Each sample is {DURATION:.1f} seconds long.\n")

    i = 1
    while i <= NUM_SAMPLES:
        #input(f"Press ENTER to record sample {i}/{NUM_SAMPLES}...")

        sleep(0.75)
        print("Recording...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print("Done recording.")

        # Convert to PCM
        audio_int16 = np.int16(audio * 32767)

        filename = os.path.join(name_dir, f"{name}_{i:03d}.wav")

        # Save file
        write(filename, SAMPLE_RATE, audio_int16)
        print(f"Saved: {filename}")

        # Ask user if recording is OK
        #choice = input("Keep this recording? (ENTER = keep, r = redo): ").strip().lower()

        # if choice == "r":
        #     print("Redoing...")
        #     os.remove(filename)
        #     continue  # redo same index

        print("✅ Recording accepted.\n")
        i += 1  # move to next sample
        sleep(0.5)

    print(f"✅ Completed all {NUM_SAMPLES} recordings for '{name}'.\n")


if __name__ == "__main__":
    print("Speech Dataset Recorder")
    print("========================")

    while True:
        name = input("\nEnter a name to record (or 'quit' to exit): ").strip().lower()
        if name == "quit":
            break
        if not name:
            print("Please enter a valid name.")
            continue

        record_audio(name)

    print("\n✅ Recording session finished.")
