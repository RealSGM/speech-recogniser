import os
import subprocess
from time import sleep

NUM_SAMPLES = 10
DURATION = 2.0
SAVE_DIR = "data\\recordings\\training"

VIDEO = "HP 5MP Camera"
AUDIO = "Microphone Array (Intel® Smart Sound Technology for Digital Microphones)"

DSHOW = f"video={VIDEO}:audio={AUDIO}"

def record_av(name: str):
	name_dir = os.path.join(SAVE_DIR, name)
	os.makedirs(name_dir, exist_ok=True)

	print(f"\nRecording {NUM_SAMPLES} samples for: {name}")
	print(f"Each sample lasts {DURATION:.1f} seconds.\n")

	for i in range(1, NUM_SAMPLES + 1):
		print(f"Recording sample {i}/{NUM_SAMPLES}...")

		outfile = os.path.join(name_dir, f"{name}_{i+10:03d}.mp4")

		cmd = [
	"ffmpeg", "-y",
	"-f", "dshow",
	"-i", DSHOW,

	"-t", str(DURATION),

	"-pix_fmt", "yuv420p",
	"-c:v", "libx264",
	"-preset", "veryfast",

	"-c:a", "aac",
	"-b:a", "128k",

	"-movflags", "+faststart",

	outfile
    ]


		subprocess.run(cmd)

		print(f"Saved: {outfile}")
		print("✔ Recording accepted.\n")

	print(f"✔ All {NUM_SAMPLES} recordings complete.\n")


if __name__ == "__main__":
	print("Audio + Video Recorder")
	print("======================")

	while True:
		name = input("Enter name (or 'quit'): ").strip().lower()
		if name == "quit":
			break
		if not name:
			print("Invalid name.")
			continue

		record_av(name)

	print("\nSession finished.")
